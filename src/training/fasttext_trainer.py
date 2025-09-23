#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import fasttext


def eprint(*a, **k): print(*a, file=sys.stderr, **k)


def print_results(N, p, r):
    print(f"Samples: {N}")
    print(f"P@1: {p:.4f}")
    print(f"R@1: {r:.4f}")


def dump_args_json(model, out_json_path: str):
    try:
        args_obj = model.f.getArgs()  # internal but widely used
        kv = {}
        for name in dir(args_obj):
            if not name.startswith("_"):
                try:
                    val = getattr(args_obj, name)
                    if name == "loss" and hasattr(val, "name"):
                        val = val.name
                    kv[name] = val
                except Exception:
                    pass
        Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(kv, f, ensure_ascii=False, indent=2)
        print(f"Wrote tuned args to {out_json_path}")
    except Exception as e:
        eprint(f"[warn] Could not dump tuned args: {e}")


def check_file(p: str):
    if not Path(p).is_file():
        eprint(f"[error] Missing file: {p}");
        sys.exit(2)


def infer_label_prefix(sample_file: str, default="__label__"):
    with open(sample_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(default):
                return default
            break
    eprint(f"[fatal] Labels in {sample_file} must start with '__label__'. Example: '__label__en Hello'");
    sys.exit(2)


def scan_dataset(file_path: str, label_prefix="__label__", top_k=15):
    counts = Counter()
    n = 0
    total_len = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            n += 1
            total_len += len(s)
            # first token that starts with prefix
            for tok in s.split():
                if tok.startswith(label_prefix):
                    counts[tok] += 1
                    break
    avg_len = (total_len / max(n, 1)) if n else 0
    print(f"\n[scan] {file_path}")
    print(f"  lines: {n:,}")
    print(f"  avg_line_len_chars: {avg_len:.1f}")
    print(f"  classes: {len(counts)}")
    for lb, c in counts.most_common(top_k):
        print(f"    {lb:>22} : {c:,}")
    if len(counts) > top_k:
        print(f"    ... (+{len(counts) - top_k} more)")


def summarize_label_distribution(model, file_path: str, top_k=15):
    try:
        stats = model.test_label(file_path)  # dict[label] -> {n, precision, recall}
    except Exception:
        eprint("[warn] fastText build without test_label(); skipping per-label stats.")
        return
    rows = []
    for lb, m in stats.items():
        rows.append((lb, m.get("n", 0), m.get("precision", 0.0), m.get("recall", 0.0)))
    rows.sort(key=lambda x: (-x[1], x[0]))
    print("\nPer-label metrics (validation):")
    for lb, n, p, r in rows[:top_k]:
        print(f"{lb:>22}  n={n:<6}  P={p:.3f}  R={r:.3f}")
    if len(rows) > top_k:
        print(f"... (+{len(rows) - top_k} more)")


def dump_predictions(model, infile: str, outfile: str, k: int = 1, sample: int = 10, label_prefix="__label__"):
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    shown = 0
    total = 0
    correct = 0
    with open(infile, "r", encoding="utf-8", errors="ignore") as fin, \
            open(outfile, "w", encoding="utf-8") as fout:
        fout.write("gold\tpred\tprob\ttext\n")
        for line in fin:
            s = line.rstrip("\n")
            if not s:
                continue
            total += 1
            # parse gold label and text
            toks = s.split()
            gold = None
            start_idx = 0
            for i, t in enumerate(toks):
                if t.startswith(label_prefix):
                    gold = t;
                    start_idx = i + 1;
                    break
            text = " ".join(toks[start_idx:]) if start_idx < len(toks) else ""
            pred, prob = model.predict(text, k=k)
            p = pred[0] if pred else ""
            pr = float(prob[0]) if prob else 0.0
            if p == gold:
                correct += 1
            fout.write(f"{gold or ''}\t{p}\t{pr:.6f}\t{text}\n")
            # log a small sample to stdout
            if shown < sample:
                print(f"[pred] gold={gold} pred={p} prob={pr:.3f} :: {text[:120]}")
                shown += 1
    acc = correct / total if total else 0.0
    print(f"[pred] Wrote predictions to {outfile}  (N={total}, acc@1={acc:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/fasttext/train.txt")
    ap.add_argument("--valid", default="data/fasttext/valid.txt")
    ap.add_argument("--test", default="data/fasttext/test.txt")
    ap.add_argument("--output", default="data/fasttext/langclf")

    # Safer defaults for supervised langID
    ap.add_argument("--epoch", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.4)  # was 0.8; dialed down
    ap.add_argument("--wordNgrams", type=int, default=1)
    ap.add_argument("--minn", type=int, default=2)
    ap.add_argument("--maxn", type=int, default=6)
    ap.add_argument("--dim", type=int, default=300)
    ap.add_argument("--bucket", type=int, default=20_000_000)
    ap.add_argument("--loss", default="ova", choices=["softmax", "hs", "ova"])
    ap.add_argument("--thread", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--minCount", type=int, default=1)
    ap.add_argument("--lrUpdateRate", type=int, default=100)
    ap.add_argument("--verbose", type=int, default=3)  # more console progress

    # Autotune
    ap.add_argument("--autotune", action="store_true")
    ap.add_argument("--autotune_duration", type=int, default=900)
    ap.add_argument("--autotune_metric", default="f1")
    ap.add_argument("--autotune_predictions", type=int, default=1)
    ap.add_argument("--autotune_model_size", default=None)

    # Logging/explainability
    ap.add_argument("--scan_files", action="store_true", help="Print class stats for train/valid/test before training")
    ap.add_argument("--per_label_valid", action="store_true", help="Per-label P/R after training")
    ap.add_argument("--save_valid_preds", default=None, help="Path to write validation predictions TSV")
    ap.add_argument("--save_test_preds", default=None, help="Path to write test predictions TSV")
    ap.add_argument("--sample_pred_log", type=int, default=10, help="How many predictions to print to stdout per split")

    args = ap.parse_args()

    # IO sanity
    for p in (args.train, args.valid, args.test):
        check_file(p)
    label_prefix = infer_label_prefix(args.train)

    # Pre-train scans
    if args.scan_files:
        scan_dataset(args.train, label_prefix)
        scan_dataset(args.valid, label_prefix)
        scan_dataset(args.test, label_prefix)

    # Train
    print("Training model...")
    train_common = dict(verbose=args.verbose, thread=args.thread)
    if args.autotune:
        model = fasttext.train_supervised(
            input=args.train,
            autotuneValidationFile=args.valid,
            autotuneDuration=args.autotune_duration,
            autotuneMetric=args.autotune_metric,
            autotunePredictions=args.autotune_predictions,
            autotuneModelSize=args.autotune_model_size,
            **train_common
        )
    else:
        model = fasttext.train_supervised(
            input=args.train,
            epoch=args.epoch,
            lr=args.lr,
            wordNgrams=args.wordNgrams,
            minn=args.minn,
            maxn=args.maxn,
            dim=args.dim,
            bucket=args.bucket,
            loss=args.loss,
            minCount=args.minCount,
            lrUpdateRate=args.lrUpdateRate,
            **train_common
        )

    out = f"{args.output}.bin"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(out)
    print(f"Model saved to {out}")

    if args.autotune:
        dump_args_json(model, f"{args.output}.args.json")

    # Eval
    print("\nValidation results:")
    print_results(*model.test(args.valid))
    if args.per_label_valid:
        summarize_label_distribution(model, args.valid, top_k=20)

    if args.save_valid_preds:
        dump_predictions(model, args.valid, args.save_valid_preds, k=1, sample=args.sample_pred_log,
                         label_prefix=label_prefix)

    print("\nTest results:")
    print_results(*model.test(args.test))
    if args.save_test_preds:
        dump_predictions(model, args.test, args.save_test_preds, k=1, sample=args.sample_pred_log,
                         label_prefix=label_prefix)


if __name__ == "__main__":
    main()

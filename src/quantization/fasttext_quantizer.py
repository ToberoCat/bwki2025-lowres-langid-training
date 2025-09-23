#!/usr/bin/env python3
import argparse, os, fasttext, json, sys, re

LABEL_RX = re.compile(r"__label__\S+")

def human(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0: return f"{n:,.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"

def file_size(path): return human(os.path.getsize(path)) if os.path.exists(path) else "N/A"

def print_results(title, res):
    if not res: return
    N, p, r = res
    print(f"{title}\n  Samples: {N}\n  P@1: {p:.4f}\n  R@1: {r:.4f}")

def count_labels(train_path: str|None) -> int:
    if not train_path or not os.path.exists(train_path): return 0
    labs=set()
    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            labs.update(LABEL_RX.findall(line))
    return len(labs)

def main():
    ap = argparse.ArgumentParser(description="Quantize a trained fastText supervised model")
    ap.add_argument("--model", required=True, help="Path to trained .bin (supervised)")
    ap.add_argument("--train", default=None, help="Training .txt (required if --retrain)")
    ap.add_argument("--output", required=True, help="Output path prefix (no extension)")
    ap.add_argument("--valid", default=None)
    ap.add_argument("--test", default=None)

    # knobs
    ap.add_argument("--qnorm", dest="qnorm", action="store_true", default=True)
    ap.add_argument("--no-qnorm", dest="qnorm", action="store_false")
    ap.add_argument("--qout", dest="qout", action="store_true", default=True)
    ap.add_argument("--no-qout", dest="qout", action="store_false")
    ap.add_argument("--retrain", dest="retrain", action="store_true", default=True)
    ap.add_argument("--no-retrain", dest="retrain", action="store_false")

    ap.add_argument("--cutoff", type=int, default=200_000)
    ap.add_argument("--dsub", type=int, default=2)
    ap.add_argument("--epoch", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--thread", type=int, default=None)

    # allow “pad vocab to 256” if you really want input-quant on tiny vocabs
    ap.add_argument("--pad-vocab", action="store_true", default=False,
                    help="If vocab<256 and --retrain, pad training text with dummy tokens to reach 256 unique tokens")

    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"Missing model: {args.model}", file=sys.stderr); sys.exit(1)
    if args.retrain and not args.train:
        print(f"--retrain requested but --train path is missing.", file=sys.stderr); sys.exit(1)
    if args.retrain and args.train and not os.path.exists(args.train):
        print(f"Missing train file for retrain: {args.train}", file=sys.stderr); sys.exit(1)

    print(f"Loading model: {args.model}  ({file_size(args.model)})")
    model = fasttext.load_model(args.model)

    # Estimate matrix sizes
    dim = int(model.get_dimension())
    # input rows ~ #words + buckets; buckets not directly accessible, use a lower bound = vocab size
    vocab_size = model.get_words(include_freq=False).__len__()
    label_count = count_labels(args.train) if args.train else len(model.get_labels())

    can_quant_input  = vocab_size >= 256
    can_quant_output = label_count >= 256

    # Optional vocab padding to reach 256 for input-quant
    padded_train = None
    if args.retrain and args.pad_vocab and not can_quant_input:
        need = 256 - vocab_size
        if need > 0 and args.train:
            padded_train = f"{args.train}.padded.tmp"
            with open(args.train, "r", encoding="utf-8", errors="ignore") as src, \
                 open(padded_train, "w", encoding="utf-8") as dst:
                for line in src: dst.write(line)
                # Add one dummy unlabeled line per needed token
                for i in range(need):
                    # 32 distinct dummy tokens per line to keep lines short-ish
                    toks = [f"__pad_tok_{i}_{j}__" for j in range(32)]
                    dst.write(" ".join(toks) + "\n")
            args.train = padded_train
            can_quant_input = True  # will be ≥256 after retrain

    # If neither side can be quantized, bail out gracefully
    if not can_quant_input and not can_quant_output:
        print("Info: Neither input (vocab<256) nor output (labels<256) matrices meet the 256-row minimum.", file=sys.stderr)
        print("→ Skipping quantization. Consider training a micro profile (e.g., dim=32, smaller bucket/minn/maxn).", file=sys.stderr)
        sys.exit(4)

    # Gate flags to avoid triggering the error in fastText core
    qnorm = args.qnorm and can_quant_input
    qout  = args.qout  and can_quant_output

    # Sanity: dsub must divide dim if we do any PQ on input/output vectors
    if (qnorm or qout) and (dim % args.dsub != 0):
        print(f"Adjusting dsub from {args.dsub} to 1 because dim={dim} is not divisible.", file=sys.stderr)
        args.dsub = 1

    # Pre-quant eval
    if args.valid: print_results("Pre-quantization (valid)", model.test(args.valid))
    if args.test:  print_results("Pre-quantization (test)",  model.test(args.test))

    print("\nQuantizing…")
    kwargs = dict(
        input=(args.train if args.retrain else None),
        qout=qout,
        retrain=args.retrain,
        cutoff=max(1, int(args.cutoff)),
        dsub=int(args.dsub),
        epoch=(args.epoch if args.retrain else None),
        lr=(args.lr if args.retrain else None),
        thread=args.thread,
        verbose=2,
    )
    # fastText Python binding: qnorm is implied via internal path when input is quantized;
    # we effectively control input-quant by allowing retrain/cutoff and having enough rows.

    # If qnorm is False (can’t quantize input), we still can quantize output via qout=True.
    model.quantize(**kwargs)

    out_bin = f"{args.output}.ftz"
    model.save_model(out_bin)
    print(f"Saved quantized model: {out_bin}  ({file_size(out_bin)})")

    if args.valid: print_results("Post-quantization (valid)", model.test(args.valid))
    if args.test:  print_results("Post-quantization (test)",  model.test(args.test))

    # Dump effective args
    try:
        args_obj = model.f.getArgs()  # type: ignore[attr-defined]
        kv = {}
        for name in dir(args_obj):
            if not name.startswith("_"):
                try:
                    val = getattr(args_obj, name)
                    if name == "loss" and hasattr(val, "name"): val = val.name
                    kv[name] = val
                except Exception: pass
        with open(f"{args.output}.quant.args.json", "w", encoding="utf-8") as f:
            json.dump(kv, f, ensure_ascii=False, indent=2)
        print(f"Wrote quantization args to {args.output}.quant.args.json")
    except Exception as e:
        print(f"[warn] Could not dump quant args: {e}")

    print(f"\nSize before: {file_size(args.model)}  |  after: {file_size(out_bin)}")

    # cleanup
    if padded_train and os.path.exists(padded_train):
        try: os.remove(padded_train)
        except Exception: pass

if __name__ == "__main__":
    main()

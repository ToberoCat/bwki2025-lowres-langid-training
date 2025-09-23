import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}TB"


def get_vocab_tokens(path: Path) -> Tuple[int, int]:
    vocab = Counter()
    tokens = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            toks = _WORD_RE.findall(line)
            tokens += len(toks)
            vocab.update(toks)
    return len(vocab), tokens


def pick_dsub(dim: int, forced: int | None) -> int:
    if forced:
        return forced if dim % forced == 0 else 1
    for d in (4, 8, 6, 5, 10, 3, 2):
        if dim % d == 0:
            return d
    return 1


def profile_by_vocab(v: int) -> Tuple[str, int, int, int, int, int]:
    if v < 128:
        return "micro", 32, 20000, 2, 3, 1
    elif v < 256:
        return "tiny", 64, 50000, 2, 4, 1
    elif v < 5000:
        return "small", 100, 100000, 2, 5, 2
    elif v < 50000:
        return "med", 200, 200000, 2, 5, 2
    else:
        return "large", 300, 500000, 2, 6, 2


def cutoff_safe(desired: int | None, vocab: int) -> int:
    if vocab < 1:
        return 1
    if desired is None:
        return min(vocab, 1_000_000_000)
    if desired < 1:
        return 1
    return min(desired, vocab)


def discover_scripts(eval_root: Path) -> List[str]:
    train_dir = eval_root / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"ERROR: Missing required dir: {train_dir}")
    scripts = set()
    for p in train_dir.glob("*_train.txt"):
        fname = p.name
        m = re.match(r"^([A-Za-z]{4})_.*", fname)
        if m:
            scripts.add(m.group(1))
    return sorted(scripts)


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_root", nargs="?", default="artifacts/splitParquet/12_fasttext_files")
    ap.add_argument("out_root", nargs="?", default="data/fasttext_experts")
    ap.add_argument("--epochs", type=int, default=int(sys.argv_env.get("EPOCHS", 5)) if hasattr(sys, "argv_env") else 5)
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--quantize", type=int,
                    default=int(sys.argv_env.get("QUANTIZE", 1)) if hasattr(sys, "argv_env") else 1)
    ap.add_argument("--cutoff", type=int,
                    default=int(sys.argv_env.get("CUTOFF", 300000)) if hasattr(sys, "argv_env") else 300000)
    ap.add_argument("--dsub", type=int, default=int(sys.argv_env.get("DSUB", 0)) if hasattr(sys, "argv_env") else 0)
    ap.add_argument("--qextra-args", default="")
    ap.add_argument("--force-quantize", type=int,
                    default=int(sys.argv_env.get("FORCE_QUANTIZE", 1)) if hasattr(sys, "argv_env") else 1)
    ap.add_argument("--trainer", default="src/training/fasttext_trainer.py")
    ap.add_argument("--quantizer", default="src/quantization/fasttext_quantizer.py")
    ap.add_argument("--uv", default="uv")
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for sub in ("train", "valid", "test"):
        d = eval_root / sub
        if not d.is_dir():
            print(f"ERROR: Missing required dir: {d}")
            raise SystemExit(1)

    scripts = discover_scripts(eval_root)
    if not scripts:
        raise SystemExit(f"ERROR: No *_train.txt under {eval_root}/train")

    print(f"Discovered scripts: {' '.join(scripts)}")
    print(f"Output root: {out_root}")

    missing_any = False
    q_fail_any = False

    for script in scripts:
        print(f"\n=== [{script}] ===")
        train = eval_root / "train" / f"{script}_train.txt"
        valid = eval_root / "valid" / f"{script}_valid.txt"
        test = eval_root / "test" / f"{script}_test.txt"
        valid_preds = out_root / script / "langclf" / "preds_valid.tsv"
        test_preds = out_root / script / "langclf" / "preds_test.tsv"

        ok = True
        if not train.is_file():
            print(f"WARN: Missing {train}")
            ok = False
        if not valid.is_file():
            print(f"WARN: Missing {valid}")
            ok = False
        if not test.is_file():
            print(f"WARN: Missing {test}")
            ok = False
        if not ok:
            print(f"WARN: Skipping [{script}] due to missing split(s).")
            missing_any = True
            continue

        vocab, tokens = get_vocab_tokens(train)
        profile, dim, bucket, minn, maxn, wngrams = profile_by_vocab(vocab)
        dsub_this = pick_dsub(dim, args.dsub if args.dsub > 0 else None)
        cutoff_this = cutoff_safe(args.cutoff, vocab)

        print(f"[{script}] stats: vocab={vocab} tokens={tokens}")
        print(f"[{script}] profile={profile} dim={dim} bucket={bucket} minn={minn} maxn={maxn} "
              f"wordNgrams={wngrams} dsub={dsub_this} cutoff={cutoff_this}")

        out_dir = out_root / script / "langclf"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_bin = out_dir.with_suffix(".bin")

        quant_dir = out_root / script / "langclf_quant"
        quant_dir.mkdir(parents=True, exist_ok=True)
        quant_prefix = quant_dir / "model"
        quant_ftz = quant_prefix.with_suffix(".ftz")

        if quant_ftz.is_file():
            print(f"[{script}] Quantized model already exists at {quant_ftz}. Skipping training & quantization.")
            continue

        if model_bin.is_file():
            print(
                f"[{script}] Found existing .bin at {model_bin}. Skipping training; proceeding to quantization (if enabled).")
        else:
            print(f"[{script}] train: {train}")
            print(f"[{script}] valid: {valid}")
            print(f"[{script}] test : {test}")
            print(f"[{script}] out  : {out_dir}")

            train_cmd = [
                args.uv, "run", args.trainer,
                "--epoch", str(args.epochs),
                "--train", str(train),
                "--valid", str(valid),
                "--test", str(test),
                "--output", str(out_dir),
                "--dim", str(dim),
                "--bucket", str(bucket),
                "--minn", str(minn),
                "--maxn", str(maxn),
                "--wordNgrams", str(wngrams),
                # "--autotune",
                # "--autotune_duration", "180",
                # "--autotune_metric", "f1",
                "--verbose", "3",
                "--scan_files",
                "--per_label_valid",
                "--sample_pred_log", "20",
                "--save_test_preds", str(test_preds),
                "--save_valid_preds", str(valid_preds),
            ]
            if args.extra_args.strip():
                train_cmd += args.extra_args.strip().split()

            try:
                run(train_cmd)
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Training failed for [{script}] (exit {e.returncode}).")
                raise

            if not model_bin.is_file():
                print(f"ERROR: Expected trained model not found: {model_bin}")
                raise SystemExit(1)

        if args.quantize != 1:
            print(f"[{script}] Quantization disabled. Keeping {model_bin}.")
            continue

        if not model_bin.is_file():
            print(f"WARN: Missing model for quantization: {model_bin}")
            q_fail_any = True
            continue

        size_before = human_bytes(model_bin.stat().st_size)
        print(f"[{script}] quantizing -> {quant_dir} (cutoff={cutoff_this}, dsub={dsub_this})")
        print(f"[{script}] size before: {size_before}")

        quant_cmd = [
            args.uv, "run", args.quantizer,
            "--model", str(model_bin),
            "--train", str(train),
            "--valid", str(valid),
            "--test", str(test),
            "--output", str(quant_prefix),
            "--cutoff", str(cutoff_this),
            "--dsub", str(dsub_this),
            "--pad-vocab",
        ]
        if args.qextra_args.strip():
            quant_cmd += args.qextra_args.strip().split()

        try:
            run(quant_cmd)
        except subprocess.CalledProcessError as e:
            if e.returncode == 4:
                print(f"INFO: [{script}] Model too tiny to quantize either matrix. Keeping optimized .bin.")
            else:
                print(f"WARN: Quantization failed for [{script}] (exit {e.returncode}); keeping original .bin.",
                      file=sys.stderr)
                q_fail_any = True
            continue

        if quant_ftz.is_file():
            size_after = human_bytes(quant_ftz.stat().st_size)
            print(f"[{script}] quantized size: {size_after}")
            print(f"[{script}] quantization done. Removing original model: {model_bin}")
            try:
                model_bin.unlink(missing_ok=True)
            except TypeError:
                if model_bin.exists():
                    model_bin.unlink()
        else:
            print(f"WARN: Expected {quant_ftz} not found after quantization.")
            q_fail_any = True

    if missing_any:
        print("Some scripts were skipped due to missing splits.")

    if args.quantize == 1 and q_fail_any:
        print("At least one quantization failed; see warnings above.")

    print("Done.")


if __name__ == "__main__":
    main()

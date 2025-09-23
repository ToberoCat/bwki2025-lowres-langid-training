#!/usr/bin/env python3
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

def per_label_split(group_df: pd.DataFrame, train_ratio: float, valid_ratio: float, seed: int):
    """Split a single label group into train/valid/test with at least 1 in each split when possible."""
    df = group_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    # base allocations
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = n - n_train - n_valid

    # try to ensure all three have at least 1 if possible
    if n >= 3:
        if n_train == 0: n_train = 1
        if n_valid == 0: n_valid = 1
        n_test = n - n_train - n_valid
        if n_test <= 0:
            # take one from the larger of train/valid to make room for test
            if n_train >= n_valid and n_train > 1:
                n_train -= 1
            elif n_valid > 1:
                n_valid -= 1
            else:
                # fallback: steal from train if both are 1
                n_train = max(1, n_train - 1)
            n_test = n - n_train - n_valid

    # guardrails (can happen for very tiny n or extreme ratios)
    if n_train < 0: n_train = 0
    if n_valid < 0: n_valid = 0
    if n_test  < 0: n_test  = 0
    # final slice
    i0 = n_train
    i1 = n_train + n_valid
    tr = df.iloc[:i0]
    va = df.iloc[i0:i1]
    te = df.iloc[i1:]
    return tr, va, te

def split_by_label(df: pd.DataFrame, label_col: str, train_ratio: float, valid_ratio: float, seed: int, min_per_class: int):
    train_list, valid_list, test_list = [], [], []

    # Optionally drop labels that are too small to ever appear in all three splits
    if min_per_class is not None and min_per_class > 0:
        counts = df[label_col].value_counts()
        keep_labels = set(counts[counts >= min_per_class].index)
        df = df[df[label_col].isin(keep_labels)]

    for _, group_df in df.groupby(label_col, sort=False):
        tr, va, te = per_label_split(group_df, train_ratio, valid_ratio, seed)
        if len(tr): train_list.append(tr)
        if len(va): valid_list.append(va)
        if len(te): test_list.append(te)

    tr = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame(columns=df.columns)
    va = pd.concat(valid_list, ignore_index=True) if valid_list else pd.DataFrame(columns=df.columns)
    te = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame(columns=df.columns)

    # Enforce identical label sets across splits: intersect
    if not tr.empty and not va.empty and not te.empty:
        labs_tr = set(tr[label_col].unique())
        labs_va = set(va[label_col].unique())
        labs_te = set(te[label_col].unique())
        common = labs_tr & labs_va & labs_te
        if common:
            tr = tr[tr[label_col].isin(common)]
            va = va[va[label_col].isin(common)]
            te = te[te[label_col].isin(common)]
        else:
            # No common labels -> return empty splits (nothing usable in a stratified sense)
            tr = tr.iloc[0:0]
            va = va.iloc[0:0]
            te = te.iloc[0:0]

    return tr, va, te

def log_split_stats(stem: str, tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, label_col: str):
    def stats(df):
        if df.empty: return 0, {}
        vc = df[label_col].value_counts().to_dict()
        return len(df), vc
    ntr, vctr = stats(tr)
    nva, vcva = stats(va)
    nte, vcte = stats(te)
    common = set(vctr) & set(vcva) & set(vcte)
    print(f"[{stem}] rows: train={ntr}, valid={nva}, test={nte} | common_labels={len(common)}")

def process_file(file_path: Path, output_dir: Path, label_col: str,
                 train_ratio: float, valid_ratio: float, seed: int, min_per_class: int, engine: str):
    df = pd.read_parquet(file_path, engine=engine)
    tr, va, te = split_by_label(df, label_col, train_ratio, valid_ratio, seed, min_per_class)

    stem = file_path.stem
    if not tr.empty:
        (output_dir / "train").mkdir(parents=True, exist_ok=True)
        tr.to_parquet(output_dir / "train" / f"{stem}_train.parquet", index=False, engine=engine)
    if not va.empty:
        (output_dir / "valid").mkdir(parents=True, exist_ok=True)
        va.to_parquet(output_dir / "valid" / f"{stem}_valid.parquet", index=False, engine=engine)
    if not te.empty:
        (output_dir / "test").mkdir(parents=True, exist_ok=True)
        te.to_parquet(output_dir / "test" / f"{stem}_test.parquet", index=False, engine=engine)

    log_split_stats(stem, tr, va, te, label_col)

    # Return per-file label sets so the caller can aggregate across all files
    labels_tr = set(tr[label_col].unique()) if not tr.empty else set()
    labels_va = set(va[label_col].unique()) if not va.empty else set()
    labels_te = set(te[label_col].unique()) if not te.empty else set()
    return labels_tr, labels_va, labels_te

def main():
    parser = argparse.ArgumentParser(description="Split parquet shards into stratified train/valid/test with identical label sets.")
    parser.add_argument("--input-dir", required=True, help="Directory containing *.parquet files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--label-col", default="lang", help="Column name with class labels")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-per-class", type=int, default=3,
                        help="Drop labels with fewer than this many rows in a file before splitting (ensures feasibility).")
    parser.add_argument("--engine", choices=["pyarrow", "fastparquet"], default="pyarrow")
    args = parser.parse_args()

    # sanity: ratios
    if not (0.0 < args.train_ratio < 1.0 and 0.0 <= args.valid_ratio < 1.0 and args.train_ratio + args.valid_ratio < 1.0):
        raise SystemExit("Ratios must satisfy: 0 < train < 1, 0 <= valid < 1, and (train + valid) < 1.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No *.parquet files found in {input_dir}")

    # Global label trackers
    total_tr, total_va, total_te = set(), set(), set()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                process_file, f, output_dir, args.label_col,
                args.train_ratio, args.valid_ratio, args.seed, args.min_per_class, args.engine
            )
            for f in files
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Splitting files"):
            labels_tr, labels_va, labels_te = fut.result()
            total_tr |= labels_tr
            total_va |= labels_va
            total_te |= labels_te

    # Final summary counters
    union_all = total_tr | total_va | total_te
    common_all = total_tr & total_va & total_te

    print("\n[SUMMARY]")
    print(f"Files processed: {len(files)}")
    print(f"Unique '{args.label_col}' in TRAIN: {len(total_tr)}")
    print(f"Unique '{args.label_col}' in VALID: {len(total_va)}")
    print(f"Unique '{args.label_col}' in TEST : {len(total_te)}")
    print(f"Unique '{args.label_col}' in ANY  : {len(union_all)}")
    print(f"'{args.label_col}' present in ALL splits: {len(common_all)}")

if __name__ == "__main__":
    main()

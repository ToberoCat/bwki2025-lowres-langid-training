#!/usr/bin/env python3
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm

def balance_batch(batch, label_col, region_col, balance, balance_target, seed):
    import pandas as pd
    df = batch.to_pandas()
    if balance != "none" or balance_target > 0:
        group_keys = [label_col] if not region_col else [label_col, region_col]
        tr_by_group = {grp: g.copy() for grp, g in df.groupby(group_keys, sort=False)}
        group_counts = {grp: len(g) for grp, g in tr_by_group.items()}

        current_max = max(group_counts.values()) if group_counts else 0
        current_median = int(np.median(list(group_counts.values()))) if group_counts else 0

        if balance == "max":
            target_per_group = {grp: current_max for grp in tr_by_group}
        elif balance == "median_x2":
            target_per_group = {grp: max(2 * current_median, group_counts[grp]) for grp in tr_by_group}
        else:
            target_per_group = {grp: group_counts[grp] for grp in tr_by_group}

        if balance_target > 0:
            for grp in target_per_group:
                target_per_group[grp] = max(target_per_group[grp], balance_target)

        balanced_dfs = []
        for grp, df_grp in tr_by_group.items():
            need, have = target_per_group[grp], len(df_grp)
            if need <= have:
                balanced_dfs.append(df_grp)
            else:
                extra_df = df_grp.sample(n=need - have, replace=True, random_state=seed)
                balanced_dfs.append(pd.concat([df_grp, extra_df], ignore_index=True))
        df = pd.concat(balanced_dfs, ignore_index=True)
    return pa.Table.from_pandas(df, preserve_index=False)

def process_file(file_path, output_dir, label_col, region_col, balance, balance_target, seed, chunk_rows=100_000):
    pf = pq.ParquetFile(file_path)
    subset = "train" if "train" in file_path.stem else ("valid" if "valid" in file_path.stem else "test")
    out_file = output_dir / subset / file_path.name
    out_file.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    for batch in pf.iter_batches(batch_size=chunk_rows):
        balanced_table = balance_batch(batch, label_col, region_col, balance, balance_target, seed)
        if writer is None:
            writer = pq.ParquetWriter(out_file, balanced_table.schema)
        writer.write_table(balanced_table)
    if writer:
        writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-col", default="lang")
    parser.add_argument("--region-col", default=None)
    parser.add_argument("--balance", choices=["none", "median_x2", "max"], default="none")
    parser.add_argument("--balance-target", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subset in ["train", "valid", "test"]:
        (output_dir / subset).mkdir(parents=True, exist_ok=True)
        files = list((input_dir / subset).glob("*.parquet"))
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_file, f, output_dir, args.label_col, args.region_col,
                                 args.balance, args.balance_target, args.seed, args.chunk_rows) for f in files]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Balancing files"):
                pass

if __name__ == "__main__":
    main()

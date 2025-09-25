#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
import polars as pl
from tqdm import tqdm
import ray
from src.text_utils import near_deduplicate

ray.init(
    ignore_reinit_error=True,
    include_dashboard=False,
    logging_level=30,
    configure_logging=True,
    runtime_env={
        "excludes": [".git/objects"],
    }
)

# ----------------- Remote Tasks -----------------
@ray.remote
def dedup_chunk(texts, langs, args):
    kept_idx, _ = near_deduplicate(
        texts,
        langs,
        bands=args.near_dedup_bands,
        ham_thresh=args.near_dedup_hamming,
        ngram=args.near_dedup_ngram,
        workers=args.workers,
        batch_size=args.near_dedup_batch_size,
        memory_limit_gb=args.memory_limit_gb,
    )
    return kept_idx

@ray.remote
def process_file(file_path: str, output_dir: str, args):
    try:
        out_file = Path(output_dir) / Path(file_path).name
        df = pl.read_parquet(file_path)

        if args.text_col not in df.columns:
            print(f"[WARN] Skipping {file_path}: missing column '{args.text_col}'")
            return 0
        if args.lang_col not in df.columns:
            df = df.with_columns(pl.lit("").alias(args.lang_col))

        texts = df[args.text_col].to_list()
        langs = df[args.lang_col].to_list()

        chunk_size = 5000
        n_chunks = math.ceil(len(texts) / chunk_size)
        futures = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(texts))
            futures.append(dedup_chunk.remote(texts[start:end], langs[start:end], args))

        kept_indices = []
        results = ray.get(futures)
        for i, idx_list in enumerate(results):
            start = i * chunk_size
            kept_indices.extend([start + idx for idx in idx_list])

        df_kept = df[kept_indices]
        df_kept.write_parquet(out_file)
        return df_kept.height

    except Exception as e:
        print(f"[ERROR] Failed processing {file_path}: {e}")
        return 0

# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--lang-col", default="lang")
    parser.add_argument("--near-dedup-bands", type=int, default=8)
    parser.add_argument("--near-dedup-hamming", type=int, default=3)
    parser.add_argument("--near-dedup-ngram", type=int, default=3)
    parser.add_argument("--near-dedup-batch-size", type=int, default=None)
    parser.add_argument("--memory-limit-gb", type=float, default=14.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list(input_dir.glob("*.parquet"))
    if not parquet_files:
        print("No parquet files found in input dir.")
        return

    futures = [process_file.remote(str(f), str(output_dir), args) for f in parquet_files]

    total_rows = 0
    with tqdm(total=len(futures), desc="Processing files") as pbar:
        remaining = futures
        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1)
            res = ray.get(done[0])
            total_rows += res
            pbar.update(1)

    print(f"Near-deduplication complete. Total rows kept: {total_rows}")

if __name__ == "__main__":
    main()

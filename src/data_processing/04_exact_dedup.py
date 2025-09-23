#!/usr/bin/env python3
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm


def process_file(file_path: Path, output_dir: Path, text_col: str, lang_col: str, engine: str = "pyarrow", compression: str = "snappy") -> int:
    try:
        # Read both text + lang
        df = pd.read_parquet(file_path, columns=[text_col, lang_col], engine=engine)

        n_rows_before = len(df)

        # Deduplicate only on text column, but keep lang
        df = df.drop_duplicates(subset=[text_col], ignore_index=True)
        n_rows_after = len(df)

        out_file = output_dir / file_path.name
        df.to_parquet(out_file, engine=engine, compression=compression)
        return n_rows_after
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Drop duplicate rows from multiple Parquet files, preserving lang column.")
    parser.add_argument("--input-dir", required=True, help="Directory containing Parquet files")
    parser.add_argument("--output-dir", required=True, help="Directory to save deduplicated files")
    parser.add_argument("--text-col", default="text", help="Column name to deduplicate on")
    parser.add_argument("--lang-col", default="lang", help="Language column to keep")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker threads")
    parser.add_argument("--engine", choices=["pyarrow", "fastparquet"], default="pyarrow", help="Parquet engine to use")
    parser.add_argument("--compression", default="snappy", choices=["snappy", "gzip", "brotli", "none"], help="Parquet compression codec")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No Parquet files found in {input_dir}")

    total_rows = 0
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for file_path in files:
            futures.append(executor.submit(
                process_file,
                file_path,
                output_dir,
                args.text_col,
                args.lang_col,
                args.engine,
                args.compression
            ))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Dropping duplicates"):
            total_rows += fut.result()

    print(f"Deduplication complete. Total unique rows across all files: {total_rows}")


if __name__ == "__main__":
    main()

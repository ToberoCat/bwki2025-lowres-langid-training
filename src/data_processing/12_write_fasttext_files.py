#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

import pyarrow.parquet as pq
from tqdm import tqdm

from src.writers import write_fasttext

SPLIT_RE = {
    "train": re.compile(r"(?:^|/)(train)(?:/|$)", re.IGNORECASE),
    "valid": re.compile(r"(?:^|/)(valid|validation|val)(?:/|$)", re.IGNORECASE),
    "test":  re.compile(r"(?:^|/)(test|eval)(?:/|$)", re.IGNORECASE),
}

def detect_subset(p: Path) -> str:
    s = f"/{p.as_posix().lower()}/"
    for name, rgx in SPLIT_RE.items():
        if rgx.search(s):
            return name
    return "train"  # sensible default

def process_file(file_path: Path, output_dir: Path, args) -> None:
    subset = detect_subset(file_path)
    out_file = output_dir / subset / f"{file_path.stem}.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Truncate file once per source parquet
    out_file.write_text("", encoding="utf-8")

    pf = pq.ParquetFile(file_path)

    # Only pull needed columns if present
    needed = [args.label_col, args.text_col]
    if args.region_col:
        needed.append(args.region_col)

    for batch in pf.iter_batches(batch_size=args.chunk_rows, columns=needed):
        df = batch.to_pandas()
        if not df.empty:
            write_fasttext(
                out_file, df,
                args.label_col, args.text_col,
                args.region_col
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-col", default="lang")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--region-col", default="region_code")
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output split dirs
    for subset in ["train", "valid", "test"]:
        (output_dir / subset).mkdir(parents=True, exist_ok=True)

    # Find all parquet files recursively
    files = sorted(input_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found under {input_dir}. "
                         f"Check your path/split names (train/valid|validation/test).")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_file, p, output_dir, args): p for p in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Writing FastText files"):
            # Surface exceptions instead of silently ignoring them
            try:
                fut.result()
            except Exception as e:
                src = futures[fut]
                raise RuntimeError(f"Failed on {src}") from e

if __name__ == "__main__":
    main()

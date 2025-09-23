#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from .writers import write_fasttext


def process_file(file_path: Path, output_dir: Path, args) -> None:
    pf = pq.ParquetFile(file_path)
    subset = "train" if "train" in file_path.stem else ("valid" if "valid" in file_path.stem else "test")
    out_file = output_dir / subset / f"{file_path.stem}.txt"
    ## Just to clear the file the first time and the file is clear
    with out_file.open("w") as fp:
        pass

    out_file.parent.mkdir(parents=True, exist_ok=True)
    for batch in pf.iter_batches(batch_size=args.chunk_rows):
        df = batch.to_pandas()
        if not df.empty:
            write_fasttext(out_file, df, args.label_col, args.text_col, args.region_col)


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
    for subset in ["train", "valid", "test"]:
        (output_dir / subset).mkdir(parents=True, exist_ok=True)

        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for file_path in (input_dir / subset).glob("*.parquet"):
                futures.append(ex.submit(process_file, file_path, output_dir, args))
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Writing FastText files"):
                pass


if __name__ == "__main__":
    main()

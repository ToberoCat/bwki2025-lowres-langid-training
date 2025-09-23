#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def process_file(file_path, output_dir, label_col, chunk_rows, keep_labels):
    pf = pq.ParquetFile(file_path)
    out_file = output_dir / file_path.name
    writer = None

    for batch in pf.iter_batches(batch_size=chunk_rows):
        df_chunk = batch.to_pandas()
        df_chunk = df_chunk[df_chunk[label_col].isin(keep_labels)]
        if df_chunk.empty:
            continue
        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_file, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-col", default="lang")
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for file_path in input_dir.glob("*.parquet"):
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(batch_size=args.chunk_rows):
            df_chunk = batch.to_pandas()
            vc = df_chunk[args.label_col].value_counts()
            for k, v in vc.items():
                counts[k] = counts.get(k, 0) + v
    keep_labels = {k for k, v in counts.items() if v >= args.min_count}

    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for file_path in input_dir.glob("*.parquet"):
            futures.append(ex.submit(
                process_file, file_path, output_dir, args.label_col, args.chunk_rows, keep_labels
            ))

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Filtering labels"):
            pass

if __name__ == "__main__":
    main()

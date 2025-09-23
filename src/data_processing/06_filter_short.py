#!/usr/bin/env python3
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from bwki2025.data_processing.fasttext_pipeline.text_utils import is_short

def process_file(file_path, output_dir, text_col, short_word_threshold, short_char_threshold, chunk_rows):
    pf = pq.ParquetFile(file_path)
    out_file = output_dir / file_path.name
    writer = None

    for batch in pf.iter_batches(batch_size=chunk_rows):
        df_chunk = batch.to_pandas()
        mask = ~df_chunk[text_col].map(lambda s: is_short(s, short_word_threshold, short_char_threshold))
        df_chunk = df_chunk[mask]
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
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--short-word-threshold", type=int, default=8)
    parser.add_argument("--short-char-threshold", type=int, default=40)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for file_path in input_dir.glob("*.parquet"):
            futures.append(ex.submit(
                process_file,
                file_path, output_dir, args.text_col,
                args.short_word_threshold, args.short_char_threshold, args.chunk_rows
            ))
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Filtering short texts"):
            pass

if __name__ == "__main__":
    main()

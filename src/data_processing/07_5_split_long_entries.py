#!/usr/bin/env python3
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def split_text(text: str, max_length: int) -> list[str]:
    if not text or not max_length:
        return [text]

    chunks = []
    while len(text) > max_length:
        cut = text.rfind(" ", 0, max_length)
        if cut == -1:
            cut = max_length
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def split_chunk(df_chunk: pd.DataFrame, text_col: str, max_length: int) -> pd.DataFrame:
    new_rows = []
    for _, row in df_chunk.iterrows():
        chunks = split_text(str(row[text_col]), max_length)
        for chunk in chunks:
            new_rows.append({
                "region_code": row.get("region_code", ""),
                "lang": row.get("lang", ""),
                text_col: chunk,
            })
    return pd.DataFrame(new_rows)


def process_file(file_path: Path, output_dir: Path, text_col: str) -> int:
    pf = pq.ParquetFile(file_path)
    out_file = output_dir / file_path.name
    writer = None
    rows_written = 0

    for batch in pf.iter_batches(batch_size=5000):
        df_chunk = batch.to_pandas()

        if df_chunk.empty:
            continue

        max_length = int(df_chunk[text_col].astype(str).str.len().mean())
        if max_length <= 0:
            max_length = 100

        # No need for ThreadPoolExecutor for a single batch
        df_chunk = split_chunk(df_chunk, text_col, max_length)

        if df_chunk.empty:
            continue

        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_file, table.schema)
        writer.write_table(table)
        rows_written += len(df_chunk)

    if writer:
        writer.close()
    return rows_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.parquet"))
    if not files:
        print("No Parquet files found in input dir.")
        return

    total_files = len(files)
    with ThreadPoolExecutor(max_workers=args.workers) as ex, tqdm(total=total_files, desc="Splitting files") as pbar:
        futures = {ex.submit(process_file, f, output_dir, args.text_col): f for f in files}
        for fut in as_completed(futures):
            rows_written = fut.result()
            if rows_written > 0:
                pbar.update(1)


if __name__ == "__main__":
    main()

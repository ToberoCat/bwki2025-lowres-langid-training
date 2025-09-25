#!/usr/bin/env python3
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from src.snippets import word_snippets, char_snippets

def process_file(file_path, subset, output_dir, args):
    pf = pq.ParquetFile(file_path)
    out_file = output_dir / subset / file_path.name.replace("09", "10")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    try:
        for batch in pf.iter_batches(batch_size=args.chunk_rows):
            df_chunk = batch.to_pandas()
            new_rows = []

            for r in df_chunk.itertuples(index=False):
                lab = getattr(r, args.label_col)
                txt = getattr(r, args.text_col)
                region = getattr(r, args.region_col)
                new_rows.append({args.label_col: lab, args.text_col: txt, args.region_col: region})
                for s in word_snippets(txt, args.snips_per_doc, args.snip_min_words, args.snip_max_words):
                    new_rows.append({args.label_col: lab, args.text_col: s, args.region_col: region})
                for s in char_snippets(txt, args.char_snips_per_doc, args.char_min, args.char_max):
                    new_rows.append({args.label_col: lab, args.text_col: s, args.region_col: region})

            if not new_rows:
                continue

            table = pa.Table.from_pandas(pd.DataFrame(new_rows), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_file, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-col", default="lang")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--region-col", default="region_code")
    parser.add_argument("--snips-per-doc", type=int, default=8)
    parser.add_argument("--snip-min-words", type=int, default=3)
    parser.add_argument("--snip-max-words", type=int, default=12)
    parser.add_argument("--char-snips-per-doc", type=int, default=4)
    parser.add_argument("--char-min", type=int, default=20)
    parser.add_argument("--char-max", type=int, default=80)
    parser.add_argument("--chunk-rows", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    for subset in ["train", "valid", "test"]:
        (output_dir / subset).mkdir(parents=True, exist_ok=True)
        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for file_path in (input_dir / subset).glob("*.parquet"):
                futures.append(ex.submit(process_file, file_path, subset, output_dir, args))
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Augmenting with snippets"):
                pass

if __name__ == "__main__":
    main()

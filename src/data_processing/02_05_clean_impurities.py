#!/usr/bin/env python3
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import langcodes
from tqdm import tqdm
import icu

MULTI_SCRIPT_LANGS = {
    "jpn": {"Hani", "Hira", "Kana"},
    "kor": {"Hang", "Hani"},
    "srp": {"Latn", "Cyrl"},
}

NOISE = {"Zyyy"}

def allowed_scripts(alpha3: str):
    if alpha3 in MULTI_SCRIPT_LANGS:
        return MULTI_SCRIPT_LANGS[alpha3] | {"Zyyy"}
    lang = langcodes.get(alpha3)
    script = lang.script if lang.script else None
    if script:
        return {script, "Zyyy"}
    return {"Zyyy"}

def scripts_in_word(word: str, use_extensions=True):
    counts = Counter()
    for cp in map(ord, word):
        if use_extensions:
            exts = icu.Script.getScriptExtensions(cp)
            exts = [icu.Script(e).getShortName() for e in exts]
            useful = [sc for sc in exts if sc not in NOISE]
            if useful:
                counts.update(useful)
        else:
            sc = icu.Script.getScript(cp).getShortName()
            if sc not in NOISE:
                counts[sc] += 1
    return set(counts.keys())

def clean_text_by_script(text: str, allowed: set):
    if not text:
        return text
    words = text.split()
    kept = []
    for word in words:
        word_scripts = scripts_in_word(word)
        if word_scripts & allowed:
            kept.append(word)
    return " ".join(kept)

def clean_chunk(df_chunk: pd.DataFrame, text_col: str, lang_col: str) -> pd.DataFrame:
    df_chunk[text_col] = df_chunk.apply(
        lambda row: clean_text_by_script(row[text_col], allowed_scripts(row[lang_col])),
        axis=1
    )
    return df_chunk

def process_file(file_path: Path, output_dir: Path, text_col: str, lang_col: str):
    pf = pq.ParquetFile(file_path)
    out_file = output_dir / file_path.name
    writer = None

    for batch in pf.iter_batches(batch_size=5000):
        df_chunk = batch.to_pandas()
        if df_chunk.empty:
            continue

        df_chunk = clean_chunk(df_chunk, text_col, lang_col)
        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_file, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()
    else:
        print(f"No data found in {file_path}, skipping write.")

def main():
    parser = argparse.ArgumentParser(description="Step 02.05: Remove words not matching language scripts")
    parser.add_argument("--input-dir", required=True, help="Directory with input Parquet files")
    parser.add_argument("--output-dir", required=True, help="Directory to save cleaned Parquet files")
    parser.add_argument("--text-col", default="text", help="Name of the text column")
    parser.add_argument("--lang-col", default="lang", help="Name of the language column (ISO 639-3)")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for file_path in input_dir.glob("*.parquet"):
            futures.append(ex.submit(process_file, file_path, output_dir, args.text_col, args.lang_col))

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Cleaning files by script"):
            pass

if __name__ == "__main__":
    main()

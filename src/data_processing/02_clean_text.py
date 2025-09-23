#!/usr/bin/env python3
import argparse
import re
import unicodedata
from pathlib import Path
from typing import Literal, Optional

import polars as pl
from tqdm import tqdm

META_PATTERNS = [
    r"^Category:.*",
    r"^File:.*",
    r"^Image:.*",
    r"^Template:.*",
    r"^Portal:.*",
    r"^Wikipedia:.*",
    r"^Special:.*",
    r"\[\[Category:[^\]]+\]\]",
]
meta_re = re.compile("|".join(META_PATTERNS), re.IGNORECASE)


def remove_wiki_metadata(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines()
        if not meta_re.match(line.strip())
    ).strip()


def safe_digit_normalize_char(ch: str) -> str:
    try:
        return str(unicodedata.decimal(ch))
    except Exception:
        try:
            return str(unicodedata.digit(ch))
        except Exception:
            return ch


def python_cleaner_factory(lowercase, normalize_digits, remove_numbers,
                           unicode_norm: Optional[Literal["NFC", "NFD", "NFKC", "NFKD"]] = "NFC",
                           remove_wiki_meta=False):
    def cleaner(text):
        if text is None or not isinstance(text, str):
            return text
        if unicode_norm:
            text = unicodedata.normalize(unicode_norm, text)
        if lowercase:
            text = text.casefold()
        if normalize_digits:
            text = "".join(safe_digit_normalize_char(ch) for ch in text)
        if remove_numbers:
            text = re.sub(r"\d+", "", text)
        if remove_wiki_meta:
            text = remove_wiki_metadata(text)
        return text

    return cleaner


def clean_column_polars(
    df: pl.DataFrame,
    col: str,
    lowercase: bool,
    remove_numbers: bool,
    normalize_digits: bool,
    remove_wiki_meta: bool,
    unicode_norm: Optional[str],
) -> pl.DataFrame:
    if lowercase:
        df = df.with_columns(pl.col(col).str.to_lowercase())
    if remove_numbers:
        df = df.with_columns(pl.col(col).str.replace_all(r"\d+", ""))

    if normalize_digits or remove_wiki_meta or (unicode_norm and unicode_norm != "NFC"):
        py_cleaner = python_cleaner_factory(
            lowercase=False,
            normalize_digits=normalize_digits,
            remove_numbers=False,
            unicode_norm=unicode_norm,
            remove_wiki_meta=remove_wiki_meta,
        )
        df = df.with_columns(
            pl.col(col).map_elements(py_cleaner).alias(col)
        )

    return df


def process_file(
    file_path: Path,
    output_dir: Path,
    text_col: str,
    lowercase: bool,
    normalize_digits: bool,
    remove_numbers: bool,
    unicode_norm: Optional[Literal["NFC", "NFD", "NFKC", "NFKD"]],
    remove_wiki_meta: bool,
    compression: Optional[str],
    batch_size: int = 200_000,
):
    out_file = output_dir / file_path.name

    scan = pl.scan_parquet(file_path)
    n_rows = scan.collect().height

    writer = None
    offset = 0
    with tqdm(total=n_rows, desc=f"Cleaning {file_path.name}") as pbar:
        for df in scan.collect().iter_slices(batch_size):
            df_clean = clean_column_polars(
                df, text_col, lowercase, remove_numbers,
                normalize_digits, remove_wiki_meta, unicode_norm
            )
            if writer is None:
                writer = pl.DataFrame.write_parquet
                df_clean.write_parquet(out_file, compression=compression if compression else "gzip")
            else:
                df_clean.write_parquet(out_file, compression=compression if compression else "gzip")
            offset += len(df_clean)
            pbar.update(len(df_clean))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--normalize-digits", action="store_true")
    parser.add_argument("--remove-numbers", action="store_true")
    parser.add_argument(
        "--unicode-norm",
        choices=["NFC", "NFD", "NFKC", "NFKD", "none"],
        default="NFC",
    )
    parser.add_argument("--remove-wiki-meta", action="store_true")
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "zstd", "gzip", "brotli", "lz4", "none"],
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression
    unicode_norm = None if args.unicode_norm == "none" else args.unicode_norm

    files = list(input_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No .parquet files found in {input_dir}")

    for fp in files:
        process_file(
            file_path=fp,
            output_dir=output_dir,
            text_col=args.text_col,
            lowercase=args.lowercase,
            normalize_digits=args.normalize_digits,
            remove_numbers=args.remove_numbers,
            unicode_norm=unicode_norm,
            remove_wiki_meta=args.remove_wiki_meta,
            compression=compression,
        )


if __name__ == "__main__":
    main()

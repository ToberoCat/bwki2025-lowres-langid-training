import argparse
import os
import subprocess
from pathlib import Path


def run_step(cmd, step_name):
    print("Running step", step_name)
    subprocess.run(cmd, check=True)
    print("------------------------------")


def main():
    ap = argparse.ArgumentParser(description="Full FastText preprocessing pipeline")
    ap.add_argument("--input", default="./artifacts/shards", help="Raw input dataset")
    ap.add_argument("--outdir", default="./artifacts/splitParquet", help="Output directory for intermediate artifacts")
    ap.add_argument("--label-col", default="lang")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))

    ap.add_argument("--lowercase", action="store_true", default=True)
    ap.add_argument("--remove-numbers", action="store_true", default=True)
    ap.add_argument("--normalize-digits", action="store_true", default=True)

    ap.add_argument("--append-meta", action="store_true")
    ap.add_argument("--meta-no-script", action="store_true")
    ap.add_argument("--meta-no-length", action="store_true")
    ap.add_argument("--meta-no-digitratio", action="store_true")

    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    steps = [
        # ("01_load_data", [
        #    "--input", str(Path(args.input).resolve()),
        #    "--output", str(outdir / "01_raw"),
        #    "--workers", str(args.workers),
        # ]),
        #
        # ("02_clean_text", [
        #    "--input-dir", str(outdir / "01_raw"),
        #    "--output-dir", str(outdir / "02_cleaned"),
        #    "--text-col", args.text_col,
        # ] + (["--lowercase"] if args.lowercase else []) +
        #    (["--normalize-digits"] if args.normalize_digits else []) +
        #    (["--remove-numbers"] if args.remove_numbers else [])
        # ),
        #
        #   ("03_filter_empty", [
        #      "--input", str(outdir / "02_cleaned"),
        #      "--output", str(outdir / "03_filtered_empty"),
        #      "--label-col", args.label_col,
        #      "--text-col", args.text_col,
        #   ]),
        #
        #  ("04_exact_dedup", [
        #     "--input", str(outdir / "03_filtered_empty"),
        #     "--output", str(outdir / "04_deduped"),
        #     "--text-col", args.text_col,
        #  ]),
        #
        #   ("05_near_dedup", ["--input", str(outdir / "04_deduped"),
        #                      "--output", str(outdir / "05_near_deduped"),
        #                      "--text-col", args.text_col]),
        #
        #  ("06_filter_short", [
        #      "--input", str(outdir / "03_filtered_empty"),
        #      "--output", str(outdir / "06_filtered_short"),
        #      "--text-col", args.text_col,
        #  ]),
        #
        #  ("07_filter_low_count_labels", [
        #      "--input", str(outdir / "04_deduped"),
        #      "--output", str(outdir / "07_filtered_low_count"),
        #      "--label-col", args.label_col,
        #      "--min-count", "1",
        #  ]),
        #
        #  ("07_5_split_long_entries", [
        #      "--input", str(outdir / "05_near_deduped"),
        #      "--output", str(outdir / "07_5_split_long_entries")
        #  ]),
        #
        #  ("08_split_data", [
        #      "--input-dir", str(outdir / "01_raw"),
        #      "--output-dir", str(outdir / "08_split"),
        #      "--label-col", args.label_col,
        #      "--train-ratio", "0.8",
        #      "--valid-ratio", "0.1",
        #      "--seed", str(args.seed),
        #      "--workers", str(args.workers),
        #  ]),
         #
         # ("09_balance_training_data", [
         #      "--input-dir", str(outdir / "08_split"),
         #      "--output-dir", str(outdir / "09_balanced"),
         #      "--label-col", args.label_col,
         #      "--region-col", "region_code",
         #      "--balance", "median_x2",
         #      "--balance-target", "0",
         #      "--seed", str(args.seed),
         #  ]),
         #
         #  ("10_augment_with_snippets", [
         #      "--input-dir", str(outdir / "09_balanced"),
         #      "--output-dir", str(outdir / "10_augmented"),
         #      "--label-col", args.label_col,
         #      "--text-col", args.text_col,
         #      "--region-col", "region_code",
         #      "--snips-per-doc", "8",
         #      "--snip-min-words", "3",
         #      "--snip-max-words", "12",
         #      "--char-snips-per-doc", "4",
         #      "--char-min", "20",
         #      "--char-max", "80",
         #      "--chunk-rows", "50000",
         #      "--workers", str(args.workers),
         #  ]),

         ("12_write_fasttext_files", [
             "--input-dir", str(outdir / "08_split"),
             "--output-dir", str(outdir / "12_fasttext_files"),
             "--label-col", args.label_col,
             "--text-col", args.text_col,
             "--region-col", "region_code",
             "--workers", str(args.workers),
         ]),
    ]

    for step_name, step_args in steps:
        script_path = Path(f"src/data_processing/{step_name}.py").resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        run_step(["python", "-m", f"src.data_processing.{step_name}"] + step_args, step_name)

    print(f"Pipeline complete. Outputs are in {outdir}")


if __name__ == "__main__":
    main()

import argparse
import gc
import multiprocessing
import shutil
import tempfile
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import icu
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

NOISE = {"Zyyy", "Zinh", "Zzzz"}

def scripts_in_text(s: str, use_extensions=True) -> Counter:
    counts = Counter()
    for cp in map(ord, s):
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
    return counts

def get_dominant_script(s: str, use_extensions=True):
    if not isinstance(s, str) or not s:
        return "Unknown"
    counts = scripts_in_text(s, use_extensions=use_extensions)
    if not counts:
        return "Unknown"
    return counts.most_common(1)[0][0]

def process_batch_to_temp_files(batch: pa.RecordBatch, text_col: str, file_temp_dir: Path,
                                batch_id: int):
    """Process a single batch and write directly to per-file temporary dir."""
    if batch.num_rows == 0:
        return []

    temp_files_created = []
    try:
        text_column_array = batch.column(text_col).to_pylist()
    except Exception as e:
        print(f"Batch {batch_id}: Error accessing text column: {e}")
        return []

    script_rows = defaultdict(list)
    for i in range(batch.num_rows):
        text = text_column_array[i]
        dominant_script = get_dominant_script(text)
        script_rows[dominant_script].append(i)

    for script, row_indices in script_rows.items():
        if not row_indices:
            continue
        arrays = []
        for field in batch.schema:
            col_data = [batch.column(field.name)[i].as_py() for i in row_indices]
            arrays.append(pa.array(col_data, type=field.type))
        table = pa.Table.from_arrays(arrays, schema=batch.schema)

        # IMPORTANT: write into a per-input-file subdir to avoid collisions
        file_temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = file_temp_dir / f"batch_{batch_id:08d}_{script}.parquet"
        try:
            pq.write_table(table, temp_file, compression='snappy')
            temp_files_created.append((script, temp_file))
        except Exception as e:
            print(f"Batch {batch_id}: Error writing temp file for script {script}: {e}")

    # Free memory
    del batch, text_column_array, script_rows
    gc.collect()
    return temp_files_created

def merge_script_files_streaming(script: str, temp_files: list[Path], output_file: Path,
                                 batch_size: int = 100_000):
    """Merge temporary files for a single script using streaming. De-dupe paths."""
    if not temp_files:
        return

    # De-duplicate and keep only existing files (protect against accidental duplicates)
    seen = set()
    unique_files = []
    for p in temp_files:
        if p in seen:
            continue
        if p.exists():
            unique_files.append(p)
            seen.add(p)

    if not unique_files:
        print(f"Warning: No temp files exist for script {script}. Skipping.")
        return

    unique_files.sort()  # deterministic order

    # Read schema from first file
    try:
        pf0 = pq.ParquetFile(unique_files[0])
        schema = pf0.schema_arrow
        del pf0
    except Exception as e:
        print(f"Error reading schema for script {script}: {e}")
        return

    with pq.ParquetWriter(output_file, schema, compression='snappy') as writer:
        for temp_file in unique_files:
            try:
                pf = pq.ParquetFile(temp_file)
                for batch in pf.iter_batches(batch_size=batch_size):
                    writer.write_batch(batch)
                del pf
            except Exception as e:
                print(f"Error merging file {temp_file} for script {script}: {e}")
            finally:
                try:
                    temp_file.unlink(missing_ok=True)
                except Exception:
                    pass

def get_memory_usage():
    return psutil.virtual_memory().percent

def get_parquet_files(input_path: Path) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        parquet_files = []
        for pattern in ('*.parquet', '*.pq'):
            parquet_files.extend(input_path.glob(pattern))
        return sorted(parquet_files)
    raise ValueError(f"Input path {input_path} is neither a file nor a directory")

def process_single_file(input_file: Path, temp_root: Path, text_col: str,
                        batch_size: int, workers: int, memory_limit: int):
    """Process a single parquet file and return {script: [temp_paths...]}."""
    script_temp_files = defaultdict(list)
    try:
        pf = pq.ParquetFile(input_file)
        total_rows = pf.metadata.num_rows
        if total_rows == 0:
            print(f"File {input_file.name} is empty, skipping.")
            return script_temp_files

        print(f"Processing {input_file.name}: {total_rows:,} rows")

        # Make a unique subdir for this input file to avoid filename collisions
        file_uid = f"{input_file.stem}_{(abs(hash(str(input_file.resolve()))) & 0xFFFFFFFF):08x}"
        file_temp_dir = temp_root / file_uid
        file_temp_dir.mkdir(parents=True, exist_ok=True)

        initial_max_pending = workers * 3
        max_pending = initial_max_pending

        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending_futures = deque()
            batch_id = 0
            rows_processed = 0
            batch_iterator = pf.iter_batches(batch_size=batch_size)

            with tqdm(total=total_rows, desc=f"Processing {input_file.name}", smoothing=0.1) as pbar:
                submission_complete = False
                while pending_futures or not submission_complete:
                    mem_usage = get_memory_usage()
                    if mem_usage > memory_limit:
                        max_pending = max(workers, max_pending // 2)
                        pbar.set_postfix({"mem": f"{mem_usage:.1f}%", "throttle": "ON"})
                        while get_memory_usage() > memory_limit - 5 and pending_futures:
                            oldest = pending_futures.popleft()
                            future, num_rows, bid = oldest
                            try:
                                temp_files = future.result(timeout=1)
                                for script, temp_file in temp_files:
                                    script_temp_files[script].append(temp_file)
                                rows_processed += num_rows
                                pbar.update(num_rows)
                            except Exception as e:
                                if not future.done():
                                    pending_futures.append(oldest)
                                else:
                                    print(f"\nError in batch {bid}: {e}")
                            time.sleep(0.05)
                    else:
                        if mem_usage < memory_limit - 10:
                            max_pending = min(workers * 4, max_pending + 1)
                        pbar.set_postfix({"mem": f"{mem_usage:.1f}%", "pending": len(pending_futures)})

                    while len(pending_futures) < max_pending and not submission_complete:
                        try:
                            batch = next(batch_iterator)
                            if batch.num_rows == 0:
                                continue
                            future = executor.submit(
                                process_batch_to_temp_files,
                                batch, text_col, file_temp_dir, batch_id
                            )
                            pending_futures.append((future, batch.num_rows, batch_id))
                            batch_id += 1
                            del batch
                        except StopIteration:
                            submission_complete = True
                            break

                    completed_count = 0
                    temp_pending = deque()
                    for future, num_rows, bid in pending_futures:
                        if future.done():
                            try:
                                temp_files = future.result()
                                for script, temp_file in temp_files:
                                    script_temp_files[script].append(temp_file)
                                rows_processed += num_rows
                                pbar.update(num_rows)
                                completed_count += 1
                            except Exception as e:
                                print(f"\nError in batch {bid}: {e}")
                                completed_count += 1
                        else:
                            temp_pending.append((future, num_rows, bid))
                    pending_futures = temp_pending

                    if not completed_count and pending_futures:
                        time.sleep(0.01)
                    if batch_id % 100 == 0:
                        gc.collect()

        print(f"Processed {rows_processed:,} rows from {input_file.name}")
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
    return script_temp_files

def split_parquet_streaming(input_path: Path, output_dir: Path, text_col: str = "text",
                            batch_size: int = 200_000, workers: int | None = None,
                            memory_limit: int = 85):
    """
    Split parquet file(s) by dominant Unicode script with adaptive memory management.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = get_parquet_files(input_path)
    if not parquet_files:
        print(f"No parquet files found in {input_path}")
        return

    temp_root = Path(tempfile.mkdtemp(dir=output_dir, prefix='.temp_'))

    try:
        workers = workers or multiprocessing.cpu_count()
        print(f"Processing {len(parquet_files)} parquet file(s) with {workers} workers")
        print(f"Batch size: {batch_size:,} rows")
        print(f"Memory limit: {memory_limit}%")
        print(f"Available RAM: {psutil.virtual_memory().available / (1024 ** 3):.1f} GB")

        all_script_temp_files: dict[str, list[Path]] = defaultdict(list)

        for parquet_file in parquet_files:
            print("\n" + "=" * 60)
            print(f"Processing: {parquet_file}")
            print("=" * 60)
            script_temp_files = process_single_file(
                parquet_file, temp_root, text_col, batch_size, workers, memory_limit
            )
            for script, temp_files in script_temp_files.items():
                all_script_temp_files[script].extend(temp_files)
            print(f"Found {len(script_temp_files)} unique scripts in {parquet_file.name}")

        print(f"\nTotal unique scripts found across all files: {len(all_script_temp_files)}")
        if not all_script_temp_files:
            print("No scripts found in any of the data files.")
            return

        print("\nMerging temporary files by script...")
        with ProcessPoolExecutor(max_workers=min(workers, max(1, len(all_script_temp_files)))) as executor:
            merge_futures = []
            for script, temp_files in all_script_temp_files.items():
                output_file = output_dir / f"{script}.parquet"
                future = executor.submit(
                    merge_script_files_streaming,
                    script, temp_files, output_file, batch_size * 2
                )
                merge_futures.append((future, script, len(temp_files)))

            with tqdm(total=len(merge_futures), desc="Merging scripts") as pbar:
                for future, script, file_count in merge_futures:
                    pbar.set_description(f"Merging {script} ({file_count} files)")
                    try:
                        future.result()
                        of = output_dir / f"{script}.parquet"
                        if of.exists():
                            size_mb = of.stat().st_size / (1024 * 1024)
                            print(f"Created {script}.parquet ({size_mb:.1f} MB)")
                    except Exception as e:
                        print(f"\nError merging script {script}: {e}")
                    pbar.update(1)

        print(f"\nSuccessfully processed {len(parquet_files)} input file(s)")
        print(f"Created {len(all_script_temp_files)} output files by dominant script")
    finally:
        try:
            shutil.rmtree(temp_root, ignore_errors=True)
        except Exception:
            print("Warning: Could not fully clean temporary directory")
        gc.collect()

def main():
    parser = argparse.ArgumentParser(
        description="Split Parquet file(s) by dominant Unicode script with adaptive memory management"
    )
    parser.add_argument("--input", required=True, help="Input Parquet file or directory containing Parquet files")
    parser.add_argument("--output-dir", required=True, help="Output directory for split files")
    parser.add_argument("--text-col", default="text", help="Name of text column (default: text)")
    parser.add_argument("--batch-size", type=int, default=200_000, help="Rows per batch (default: 200000)")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers (default: all CPUs)")
    parser.add_argument("--memory-limit", type=int, default=85, help="Memory usage percentage limit (default: 85)")
    args = parser.parse_args()

    split_parquet_streaming(
        args.input,
        args.output_dir,
        text_col=args.text_col,
        batch_size=args.batch_size,
        workers=args.workers,
        memory_limit=args.memory_limit
    )

if __name__ == "__main__":
    main()

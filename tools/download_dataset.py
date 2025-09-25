import json
import os
import sys
import time
import shutil
from pathlib import Path

from filelock import FileLock, Timeout
from huggingface_hub import snapshot_download

DATASETS_DIR = Path("./artifacts/shards")
HF_REPO = "Tobero/bwki2025-lowres-langid-dataset"
HF_REPO_TYPE = "dataset"

HF_REV = "main"
if "--small" in sys.argv:
    HF_REV = "small"

MARKER = DATASETS_DIR / ".hf_snapshot.json"
LOCKFILE = DATASETS_DIR / ".download.lock"
LOCK = FileLock(str(LOCKFILE))


def _is_populated() -> bool:
    if MARKER.exists():
        try:
            meta = json.loads(MARKER.read_text())
            return (
                meta.get("repo") == HF_REPO
                and meta.get("rev") == HF_REV
                and meta.get("repo_type") == HF_REPO_TYPE
            )
        except Exception:
            pass

    housekeeping = {MARKER.name, LOCKFILE.name}
    try:
        return any(
            p for p in DATASETS_DIR.iterdir()
            if p.name not in housekeeping
        )
    except FileNotFoundError:
        return False


def _move_train_shards():
    """Move shard files from nested data/train/ into DATASETS_DIR."""
    train_dir = DATASETS_DIR / "data" / "train"
    if train_dir.exists():
        for file in train_dir.glob("*"):
            target = DATASETS_DIR / file.name
            if target.exists():
                print(f"[dataset] Skipping existing {target}")
                continue
            shutil.move(str(file), str(target))
            print(f"[dataset] Moved {file.name} -> {target}")
        # optionally clean up the empty directories
        try:
            shutil.rmtree(DATASETS_DIR / "data")
            print("[dataset] Cleaned up nested data/ folder")
        except Exception as e:
            print(f"[dataset] Cleanup failed: {e}")


def ensure_dataset(timeout_s: int = 60 * 30) -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with LOCK.acquire(timeout=timeout_s):
            if _is_populated():
                print(f"[dataset] Already populated: {DATASETS_DIR}. Remove to re-download.")
                return

            t0 = time.time()
            kwargs = dict(
                repo_id=HF_REPO,
                repo_type=HF_REPO_TYPE,
                revision=HF_REV,
                local_dir=str(DATASETS_DIR),
                local_dir_use_symlinks=False,
                max_workers=max(os.cpu_count() or 4, 4),
            )

            snapshot_download(**kwargs)

            # Move shards out of nested structure if needed
            _move_train_shards()

            MARKER.write_text(json.dumps({
                "repo": HF_REPO,
                "rev": HF_REV,
                "repo_type": HF_REPO_TYPE,
                "ts": int(time.time()),
            }))
            print(f"[dataset] Ready in {time.time() - t0:.1f}s -> {DATASETS_DIR}")
    except Timeout:
        raise RuntimeError("Dataset download lock timed out")


ensure_dataset()

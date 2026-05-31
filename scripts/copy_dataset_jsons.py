#!/usr/bin/env python3
"""Copy the dataset split JSONs shipped in this repo (OGKD/data/<DATASET>/) into
the dataset root expected by the dataset classes (default /workspace/dataset_raw,
overridable via DATA_ROOT), preserving the per-dataset folder structure.

Run this after downloading the raw images with download_data.py.
"""
import os
import shutil
import sys
from pathlib import Path


def copy_json_files(src_root: Path, dst_root: Path) -> int:
    if not src_root.exists() or not src_root.is_dir():
        print(f"Source directory does not exist or is not a directory: {src_root}")
        return 1

    dst_root.mkdir(parents=True, exist_ok=True)

    total_copied = 0

    for dataset_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        dest_dataset_dir = dst_root / dataset_dir.name
        dest_dataset_dir.mkdir(parents=True, exist_ok=True)

        copied_in_dataset = 0
        for json_path in dataset_dir.rglob("*.json"):
            relative_path = json_path.relative_to(dataset_dir)
            dest_path = dest_dataset_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(json_path, dest_path)
            copied_in_dataset += 1
            total_copied += 1

        print(f"{dataset_dir.name}: copied {copied_in_dataset} JSON file(s)")

    print(f"Done. Total JSON files copied: {total_copied}")
    return 0


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    src_root = Path(os.environ.get("SPLIT_JSON_SRC", repo_root / "data"))
    dst_root = Path(os.environ.get("DATA_ROOT", "/workspace/dataset_raw"))
    return copy_json_files(src_root, dst_root)


if __name__ == "__main__":
    sys.exit(main())

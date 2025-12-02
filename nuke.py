#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cleanup_explainers.py

Recursively delete all CfExplainer / PyExplainer result directories
so you can rerun the pipeline from scratch.

By default, it scans the current working directory (".").
You can optionally point it to a specific root like PROPOSED_CHANGES
or EXPERIMENTS.

Usage:
    python cleanup_explainers.py
    python cleanup_explainers.py --root PROPOSED_CHANGES
    python cleanup_explainers.py --root experiments_closest
"""

import argparse
import shutil
from pathlib import Path


EXPLAINER_DIR_NAMES = {"CfExplainer", "PyExplainer"}


def cleanup(root: Path, dry_run: bool = False) -> None:
    if not root.exists():
        print(f"[WARN] Root path does not exist: {root}")
        return

    print(f"[INFO] Scanning under root: {root.resolve()}")
    to_delete = []

    # Look for directories named exactly "CfExplainer" or "PyExplainer"
    for path in root.rglob("*"):
        if path.is_dir() and path.name in EXPLAINER_DIR_NAMES:
            to_delete.append(path)

    if not to_delete:
        print("[INFO] No CfExplainer / PyExplainer directories found.")
        return

    print(f"[INFO] Found {len(to_delete)} directories to delete:\n")
    for d in to_delete:
        print(f"  - {d}")

    if dry_run:
        print("\n[DRY-RUN] No directories were actually deleted.")
        return

    print("\n[DELETE] Deleting directories...")
    for d in to_delete:
        try:
            shutil.rmtree(d)
            print(f"[OK] Removed {d}")
        except Exception as e:
            print(f"[ERROR] Failed to remove {d}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory to scan (default: current directory '.')",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be deleted without actually deleting anything.",
    )
    args = ap.parse_args()

    cleanup(Path(args.root), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

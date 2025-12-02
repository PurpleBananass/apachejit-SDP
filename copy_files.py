#!/usr/bin/env python3
import shutil
from pathlib import Path

# Roots (adjust if needed)
SRC_ROOT = Path("experiments11")
DST_ROOT = Path("experiments")

def main():
    # pattern: experiments1/{project}/{model}/CF_all.csv
    cf_files = list(SRC_ROOT.glob("*/*/CF_all.csv"))

    if not cf_files:
        print("No CF_all.csv files found under", SRC_ROOT)
        return

    print(f"Found {len(cf_files)} CF_all.csv file(s) to copy.\n")

    copied = 0
    for src_path in cf_files:
        # relative path like {project}/{model}/CF_all.csv
        rel_path = src_path.relative_to(SRC_ROOT)
        dst_path = DST_ROOT / rel_path

        # make sure target directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # copy (overwrite if exists)
        shutil.copy2(src_path, dst_path)
        print(f"[COPY] {src_path}  -->  {dst_path}")
        copied += 1

    print(f"\nDone. Total files copied: {copied}")

if __name__ == "__main__":
    main()

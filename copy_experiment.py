#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path
from argparse import ArgumentParser


def copy_cfexplainer_experiments(
    src_root: Path,
    dst_root: Path,
    explainer_tag: str = "CfExplainer",
) -> None:
    """
    Copy CfExplainer experiment CSVs from one experiments root to another.

    Assumed structure:
        src_root/{project}/{model}/... {explainer_tag}* ... (e.g. CfExplainer_all.csv)
      -> dst_root/{project}/{model}/... same filenames

    Only files whose *filename* contains `explainer_tag` are copied.
    """
    if not src_root.exists():
        raise FileNotFoundError(f"Source experiments directory does not exist: {src_root}")

    total_files = 0

    for project_dir in src_root.iterdir():
        if not project_dir.is_dir():
            continue
        project = project_dir.name

        for model_dir in project_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            # find explainer files like CfExplainer_all.csv (or whatever tag you use)
            for src_file in model_dir.rglob("*"):
                if not src_file.is_file():
                    continue
                if explainer_tag not in src_file.name:
                    continue

                # mirror relative path under dst_root
                rel_path = src_file.relative_to(src_root)
                dst_file = dst_root / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} -> {dst_file}")
                total_files += 1

    print(f"\nDone. Total files copied: {total_files}")


def main():
    parser = ArgumentParser(
        description="Copy CfExplainer experiment outputs from one experiments root to another."
    )
    parser.add_argument(
        "--src-experiments",
        type=str,
        required=True,
        help="Path to SOURCE experiments root (the one that has CfExplainer_* files).",
    )
    parser.add_argument(
        "--dst-experiments",
        type=str,
        required=True,
        help="Path to DESTINATION experiments root (original one).",
    )
    parser.add_argument(
        "--explainer-tag",
        type=str,
        default="CfExplainer",
        help='Substring used in filenames for this explainer (default: "CfExplainer").',
    )

    args = parser.parse_args()

    src_root = Path(args.src_experiments).resolve()
    dst_root = Path(args.dst_experiments).resolve()

    copy_cfexplainer_experiments(src_root, dst_root, args.explainer_tag)


if __name__ == "__main__":
    main()

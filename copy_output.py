#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path
from argparse import ArgumentParser


def copy_cfexplainer_outputs(src_output: Path, dst_output: Path, explainer_name: str = "CfExplainer") -> None:
    """
    Copy CfExplainer outputs from one OUTPUT root to another.

    Assumed structure:
        src_output/{project}/{explainer_name}/{model}/...
      -> dst_output/{project}/{explainer_name}/{model}/...

    Only files are copied (folders are recreated as needed).
    """

    if not src_output.exists():
        raise FileNotFoundError(f"Source OUTPUT directory does not exist: {src_output}")

    total_files = 0

    # Iterate over projects in source OUTPUT
    for project_dir in src_output.iterdir():
        if not project_dir.is_dir():
            continue
        project = project_dir.name

        src_expl_dir = project_dir / explainer_name
        if not src_expl_dir.exists():
            # No CfExplainer folder for this project, skip
            continue

        # Iterate over models under CfExplainer (RandomForest, SVM, etc.)
        for model_dir in src_expl_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            src_dir = model_dir
            dst_dir = dst_output / project / explainer_name / model_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            # Copy all files (recursively) from src_dir to dst_dir
            for src_file in src_dir.rglob("*"):
                if src_file.is_dir():
                    continue
                rel_path = src_file.relative_to(src_dir)
                dst_file = dst_dir / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} -> {dst_file}")
                total_files += 1

    print(f"\nDone. Total files copied: {total_files}")


def main():
    parser = ArgumentParser(description="Copy CfExplainer outputs from one OUTPUT root to another.")
    parser.add_argument(
        "--src-output",
        type=str,
        required=True,
        help="Path to the SOURCE OUTPUT directory (the one that has CfExplainer results).",
    )
    parser.add_argument(
        "--dst-output",
        type=str,
        required=True,
        help="Path to the DESTINATION OUTPUT directory (original one with LIME/LIME-HPO, etc.).",
    )
    parser.add_argument(
        "--explainer-name",
        type=str,
        default="CfExplainer",
        help='Explainer folder name (default: "CfExplainer").',
    )

    args = parser.parse_args()

    src_output = Path(args.src_output).resolve()
    dst_output = Path(args.dst_output).resolve()

    copy_cfexplainer_outputs(src_output, dst_output, args.explainer_name)


if __name__ == "__main__":
    main()

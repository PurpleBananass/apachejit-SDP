#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path
from argparse import ArgumentParser


def copy_cfexplainer_plans(src_root: Path, dst_root: Path, explainer_name: str = "CfExplainer") -> None:
    """
    Copy CfExplainer plans from one PROPOSED_CHANGES root to another.

    Assumed structure for PLANS:
        src_root/{project}/{model}/{explainer_name}/...
      -> dst_root/{project}/{model}/{explainer_name}/...

    Only files are copied (folders are recreated as needed).
    """

    if not src_root.exists():
        raise FileNotFoundError(f"Source plans directory does not exist: {src_root}")

    total_files = 0

    # Iterate over projects in source PLANS root
    for project_dir in src_root.iterdir():
        if not project_dir.is_dir():
            continue
        project = project_dir.name

        # Under each project, iterate over models (RandomForest, SVM, etc.)
        for model_dir in project_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            src_expl_dir = model_dir / explainer_name
            if not src_expl_dir.exists():
                # This model has no CfExplainer plans, skip
                continue

            dst_expl_dir = dst_root / project / model_name / explainer_name
            dst_expl_dir.mkdir(parents=True, exist_ok=True)

            # Copy all files (recursively) inside this explainer folder
            for src_file in src_expl_dir.rglob("*"):
                if src_file.is_dir():
                    continue
                rel_path = src_file.relative_to(src_expl_dir)
                dst_file = dst_expl_dir / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} -> {dst_file}")
                total_files += 1

    print(f"\nDone. Total files copied: {total_files}")


def main():
    parser = ArgumentParser(description="Copy CfExplainer plans from one PROPOSED_CHANGES root to another.")
    parser.add_argument(
        "--src-plans",
        type=str,
        required=True,
        help="Path to the SOURCE plans root (PROPOSED_CHANGES) that has CfExplainer plans.",
    )
    parser.add_argument(
        "--dst-plans",
        type=str,
        required=True,
        help="Path to the DESTINATION plans root (original PROPOSED_CHANGES).",
    )
    parser.add_argument(
        "--explainer-name",
        type=str,
        default="CfExplainer",
        help='Explainer folder name (default: "CfExplainer").',
    )

    args = parser.parse_args()

    src_root = Path(args.src_plans).resolve()
    dst_root = Path(args.dst_plans).resolve()

    copy_cfexplainer_plans(src_root, dst_root, args.explainer_name)


if __name__ == "__main__":
    main()

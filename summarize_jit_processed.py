#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

# 라벨/메타데이터로 보고 피처에서 제외할 컬럼들
NON_FEATURE_COLS = {
    "commit_id",
    "project",
    "buggy",
    "bugcount",
    "fixcount",
    "author_date",
}


def _safe_project_dir(name: str) -> str:
    """'apache/groovy' -> 'apache_groovy'처럼 디렉토리 이름 안전하게 변환."""
    return name.replace("/", "_").replace("\\", "_")


def _convert_bool_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    true/false/yes/no/t/f/y/n 같은 값만 있는 컬럼은 0/1로 바꿔준다.
    (요약에는 큰 영향 없지만, 피처 카운트 논리 일관성 유지용)
    """
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if s.dtype == bool:
            df[col] = s.astype(int)
            continue

        s_str = s.astype(str).str.lower()
        unique_vals = set(s_str.dropna().unique().tolist())
        bool_like = {"true", "false", "t", "f", "yes", "no", "y", "n"}

        if unique_vals and unique_vals.issubset(bool_like):
            df[col] = s_str.isin({"true", "t", "yes", "y"}).astype(int)
    return df


def load_with_bug_label(csv_path: Path) -> pd.DataFrame:
    """
    원본 CSV를 읽어서 'buggy' 열을 만들어서 반환.
    지원 포맷:
      - apache 스타일: buggy 있음
      - qt/openstack 스타일: bugcount 기반
    """
    df = pd.read_csv(csv_path)

    required_common = {"commit_id", "author_date"}
    missing = required_common - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    if "buggy" in df.columns:
        df["buggy"] = (
            df["buggy"].astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
        ).astype(int)
    elif "bugcount" in df.columns:
        bugcount = pd.to_numeric(df["bugcount"], errors="coerce").fillna(0.0)
        df["buggy"] = (bugcount > 0).astype(int)
    else:
        raise ValueError("CSV must contain either 'buggy' or 'bugcount'.")

    df = _convert_bool_like_columns(df)
    return df


def count_original_features(df: pd.DataFrame) -> int:
    """원본 CSV에서 사용 가능한 피처 수 (NON_FEATURE_COLS 제외)."""
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return len(feat_cols)


def count_processed_features(proc_root: Path, group_name: str) -> int | float:
    """
    전처리된 train.csv에서 피처 수 (target 제외).
    없으면 NaN 반환.
    """
    subdir = proc_root / _safe_project_dir(group_name)
    train_path = subdir / "train.csv"
    if not train_path.exists():
        return np.nan

    tr = pd.read_csv(train_path, index_col=0)
    cols = [c for c in tr.columns if c != "target"]
    return len(cols)


def _compute_duration(sub: pd.DataFrame) -> tuple[str, float]:
    """
    author_date로부터 Duration 문자열과 일 수(duration_days)를 계산.
    - author_date가 유닉스 초(timestamp) 또는 date string이라고 가정.
    """
    # numeric으로 시도 (유닉스 초)
    ts_num = pd.to_numeric(sub["author_date"], errors="coerce")
    if ts_num.notna().any():
        min_ts = float(ts_num.min())
        max_ts = float(ts_num.max())
        # 초 단위라고 가정
        min_dt = pd.to_datetime(min_ts, unit="s", errors="coerce")
        max_dt = pd.to_datetime(max_ts, unit="s", errors="coerce")
    else:
        # 문자열 날짜로 재시도
        dt = pd.to_datetime(sub["author_date"], errors="coerce")
        if dt.notna().any():
            min_dt = dt.min()
            max_dt = dt.max()
            # 일수 차이 계산을 위해 타임스탬프 변환
            min_ts = min_dt.timestamp()
            max_ts = max_dt.timestamp()
        else:
            return "", np.nan

    if pd.isna(min_dt) or pd.isna(max_dt):
        return "", np.nan

    min_str = min_dt.date().isoformat()
    max_str = max_dt.date().isoformat()
    duration_days = (max_ts - min_ts) / 86400.0  # 초 -> 일

    return f"{min_str} ~ {max_str}", duration_days


def summarize_dataset(
    csv_path: Path,
    proc_root: Path,
    group_col: str = "project",
) -> pd.DataFrame:
    """
    한 개 CSV에 대해 요약 정보 DataFrame 생성.

    group_col:
      - 기본: 'project'
      - CSV에 group_col이 없으면 전체를 하나의 그룹으로 취급.
    """
    df = load_with_bug_label(csv_path)
    dataset_name = csv_path.stem

    if group_col in df.columns:
        groups = df[group_col].dropna().unique().tolist()
    else:
        groups = [dataset_name]

    orig_feat_count = count_original_features(df)

    rows: List[Dict] = []

    for g in groups:
        if group_col in df.columns:
            sub = df[df[group_col] == g].copy()
            group_name = str(g)
        else:
            sub = df.copy()
            group_name = dataset_name

        if sub.empty:
            continue

        total = len(sub)
        defective = int(sub["buggy"].sum())
        clean = int(total - defective)

        defect_ratio = defective / total if total > 0 else np.nan
        clean_ratio = clean / total if total > 0 else np.nan

        # 논문 스타일: "285 (11.79%)"
        if total > 0:
            defective_str = f"{defective} ({defect_ratio * 100:.2f}%)"
            clean_str = f"{clean} ({clean_ratio * 100:.2f}%)"
        else:
            defective_str = f"{defective} (0.00%)"
            clean_str = f"{clean} (0.00%)"

        proc_feat_count = count_processed_features(proc_root, group_name)

        # Duration 계산
        duration_str, duration_days = _compute_duration(sub)

        rows.append(
            {
                "Dataset": dataset_name,
                "Group": group_name,                     # project 또는 release
                "Total": total,                          # 총 인스턴스 수
                "Defective": defective,                  # 디펙티브 인스턴스 수
                "Clean": clean,                          # 클린 인스턴스 수
                "Defect_Ratio": defect_ratio,            # defective / total (0~1)
                "Clean_Ratio": clean_ratio,              # clean / total (0~1)
                "Defective (count, %)": defective_str,   # "285 (11.79%)"
                "Clean (count, %)": clean_str,           # "2132 (88.21%)"
                "Duration": duration_str,                # "2005-12-14 ~ 2019-12-03"
                "Duration_days": duration_days,          # 465.3 같은 값
                "Orig_Features": orig_feat_count,        # 오리지널 피처 수
                "Post_Features": proc_feat_count,        # 전처리 후 피처 수
            }
        )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Summarize JIT datasets: per project/release, "
            "show clean/defective counts, percentages, total, duration, "
            "original feature count, and post-preprocess feature count."
        )
    )
    ap.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to original CSV (apachejit_total.csv, qt.csv, openstack.csv, ...).",
    )
    ap.add_argument(
        "--proc-dir",
        type=str,
        required=True,
        help=(
            "Root directory of processed data "
            "(the one where preprocess_jit.py wrote per-project subdirs)."
        ),
    )
    ap.add_argument(
        "--group-col",
        type=str,
        default="project",
        help=(
            "Column to group by (default: 'project'). "
            "If this column does not exist, the whole CSV is treated as one group. "
            "For release-level summary, pass e.g. --group-col release."
        ),
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to save the summary CSV (e.g., ./summary.csv).",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    proc_root = Path(args.proc_dir)

    df_summary = summarize_dataset(csv_path, proc_root, group_col=args.group_col)

    if df_summary.empty:
        print("[WARN] No groups found / summary is empty.")
        return

    # 논문용으로 자주 볼 만한 주요 컬럼만 출력
    cols_for_view = [
        "Dataset",
        "Group",
        "Defective (count, %)",
        "Clean (count, %)",
        "Total",
        "Duration",
        "Orig_Features",
        "Post_Features",
    ]

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_summary[cols_for_view].to_string(index=False))

    # 파일 저장 옵션
    if args.out:
        out_path = Path(args.out)
        df_summary.to_csv(out_path, index=False)
        print(f"\n[INFO] Saved summary to {out_path}")


if __name__ == "__main__":
    main()

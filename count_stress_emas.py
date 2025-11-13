#!/usr/bin/env python3
"""
count_stress_emas.py
--------------------
Scan `hp/hp{PID}/questionnaire_responses_ID{PID}.csv` for a list of participants,
and compute per-participant counts of: total stress EMAs and High-stress EMAs (> threshold).

USAGE
-----
# from your project root (the directory that contains `hp/`)
python count_stress_emas.py --pids 10,15,16,17,18,20,22,23,24,25,26,30,31,32,33,34,35,36,39,40 \
                            --root . --threshold 5 --out stress_ema_counts.csv

Notes
-----
- Expects columns: `stressLevel_value` (float in [0..10]) and `local_created_at` (timestamp).
  If columns differ, the script will try common fallbacks (see STRESS_COL_CANDIDATES and TIME_COL_CANDIDATES).
- "High stress" is defined as strictly greater than the threshold (e.g., > 5 → 6–10).
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys

STRESS_COL_CANDIDATES = [
    "stressLevel_value", "stress_level", "stress", "stress_value", "stress_score"
]
TIME_COL_CANDIDATES = [
    "local_created_at", "created_at", "timestamp", "time", "datetime"
]

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def summarize_pid(pid: str, root: Path, threshold: float):
    csv_path = root / "hp" / f"hp{pid}" / f"questionnaire_responses_ID{pid}.csv"
    if not csv_path.exists():
        return {
            "pid": pid, "file_found": False,
            "n_ema": 0, "n_high": 0,
            "pct_high": np.nan, "first_date": None, "last_date": None
        }
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {
            "pid": pid, "file_found": False, "error": str(e),
            "n_ema": 0, "n_high": 0, "pct_high": np.nan,
            "first_date": None, "last_date": None
        }

    stress_col = find_col(df, STRESS_COL_CANDIDATES)
    time_col   = find_col(df, TIME_COL_CANDIDATES)

    if stress_col is None:
        return {
            "pid": pid, "file_found": True, "error": "stress column not found",
            "n_ema": 0, "n_high": 0, "pct_high": np.nan,
            "first_date": None, "last_date": None
        }

    s = pd.to_numeric(df[stress_col], errors="coerce")
    n_ema  = int(s.notna().sum())
    n_high = int((s > threshold).sum())

    first_date = None
    last_date  = None
    if time_col is not None:
        t = pd.to_datetime(df[time_col], errors="coerce")
        if t.notna().any():
            first_date = t.min()
            last_date  = t.max()

    pct_high = (n_high / n_ema) if n_ema else np.nan
    return {
        "pid": pid, "file_found": True,
        "n_ema": n_ema, "n_high": n_high, "pct_high": pct_high,
        "first_date": first_date, "last_date": last_date
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True,
                    help="Comma-separated list of participant IDs, e.g., 10,15,16,...")
    ap.add_argument("--root", default=".", help="Project root (contains hp/)")
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="High-stress threshold; counts values strictly greater than this")
    ap.add_argument("--out", default="stress_ema_counts.csv",
                    help="Output CSV filename")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    pids = [p.strip().zfill(2) for p in args.pids.split(",") if p.strip()]

    rows = [summarize_pid(pid, root, args.threshold) for pid in pids]
    df = pd.DataFrame(rows).sort_values("pid")

    # Compute cohort-level summary
    df_found = df[df["file_found"] == True]
    n_participants = len(df_found)
    total_emas  = int(df_found["n_ema"].sum())
    total_high  = int(df_found["n_high"].sum())
    mean_emas   = float(df_found["n_ema"].mean()) if n_participants else float("nan")
    sd_emas     = float(df_found["n_ema"].std(ddof=1)) if n_participants > 1 else float("nan")
    mean_high   = float(df_found["n_high"].mean()) if n_participants else float("nan")
    sd_high     = float(df_found["n_high"].std(ddof=1)) if n_participants > 1 else float("nan")
    overall_high_pct = (total_high / total_emas) if total_emas else float("nan")

    # Save per-participant table
    df.to_csv(args.out, index=False)

    # Print a human-friendly summary
    print("\nPer-participant stress EMA counts saved →", args.out)
    print("\nCohort summary (files found):")
    print(f"  Participants with EMA files: {n_participants}")
    print(f"  Total EMAs: {total_emas}")
    print(f"  Total High-stress EMAs (> {args.threshold}): {total_high}  "
          f"({overall_high_pct*100:.1f}% of EMAs)" if total_emas else
          f"  Total High-stress EMAs (> {args.threshold}): {total_high}")
    print(f"  Mean (SD) EMAs per participant: {mean_emas:.1f} ({sd_emas:.1f})")
    print(f"  Mean (SD) High-stress EMAs per participant: {mean_high:.1f} ({sd_high:.1f})")

if __name__ == "__main__":
    main()

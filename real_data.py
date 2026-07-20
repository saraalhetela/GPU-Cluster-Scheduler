"""
real_data.py

CLI:
    python real_data.py parse-philly philly-traces/cluster_job_log.json
        -> data/jobs_real_philly.csv
    python real_data.py parse-alibaba pai_job_table.csv pai_task_table.csv
        -> data/jobs_real_alibaba.csv
    python real_data.py build
        -> data/jobs.csv  (real + guaranteed edge cases + random synthetic)
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from data_generation import generate_jobs, generate_edge_case_jobs

SEED = 42

# --- parsing ---
MAX_GPU_HOURS = 96.0
VALID_MAX_GPUS = [2, 4, 6, 8]
PRIORITY_BUCKETS = ["Low", "Medium", "High"]
PRIORITY_WEIGHTS = [0.5, 0.35, 0.15]
REAL_PATH = "data/jobs_real_philly.csv"
REAL_PATH_ALIBABA = "data/jobs_real_alibaba.csv"

ALIBABA_JOB_COLS = ["job_name", "inst_id", "user", "status", "start_time", "end_time"]
ALIBABA_TASK_COLS = [
    "job_name", "task_name", "inst_num", "status", "start_time", "end_time",
    "plan_cpu", "plan_mem", "plan_gpu", "gpu_type",
]

# --- merging ---
N_JOBS = 200
MAX_REAL = 170     # cap real jobs even if more are available, so the
                    # guaranteed edge cases + some random variety always
                    # have room
N_EDGE_CASES = 10
OUT_PATH = "data/jobs.csv"

REQUIRED_COLS = [
    "job_id", "arrival_time", "deadline", "gpu_hours_required",
    "priority", "max_gpus", "initial_progress",
]


# ---------------------------------------------------------------- parsing --


def _hour_of_day_alibaba(seconds) -> float:
    """
    Alibaba's start_time/end_time are numeric seconds, NOT date strings --
    desensitized Unix-like timestamps with a constant offset applied. Per
    the trace's own README: interpreting them AS real Unix timestamps in
    UTC+8 ("Asia/Shanghai") recovers the correct real hour-of-day and
    day-of-week, even though the date/month/year is fake. So: treat the
    raw number as a Unix timestamp, convert to UTC+8, read the hour.
    Returns None if the value can't be parsed as a number.
    """
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return None
    dt = datetime.fromtimestamp(seconds, tz=timezone(timedelta(hours=8)))
    return dt.hour + dt.minute / 60.0


def _duration_hours_alibaba(start_seconds, end_seconds):
    try:
        st, et = float(start_seconds), float(end_seconds)
    except (TypeError, ValueError):
        return None
    if et <= st:
        return None
    return (et - st) / 3600.0


def _vc_priority(vc: str) -> str:
    """Deterministic hash bucket -- same vc always maps to same priority."""
    h = int(hashlib.md5(vc.encode()).hexdigest(), 16)
    r = (h % 1000) / 1000.0
    cum = 0.0
    for label, w in zip(PRIORITY_BUCKETS, PRIORITY_WEIGHTS):
        cum += w
        if r < cum:
            return label
    return PRIORITY_BUCKETS[-1]


def _snap_max_gpus(n: int) -> int:
    return min(VALID_MAX_GPUS, key=lambda x: abs(x - n))


def parse_job_log(path: str, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    with open(path) as f:
        raw_jobs = json.load(f)

    rows = []
    for j in raw_jobs:
        if j.get("status") != "Pass":
            continue  # Failed/Killed jobs have unreliable end_time
        attempts = j.get("attempts") or []
        if not attempts:
            continue

        submitted = _parse_ts_philly(j.get("submitted_time"))
        if submitted is None:
            continue

        total_gpu_hours = 0.0
        peak_gpus = 0
        valid_attempt = False
        for a in attempts:
            st, et = _parse_ts_philly(a.get("start_time")), _parse_ts_philly(a.get("end_time"))
            if st is None or et is None or et <= st:
                continue
            n_gpus = sum(len(d.get("gpus", [])) for d in a.get("detail", []))
            if n_gpus <= 0:
                continue
            duration_h = (et - st).total_seconds() / 3600.0
            total_gpu_hours += duration_h * n_gpus
            peak_gpus = max(peak_gpus, n_gpus)
            valid_attempt = True

        if not valid_attempt or peak_gpus == 0:
            continue

        gpu_hours_required = round(min(total_gpu_hours, MAX_GPU_HOURS), 2)
        max_gpus = _snap_max_gpus(peak_gpus)
        arrival_time = round(submitted.hour + submitted.minute / 60.0, 2)

        min_hours_needed = gpu_hours_required / max_gpus if max_gpus else gpu_hours_required
        slack_factor = rng.uniform(1.1, 4.0)
        deadline = arrival_time + max(min_hours_needed * slack_factor, 1.0)
        deadline = round(min(deadline, 24.0), 2)
        if deadline <= arrival_time:
            continue

        rows.append(
            {
                "job_id": j["jobid"],
                "arrival_time": arrival_time,
                "deadline": deadline,
                "gpu_hours_required": gpu_hours_required,
                "priority": _vc_priority(j.get("vc", "")),
                "max_gpus": max_gpus,
                "initial_progress": 0.0,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("arrival_time").reset_index(drop=True)


def _check_columns(df: pd.DataFrame, expected: list, file_label: str):
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_label}: expected columns {missing} not found. "
            f"Actual columns in the file: {list(df.columns)}. "
            f"The Alibaba trace's exact schema can differ across mirrors/"
            f"releases -- open the CSV yourself (pd.read_csv(path, nrows=5)) "
            f"and update ALIBABA_JOB_COLS / ALIBABA_TASK_COLS in real_data.py "
            f"to match, or pass header=None + your own names if the file "
            f"ships without a header row."
        )


def _read_alibaba_csv(path: str, expected_cols: list) -> pd.DataFrame:
    """
    The Alibaba PAI trace is documented to ship WITHOUT a header row (the
    schema lives in the README/paper, not the file). This is exactly what
    bit the real download: pandas silently treated the first data row as
    the header, so every column name came back wrong (a real job_name hash
    where 'job_name' should have been, etc.) -- _check_columns then
    correctly rejected it.

    Try a normal read first, in case a particular mirror *does* include a
    header row. If the columns don't match what's expected, assume it's
    headerless and retry with the documented column order. If even that
    doesn't line up (wrong column count), raise a clear, specific error --
    don't guess further.
    """
    df = pd.read_csv(path)
    if list(df.columns) == expected_cols:
        return df

    df_retry = pd.read_csv(path, header=None, names=expected_cols)
    if len(df_retry.columns) != len(expected_cols):
        raise ValueError(
            f"{path}: normal read gave columns {list(df.columns)} (doesn't "
            f"match expected {expected_cols}), and retrying headerless "
            f"produced {len(df_retry.columns)} columns instead of the "
            f"expected {len(expected_cols)}. This mirror's schema doesn't "
            f"match what's documented -- run pd.read_csv('{path}', nrows=5) "
            f"yourself and update ALIBABA_JOB_COLS / ALIBABA_TASK_COLS."
        )
    return df_retry


def parse_alibaba_pai_trace(job_table_path: str, task_table_path: str, seed: int = SEED) -> pd.DataFrame:
    """
    Parses the REAL Alibaba PAI GPU cluster trace 2020 (alibaba/clusterdata,
    NSDI '22) into jobs.csv-shaped rows. Recommended fallback if the Philly
    trace (Source A) is unreachable due to Git LFS quota issues.

    Confirmed against the trace's actual published README (previous version
    of this function guessed the timestamp format wrong -- fixed here):
      - pai_job_table: job_name, inst_id, user, status, start_time, end_time.
        status is one of 'Running'/'Terminated'/'Failed'/'Waiting' --
        only 'Terminated' means the job actually finished successfully.
      - pai_task_table: job_name, task_name, inst_num, status, start_time,
        end_time, plan_cpu, plan_mem, plan_gpu, gpu_type. plan_gpu is a
        PERCENTAGE of one GPU per instance (50.0 = half a GPU), not a
        whole-GPU count.
      - start_time/end_time in BOTH tables are numeric seconds, NOT date
        strings -- desensitized Unix-like timestamps with a constant
        offset applied. Per the README, interpreting the raw number AS a
        real Unix timestamp in UTC+8 ("Asia/Shanghai") recovers the
        correct real hour-of-day (the date itself is fake, the time-of-day
        isn't). See _hour_of_day_alibaba().

    job_table gives one row per job (status, start/end time).
    task_table gives one row per task within a job -- summed across a
    job's tasks, (plan_gpu/100)*inst_num is the real total GPU demand.
    """
    rng = np.random.default_rng(seed)

    job_df = _read_alibaba_csv(job_table_path, ALIBABA_JOB_COLS)
    task_df = _read_alibaba_csv(task_table_path, ALIBABA_TASK_COLS)

    # real total GPU demand per job: sum over tasks of (plan_gpu% / 100) * inst_num
    task_df["gpu_count"] = (task_df["plan_gpu"].fillna(0) / 100.0) * task_df["inst_num"].fillna(1)
    gpu_by_job = task_df.groupby("job_name")["gpu_count"].sum()

    status_counts = job_df["status"].value_counts()
    if "Terminated" not in status_counts.index:
        print(f"WARNING: no rows with status=='Terminated' found. Actual status "
              f"value counts:\n{status_counts}\nNothing will be parsed -- the "
              f"trace's status vocabulary may have changed since this was written.")

    rows = []
    for _, j in job_df.iterrows():
        if str(j.get("status")) != "Terminated":
            continue

        arrival_time = _hour_of_day_alibaba(j.get("start_time"))
        duration_h = _duration_hours_alibaba(j.get("start_time"), j.get("end_time"))
        if arrival_time is None or duration_h is None:
            continue

        peak_gpus_raw = gpu_by_job.get(j["job_name"], 0.0)
        if peak_gpus_raw <= 0:
            continue

        max_gpus = _snap_max_gpus(max(1, round(peak_gpus_raw)))
        gpu_hours_required = round(min(duration_h * max_gpus, MAX_GPU_HOURS), 2)
        arrival_time = round(arrival_time, 2)

        min_hours_needed = gpu_hours_required / max_gpus if max_gpus else gpu_hours_required
        slack_factor = rng.uniform(1.1, 4.0)
        deadline = arrival_time + max(min_hours_needed * slack_factor, 1.0)
        deadline = round(min(deadline, 24.0), 2)
        if deadline <= arrival_time:
            continue

        rows.append(
            {
                "job_id": str(j["job_name"]),
                "arrival_time": arrival_time,
                "deadline": deadline,
                "gpu_hours_required": gpu_hours_required,
                "priority": _vc_priority(str(j.get("user", ""))),  # proxy, same
                                                                     # hash-bucket approach as Philly's vc
                "max_gpus": max_gpus,
                "initial_progress": 0.0,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("WARNING: 0 rows survived parsing -- check job_df['status'].value_counts() "
              "and the accepted-status list in parse_alibaba_pai_trace(), the real "
              "status vocabulary may differ from what's assumed here.")
        return df
    return df.sort_values("arrival_time").reset_index(drop=True)


# ----------------------------------------------------------------- merging --

def build_hybrid(n_jobs: int = N_JOBS, seed: int = SEED) -> pd.DataFrame:
    real_parts = []
    if os.path.exists(REAL_PATH):
        philly_df = pd.read_csv(REAL_PATH)
        philly_df["source"] = "real_philly"
        real_parts.append(philly_df)
    if os.path.exists(REAL_PATH_ALIBABA):
        alibaba_df = pd.read_csv(REAL_PATH_ALIBABA)
        alibaba_df["source"] = "real_alibaba"
        real_parts.append(alibaba_df)

    if real_parts:
        real_df = pd.concat(real_parts, ignore_index=True)
    else:
        print(f"WARNING: neither {REAL_PATH} nor {REAL_PATH_ALIBABA} found -- run "
              f"'python real_data.py parse-philly <path>' or "
              f"'python real_data.py parse-alibaba <job_table> <task_table>' first. "
              f"Falling back to 100% synthetic for now.")
        real_df = pd.DataFrame(columns=REQUIRED_COLS + ["source"])

    n_real = min(len(real_df), MAX_REAL, n_jobs)
    real_df = (
        real_df.sample(n=n_real, random_state=seed).reset_index(drop=True)
        if n_real else real_df
    )

    n_remaining = n_jobs - n_real
    n_edge = min(N_EDGE_CASES, n_remaining)
    n_random = max(n_remaining - n_edge, 0)

    edge_df = generate_edge_case_jobs(n_edge, seed=seed) if n_edge else pd.DataFrame(columns=REQUIRED_COLS)
    edge_df["source"] = "synthetic_edge_case"

    random_df = generate_jobs(n_random, seed=seed) if n_random else pd.DataFrame(columns=REQUIRED_COLS)
    random_df["source"] = "synthetic_random"

    combined = pd.concat([real_df, edge_df, random_df], ignore_index=True)
    combined = combined.sort_values("arrival_time").reset_index(drop=True)
    combined["job_id"] = combined["job_id"].astype(str)
    return combined

DEFAULT_PHILLY_PATHS = [
    "cluster_job_log.json",
    "philly-traces/cluster_job_log.json",
]
DEFAULT_ALIBABA_JOB_PATH = "pai_job_table.csv"
DEFAULT_ALIBABA_TASK_PATH = "pai_task_table.csv"


def run_all():
    print("=== real_data.py: running full pipeline ===\n")

    print("[1/2] Alibaba PAI trace")
    if os.path.exists(DEFAULT_ALIBABA_JOB_PATH) and os.path.exists(DEFAULT_ALIBABA_TASK_PATH):
        try:
            df = parse_alibaba_pai_trace(
                DEFAULT_ALIBABA_JOB_PATH,
                DEFAULT_ALIBABA_TASK_PATH,
            )
            df.to_csv(REAL_PATH_ALIBABA, index=False)
            print(f"  parsed {len(df)} real jobs -> {REAL_PATH_ALIBABA}")
        except Exception as e:
            print(f"  parsing failed: {e}")
    else:
        print("  Alibaba trace files not found.")

    print("\n[2/2] Building jobs.csv")
    df = build_hybrid()
    df.to_csv(OUT_PATH, index=False)

    counts = df["source"].value_counts()
    print(df.head(10).to_string(index=False))

    print(f"\n{len(df)} total jobs -> {OUT_PATH}")
    for source, n in counts.items():
        print(f"  {source}: {n} ({n/len(df)*100:.1f}%)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("all", "run"):
        run_all()

    elif mode == "parse-alibaba":
        if len(sys.argv) < 4:
            print("Usage: python real_data.py parse-alibaba <pai_job_table.csv> <pai_task_table.csv>")
            sys.exit(1)

        df = parse_alibaba_pai_trace(sys.argv[2], sys.argv[3])
        df.to_csv(REAL_PATH_ALIBABA, index=False)

        if not df.empty:
            print(df.head(10).to_string(index=False))

        print(f"\n{len(df)} real jobs parsed -> {REAL_PATH_ALIBABA}")

    elif mode == "build":
        df = build_hybrid()
        df.to_csv(OUT_PATH, index=False)
        print(f"{len(df)} total jobs -> {OUT_PATH}")

    else:
        print("Usage:")
        print("  python real_data.py")
        print("  python real_data.py parse-alibaba <pai_job_table.csv> <pai_task_table.csv>")
        print("  python real_data.py build")
        sys.exit(1)

"""Measure the RAM/time cost of the dict-in-artifact design (CP-I6).

``train_seen`` and ``user_category_history`` are plain Python dicts embedded in the
joblib artifact and loaded fully into the serving process's memory (see
src/train_baseline.py and src/api/main.py's ``get_engine``). This script builds the
same shape of structure at increasing user counts and measures Python-attributable
memory (via ``tracemalloc``, stdlib, no extra dependency) plus dump/load time, so the
decision to change architecture (CP-I6) is based on a number, not a guess.

Run:
  python scripts/benchmark_scaling.py
  python scripts/benchmark_scaling.py --users 100000
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

import joblib
import numpy as np

CATEGORIES = ["xanh_sm", "vinpearl", "vinmec", "shopping"]


def build_artifact_shapes(n_users: int, items_per_user: int = 15) -> dict:
    rng = np.random.default_rng(0)
    train_seen = {}
    user_category_history = {}
    for user_index in range(n_users):
        user_id = f"user_{user_index:08d}"
        items = [f"item_{i:06d}" for i in rng.integers(0, 50_000, size=items_per_user)]
        train_seen[user_id] = items
        user_category_history[user_id] = {
            category: int(rng.integers(1, 20)) for category in rng.choice(CATEGORIES, size=2, replace=False)
        }
    return {"train_seen": train_seen, "user_category_history": user_category_history}


def benchmark(n_users: int, tmp_dir: Path) -> dict:
    print(f"Building artifact shape for {n_users:,} users...", flush=True)
    tracemalloc.start()
    build_started = time.perf_counter()
    artifact = build_artifact_shapes(n_users)
    build_seconds = time.perf_counter() - build_started
    _, build_peak_bytes = tracemalloc.get_traced_memory()

    dump_path = tmp_dir / f"artifact_{n_users}.pkl"
    print(f"Dumping temporary artifact to {dump_path}...", flush=True)
    dump_started = time.perf_counter()
    joblib.dump(artifact, dump_path)
    dump_seconds = time.perf_counter() - dump_started
    file_size_mb = dump_path.stat().st_size / (1024 * 1024)

    del artifact
    tracemalloc.stop()

    tracemalloc.start()
    print(f"Loading temporary artifact from {dump_path}...", flush=True)
    load_started = time.perf_counter()
    loaded = joblib.load(dump_path)
    load_seconds = time.perf_counter() - load_started
    _, load_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del loaded
    dump_path.unlink()

    return {
        "n_users": n_users,
        "build_seconds": round(build_seconds, 3),
        "build_peak_mb": round(build_peak_bytes / (1024 * 1024), 1),
        "dump_seconds": round(dump_seconds, 3),
        "file_size_mb": round(file_size_mb, 1),
        "load_seconds": round(load_seconds, 3),
        "load_peak_mb": round(load_peak_bytes / (1024 * 1024), 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark memory and time cost for dict-in-artifact serving state."
    )
    parser.add_argument(
        "--users",
        type=int,
        nargs="+",
        help="User counts to benchmark. Example: --users 100000 500000",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tmp_dir = Path("reports")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    user_counts = args.users or [100_000, 500_000, 1_000_000]
    rows = []
    for n_users in user_counts:
        row = benchmark(n_users, tmp_dir)
        rows.append(row)
        print(row, flush=True)

    lines = [
        "# Scaling benchmark: dict-in-artifact design (CP-I6)",
        "",
        "Measures `train_seen` + `user_category_history` at increasing user counts, "
        "shaped like the real artifact (15 seen items/user, 2 categories/user). "
        "`*_peak_mb` is Python-attributable memory via `tracemalloc`, not full process "
        "RSS — treat it as a lower bound, real RSS will be somewhat higher.",
        "",
        "| Users | Build time (s) | Build peak (MB) | Dump time (s) | File size (MB) | Load time (s) | Load peak (MB) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['n_users']:,} | {row['build_seconds']} | {row['build_peak_mb']} | "
            f"{row['dump_seconds']} | {row['file_size_mb']} | {row['load_seconds']} | "
            f"{row['load_peak_mb']} |"
        )
    lines += [
        "",
        "## Reading this",
        "",
        "Extrapolate `load_peak_mb` linearly against the expected real user count for the "
        "target deployment. If projected load time or memory would degrade cold-start "
        "latency or exceed the serving instance's memory budget, that is the trigger for "
        "moving `train_seen`/`user_category_history` out of the artifact into an external "
        "store (Redis/SQLite) as described in IMPROVEMENT_PLAN.md CP-I6 — not before, since "
        "premature optimization here adds an operational dependency for no measured benefit.",
    ]
    output = Path("reports/scaling_benchmark.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()

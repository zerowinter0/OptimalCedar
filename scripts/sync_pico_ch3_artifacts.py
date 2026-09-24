"""Copy the chapter-3 deliverable files into the repository (small files only).

Usage:
  python -u scripts/sync_pico_ch3_artifacts.py \
      --run-dir outputs/pico_ch3_20260924 --dest docs/mechanism_20260923/pico_ch3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

SMALL = (
    "README.md", "audit.json", "protocol.json", "predictions.csv",
    "measurements.csv", "boundary_results.csv", "summary.csv",
    "figure_data.json", "component_predictions.json", "operator_results.csv",
    "reorder_summary.csv", "reorder_analysis.json", "pipeline_scoring.csv",
    "pipeline_scoring.json", "expA_corrected.csv", "expA_analysis.json",
    "profile/synthetic_anchor.json",
    "expA1_main/expA1_summary.csv", "expA1_main/expA1_config.json",
    "expA1_main/expA1_run_order.json", "expA1_main/expA1_verification.json",
)
LARGE = (
    "expA1_main/expA1_raw.csv",
    "expA1_main/expA1_operator_timing.csv",
    "expA1_main.log",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size,
                             "sha256": sha256(path)}
    if path.suffix == ".csv":
        with path.open() as handle:
            entry["rows"] = sum(1 for _ in handle) - 1
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--max-small-bytes", type=int, default=1_500_000)
    args = parser.parse_args()
    run, dest = args.run_dir.resolve(), args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for relative in SMALL:
        source = run / relative
        if not source.exists() or source.stat().st_size > args.max_small_bytes:
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(relative)
    for relative in LARGE:
        source = run / relative
        if source.exists() and source.stat().st_size <= args.max_small_bytes:
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(relative)
    large = [
        describe(run / relative)
        for relative in LARGE
        if (run / relative).exists()
        and str(relative) not in copied
    ]
    (dest / "MANIFEST.json").write_text(
        json.dumps(
            {
                "run_dir": str(run),
                "copied_files": copied,
                "large_artifacts_kept_in_outputs": large,
                "notes": (
                    "Large per-batch/per-event measurement files stay under "
                    "outputs/; the manifest records path, size, rows and sha256."
                ),
            },
            indent=2,
        )
    )
    print(f"copied {len(copied)} files, registered {len(large)} large artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

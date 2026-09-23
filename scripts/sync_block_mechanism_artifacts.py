"""Copy the mechanism experiment's small artifacts into the repository.

Large measurement files (per-event `operator_timing.csv`, control-raw CSVs,
the captured input blob, logs) stay in ``outputs/``; the manifest records their
path, size, row count and sha256 instead.

Usage:
  python -u scripts/sync_block_mechanism_artifacts.py \
      --run-dir outputs/simclrv2_fusion_offload_mechanism_20260923 \
      --dest docs/mechanism_20260923
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

SMALL_FILES = (
    "README.md",
    "run_manifest.json",
    "service_raw.csv",
    "service_summary.csv",
    "service_summary_repeats.csv",
    "service_meta.json",
    "service_verification.json",
    "operator_timing.columns.json",
    "operator_timing_summary.csv",
    "cedar_cost_breakdown.json",
    "figure_data.json",
    "figure_data.md",
    "pipeline_results.csv",
    "actor_probe_R-F_w1.json",
    "actor_probe_R-F_w64.json",
    "actor_probe_R-U_w1.json",
    "actor_placement_check.json",
    "cpu_topology.json",
    "wrapper_overhead.json",
    "burst_idle_probe_remote.json",
    "burst_idle_probe_remote_gap80.json",
    "burst_idle_probe_paired.json",
    "tensor_state_probe.json",
    "reaggregation_report.json",
    "control_interleaved/service_summary.csv",
    "control_interleaved/service_summary_repeats.csv",
    "control_samecore/service_summary.csv",
    "control_samecore/service_summary_repeats.csv",
    "repeat2/service_summary.csv",
    "repeat2/service_summary_repeats.csv",
    "instrument_overhead_full/service_summary.csv",
    "instrument_overhead_full/service_summary_repeats.csv",
    "instrument_overhead_none/service_summary.csv",
    "instrument_overhead_none/service_summary_repeats.csv",
    "probe_ray_alone/service_summary.csv",
    "probe_ray_alone/service_summary_repeats.csv",
)

LARGE_PATTERNS = (
    "inputs/block_inputs.pt",
    "operator_timing.csv",
    "control_interleaved/operator_timing.csv",
    "control_samecore/operator_timing.csv",
    "control_interleaved/service_raw.csv",
    "control_samecore/service_raw.csv",
    "repeat2/service_raw.csv",
    "probe_ray_alone/service_raw.csv",
    "instrument_overhead_full/service_raw.csv",
    "instrument_overhead_none/service_raw.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
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
    run_dir, dest = args.run_dir.resolve(), args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    skipped: List[Dict[str, Any]] = []
    for relative in SMALL_FILES:
        source = run_dir / relative
        if not source.exists():
            continue
        if source.stat().st_size > args.max_small_bytes:
            skipped.append({**describe(source), "reason": "larger than limit"})
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(relative)

    manifest: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "copied_files": copied,
        "copied_sizes": {
            relative: (dest / relative).stat().st_size for relative in copied
        },
        "large_artifacts_kept_in_outputs": [],
        "notes": (
            "Large per-event/per-batch files stay under outputs/; this manifest "
            "carries their path, size, row count and sha256 for verification."
        ),
    }
    for relative in LARGE_PATTERNS:
        source = run_dir / relative
        if source.exists():
            manifest["large_artifacts_kept_in_outputs"].append(describe(source))
    manifest["large_artifacts_kept_in_outputs"].extend(skipped)
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("copied_files",)}, indent=2))
    print(f"copied {len(copied)} files to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Copy the fusion-discount experiment's small artifacts into the repository.

Large files (per-batch raw CSVs, per-event operator timings, the 270 MB real
block input blob, logs) stay under ``outputs/``; the manifest records their
path, size and sha256.

Usage:
  python -u scripts/sync_fusion_discount_artifacts.py \
      --run-dir outputs/fusion_discount_20260923 \
      --dest docs/mechanism_20260923/fusion_discount
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

SMALL_PATTERNS = (
    "README.md",
    "candidate_blocks.json",
    "cedar_costs.json",
    "figure_data.json",
    "figure_data.csv",
    "figure_data.md",
    "expA/expA_config.json",
    "expA/expA_summary.csv",
    "expA/expA_raw.csv",
    "expA/expA_verification.json",
    "expA/expA_run_order.json",
    "expA_instrument_off/expA_off_config.json",
    "expA_instrument_off/expA_off_summary.csv",
    "expA_instrument_off/expA_off_raw.csv",
    "expB/expB_config.json",
    "expB/expB_summary.csv",
    "expB/expB_raw.csv",
    "expB/expB_verification.json",
    "expB/expB_run_order.json",
    "expC/plans/U.yaml",
    "expC/plans/P.yaml",
    "expC/plans/F.yaml",
    "expC/scores/U.json",
    "expC/scores/P.json",
    "expC/scores/F.json",
    "expC/actor_probe_expC_U.json",
    "expC/actor_probe_expC_P.json",
    "expC/actor_probe_expC_F.json",
)
LARGE_GLOBS = (
    "expA/expA_raw.csv",
    "expA/expA_operator_timing.csv",
    "expB/expB_raw.csv",
    "expB/expB_operator_timing.csv",
    "expB/inputs/block_inputs.pt",
    "expC/throughput/*.json",
    "expA_instrument_off/expA_off_raw.csv",
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
    parser.add_argument("--max-small-bytes", type=int, default=400_000)
    args = parser.parse_args()
    run_dir, dest = args.run_dir.resolve(), args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    for relative in SMALL_PATTERNS:
        source = run_dir / relative
        if not source.exists():
            continue
        if source.stat().st_size > args.max_small_bytes:
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(relative)
    # experiment C throughput JSONs are small
    for source in sorted((run_dir / "expC/throughput").glob("*.json")):
        target = dest / "expC/throughput" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(str(target.relative_to(dest)))

    large: List[Dict[str, Any]] = []
    for pattern in LARGE_GLOBS:
        for source in sorted(run_dir.glob(pattern)):
            if source.is_file() and str(source.relative_to(run_dir)) not in copied:
                large.append(describe(source))
    manifest = {
        "run_dir": str(run_dir),
        "copied_files": copied,
        "large_artifacts_kept_in_outputs": large,
        "notes": (
            "Large measurement files (per-batch raw, per-event timings, the real "
            "block input blob, logs) stay under outputs/; this manifest carries "
            "their path, size, row count and sha256."
        ),
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"copied {len(copied)} files; registered {len(large)} large artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

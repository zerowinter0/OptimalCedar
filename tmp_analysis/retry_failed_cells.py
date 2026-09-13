"""Re-run the cells a ten-workload matrix marked failed, then merge them back.

The matrix runner never resumes: it re-runs every requested cell.  This driver
therefore replays only the failed cells in a sibling directory, reusing the
main run's profiles so the retry measurement stays comparable, and finally
copies the retried results and status entries back into the main run.

Usage (inside the container):
  python -u tmp_analysis/retry_failed_cells.py outputs/pico_ten_workloads_20260913b
  ... --dry-run          # list what would be retried
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
RUNNER = ROOT / "tmp_analysis/run_ten_workload_matrix.py"
RETRYABLE = {"failed", "timeout", "profile_failed"}


def known_cells():
    """Workload/optimizer names the runner still defines."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("matrix", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["matrix"] = module
    spec.loader.exec_module(module)
    workloads = {name for name, _, _, _ in module.WORKLOADS}
    return {f"{name}:{opt}" for name in workloads for opt in module.OPTIMIZERS}


def failed_cells(run_dir: Path):
    payload = json.loads((run_dir / "status.json").read_text())
    cells = payload.get("cells", {})
    known = known_cells()
    failed = [
        key
        for key, entry in cells.items()
        # Cells for workloads or optimizers that no longer exist (removed
        # workloads, per-workload profile failures) are not retryable.
        if entry.get("status") in RETRYABLE and key in known
    ]
    workloads, optimizers = set(), set()
    for key in failed:
        workload, optimizer = key.split(":", 1)
        workloads.add(workload)
        optimizers.add(optimizer)
    return failed, sorted(workloads), sorted(optimizers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--retry-out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    retry_dir = args.retry_out or (run_dir / "retry")
    failed, workloads, optimizers = failed_cells(run_dir)
    print(f"{len(failed)} failed cells: {failed}", flush=True)
    if not failed or args.dry_run:
        return

    (retry_dir / "profiles").mkdir(parents=True, exist_ok=True)
    for profile in (run_dir / "profiles").glob("*_profile.yaml"):
        shutil.copy2(profile, retry_dir / "profiles" / profile.name)
    command = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--output",
        str(retry_dir.relative_to(ROOT)),
        "--workloads",
        *workloads,
        "--optimizers",
        *optimizers,
    ]
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=False)

    retry_status = json.loads((retry_dir / "status.json").read_text())
    retry_cells = retry_status.get("cells", {})
    for key in failed:
        if key not in retry_cells:
            continue
        workload, optimizer = key.split(":", 1)
        for kind in ("results", "logs"):
            source = retry_dir / kind / (
                f"{workload}__{optimizer}.json"
                if kind == "results"
                else f"{workload}__{optimizer}.log"
            )
            if source.is_file():
                shutil.copy2(source, run_dir / kind / source.name)
    merged = json.loads((run_dir / "status.json").read_text())
    # Drop entries for workloads or optimizers that no longer exist (for
    # example a removed workload's profile failure) so the summary reflects
    # the run that was actually requested.
    known = known_cells()
    merged["cells"] = {
        key: entry for key, entry in merged["cells"].items() if key in known
    }
    merged["cells"].update(retry_cells)
    merged["retry"] = {
        "source": str(retry_dir.relative_to(ROOT)),
        "cells": {key: retry_cells.get(key, {}).get("status") for key in failed},
    }
    (run_dir / "status.json").write_text(json.dumps(merged, indent=2))
    print(
        "merged", len(retry_cells), "retried cells into", run_dir / "status.json"
    )


if __name__ == "__main__":
    main()

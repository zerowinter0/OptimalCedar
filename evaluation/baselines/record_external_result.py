#!/usr/bin/env python3
"""Record one external baseline invocation using the shared result envelope."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_state(path: Path) -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={path}",
                "-C",
                str(path),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
        status = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={path}",
                "-C",
                str(path),
                "status",
                "--short",
            ],
            text=True,
        ).strip()
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--started-at", type=float, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--results-path", type=Path, required=True)
    args = parser.parse_args()

    finished_at = time.time()
    payload = {
        "schema_version": 1,
        "system": args.system,
        "workload": args.workload,
        "started_at_unix_sec": args.started_at,
        "finished_at_unix_sec": finished_at,
        "wall_time_sec": finished_at - args.started_at,
        "artifact": str(args.artifact.resolve()),
        "artifact_exists": args.artifact.exists(),
        "log": str(args.log.resolve()) if args.log else None,
        "optimalcedar_git": _git_state(REPO_ROOT),
        "datajuicer_git": _git_state(REPO_ROOT / "data-juicer"),
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

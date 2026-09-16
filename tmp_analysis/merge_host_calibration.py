"""Merge the machine-level cross-host calibration into one workload profile.

The values are properties of the *machine pair* (worker host <-> Ray host), not
of a workload: per-worker payload rate, shared link rate, per-worker prefetch
budget and the overflow penalty.  Without them the DP prices a cross-host stage
with no shared-capacity or in-flight bound at all, which is why a plan that
ships hundreds of KB per record can look free.

    python tmp_analysis/merge_host_calibration.py <profile.yaml> [...]
"""

import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CROSS_HOST = ROOT / "outputs/cross_host_inplan_20260913/cross_host.json"
ECHO = ROOT / "outputs/transport_microbench_20260912/ray_transport.json"


def measure_link_bytes_per_sec() -> float:
    """Aggregate cross-host rate measured by the echo sweep."""
    if not ECHO.is_file():
        return 200_000_000.0
    runs = json.loads(ECHO.read_text())
    best = 0.0
    for run in runs:
        try:
            rate = float(run["aggregate_mb_per_second"]) * 1e6
        except (KeyError, TypeError, ValueError):
            continue
        best = max(best, rate)
    return best or 200_000_000.0


def main() -> int:
    if not CROSS_HOST.is_file():
        print(f"missing calibration {CROSS_HOST}", file=sys.stderr)
        return 1
    calibration = json.loads(CROSS_HOST.read_text())
    link = measure_link_bytes_per_sec()
    for name in sys.argv[1:]:
        path = pathlib.Path(name)
        data = yaml.safe_load(path.read_text())
        transport = data.setdefault("physical_model", {}).setdefault(
            "transport", {}
        )
        transport.setdefault(
            "cross_host_bytes_per_worker_per_sec",
            round(float(calibration["worker_payload_bytes_per_sec"]), 1),
        )
        transport.setdefault("cross_host_link_bytes_per_sec", round(link, 1))
        transport.setdefault(
            "cross_host_prefetch_budget_bytes",
            float(calibration["prefetch_budget_bytes"]),
        )
        transport.setdefault(
            "cross_host_oversized_window_penalty",
            round(float(calibration["overflow_penalty"]), 4),
        )
        transport.setdefault(
            "cross_host_calibration_source",
            str(CROSS_HOST.relative_to(ROOT)),
        )
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        print(
            f"{path.name}: per_worker={transport['cross_host_bytes_per_worker_per_sec']/1e6:.1f} MB/s "
            f"link={transport['cross_host_link_bytes_per_sec']/1e6:.1f} MB/s "
            f"prefetch={transport['cross_host_prefetch_budget_bytes']/1e6:.0f} MB "
            f"penalty={transport['cross_host_oversized_window_penalty']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

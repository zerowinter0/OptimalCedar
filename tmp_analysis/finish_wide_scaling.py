"""Merge the wide-width scaling points, then re-measure and re-render SimCLRv2.

Waits for ``outputs/simclr_scaling_wide_top/simclr_profile.yaml`` (the 32- and
56-wide per-actor scaling measurements), merges those width points into the
calibrated profile, re-measures the four planner plans plus the unoptimized
plan (three repeats, no fixed worker count), re-scores every plan with every
system cost model and re-renders both figure variants.

Run:  nohup python -u tmp_analysis/finish_wide_scaling.py > tmp_analysis/wide_scaling_finish.log 2>&1 &
"""

import json
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

import tmp_analysis.finish_simclr_rerun as base  # noqa: E402

WIDE = ROOT / "outputs/simclr_scaling_wide_top/simclr_profile.yaml"
BASE_PROFILE = ROOT / "outputs/pico_default_profiles/simclr_profile.yaml"


def log(message):
    print(f"[wide {time.strftime('%H:%M:%S')}] {message}", flush=True)


def wait_for_profile(timeout_sec=2400):
    started = time.time()
    while time.time() - started < timeout_sec:
        if WIDE.is_file() and WIDE.stat().st_size > 20000:
            return True
        time.sleep(20)
    return False


def merge_scaling_points(target: Path) -> dict:
    wide = yaml.safe_load(WIDE.read_text())
    profile = yaml.safe_load(BASE_PROFILE.read_text())
    wide_scaling = wide.get("physical_model", {}).get("scaling", {})
    scaling = profile.setdefault("physical_model", {}).setdefault("scaling", {})
    merged = {}
    for variant, ops in wide_scaling.items():
        if not isinstance(ops, dict):
            continue
        for raw_pid, entry in ops.items():
            widths = entry.get("widths") if isinstance(entry, dict) else None
            if not isinstance(widths, dict):
                continue
            target_entry = scaling.setdefault(variant, {}).setdefault(
                raw_pid, {"method": entry.get("method"), "widths": {}}
            )
            target_widths = target_entry.setdefault("widths", {})
            for width, timing in widths.items():
                target_widths[width] = timing
            merged.setdefault(variant, {})[raw_pid] = sorted(
                int(w) for w in target_widths
            )
    target.write_text(yaml.safe_dump(profile, sort_keys=False))
    return merged


def main():
    out = ROOT / "outputs/simclr_rerun_wide2_20260913"
    base.OUT = out
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    base.status("waiting_for_wide_scaling")
    if not wait_for_profile():
        base.status("failed: wide scaling profile never appeared")
        return
    profile = out / "simclr_profile.yaml"
    merged = merge_scaling_points(profile)
    log(f"merged scaling widths: {json.dumps(merged)}")
    base.status("measuring", {"profile": str(profile), "scaling_widths": merged})

    measured = {}
    for optimizer, key in (
        ("dp_optimizer", "pico"),
        ("optimizer", "cedar"),
        ("plumber_optimizer", "plumber"),
        ("raydata_optimizer", "raydata"),
    ):
        results = out / "results" / f"{optimizer}.json"
        code, timed_out, seconds = base.measure_plan(profile, optimizer, results)
        value = None
        if results.is_file():
            payload = json.loads(results.read_text())
            for entry in payload.get("runs", []):
                value = entry.get("perf_time_sec")
        log(f"{key}: exit={code} timed_out={timed_out} perf={value}")
        if value:
            measured[key] = value
        base.status("measuring", {"measured": measured, "scaling_widths": merged})

    unoptimized = base.measure_unoptimized(profile)
    if unoptimized:
        measured["unoptimized"] = unoptimized
    (out / "measured.json").write_text(json.dumps(measured, indent=2))
    base.status("scoring", {"measured": measured})

    import os

    env = dict(os.environ)
    env["CEDAR_MEASURED_JSON"] = str(out / "measured.json")
    matrix = ROOT / "outputs/plumber_bench_20260912/system_cost_matrix.json"
    for command in (
        [sys.executable, "-u", "tmp_analysis/score_plans_all_models.py",
         str(profile), str(matrix)],
        [sys.executable, "tmp_analysis/make_system_estimates_figure.py",
         "--basis", "baseline", "--matrix", str(matrix),
         "--out", "simclr_model_estimates"],
        [sys.executable, "tmp_analysis/make_system_estimates_figure.py",
         "--basis", "affine", "--matrix", str(matrix),
         "--out", "simclr_model_estimates_affine"],
    ):
        code, _, seconds = base.run(command, out / "logs" / "scoring.log", env, timeout=1800)
        log(f"{command[1]}: exit={code} wall={seconds}s")
    for suffix in ("pdf", "png"):
        source = ROOT / f"outputs/plumber_bench_20260912/figures/simclr_model_estimates.{suffix}"
        if source.is_file():
            shutil.copy2(
                source,
                ROOT / f"my_paper/69e75a0100d7b4afeb1cfc20/figures/simclr_model_estimates.{suffix}",
            )
            shutil.copy2(
                source,
                ROOT / f"outputs/plumber_bench_20260912/figures/simclr_model_estimates_trace.{suffix}",
            )
    base.status("completed", {"measured": measured, "scaling_widths": merged})
    log("done")


if __name__ == "__main__":
    main()

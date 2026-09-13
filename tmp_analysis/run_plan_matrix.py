"""Diagnostic A/B of the four saved DP plans and hand-made hybrids.

Every variant runs the same workload, sample count and Ray deployment as the
original comparison; only the physical plan file changes.
"""
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

RUN = Path("/workspace/OptimalCedar/outputs/simclrv2_four_remote_20260911")
ROOT = Path("/workspace/OptimalCedar")
HERE = ROOT / "tmp_analysis"
PLANS = HERE / "plans"
LOGS = HERE / "logs"
PROFILE = RUN / "profile.yaml"
DATA = "evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py"
ENTRY = RUN / "entry.py"
ADDRESS = "172.23.166.105:6379"


def saved(name):
    plan = yaml.safe_load(
        (RUN / "plans" / name / "round_1.yaml").read_text()
    )["feature_r0"]
    # YAML round-tripping turns the integer pipe ids into strings, while the
    # logical graph and the runtime plan checks use ints.
    plan["graph"] = {int(k): v for k, v in plan["graph"].items()}
    plan["pipes"] = {int(k): v for k, v in plan["pipes"].items()}
    return plan


def write_plan(name, plan):
    PLANS.mkdir(parents=True, exist_ok=True)
    (PLANS / (name + ".yaml")).write_text(
        yaml.safe_dump({"physical_plan": plan})
    )


def set_width(plan, pid, width):
    ctx = plan["pipes"][pid]["variant_ctx"]
    if plan["pipes"][pid]["variant"] == "RAY":
        ctx["n_actors"] = width
        ctx["max_inflight"] = ctx["max_prefetch"] = 48 * width
    else:
        ctx["n_procs"] = width


def unfold_smp_block(plan, block_pid, order):
    """Replace a fused SMP block by its members running INPROCESS."""
    graph = plan["graph"]
    block_key = str(block_pid)
    before = next(k for k, v in graph.items() if v == block_key)
    after = graph[block_pid]
    del plan["pipes"][block_pid]
    for index, pid in enumerate(order):
        desc = copy.deepcopy(plan["pipes"][pid])
        desc.pop("fused_pipes", None)
        desc["variant"] = "INPROCESS"
        desc["variant_ctx"] = {"variant_type": "INPROCESS"}
        plan["pipes"][pid] = desc
        successor = after if index == len(order) - 1 else str(order[index + 1])
        graph[pid] = successor
    graph[before] = str(order[0])
    del graph[block_pid]


def variants():
    dp = saved("dp_optimizer")
    old = saved("old_dp_optimizer")
    out = {"A_dp_saved": dp, "B_old_saved": old}

    c = copy.deepcopy(dp)
    set_width(c, 6, 6)
    set_width(c, 10, 5)
    out["C_dp_order_old_widths"] = c

    d = copy.deepcopy(old)
    set_width(d, 6, 4)
    set_width(d, 10, 7)
    out["D_old_order_dp_widths"] = d

    e = copy.deepcopy(dp)
    unfold_smp_block(e, 11, [7, 1])
    out["E_dp_order_no_extra_stage"] = e

    out["F_cedar_saved"] = saved("optimizer")
    out["G_cm_saved"] = saved("cm_optimizer")
    return out


def run(name, path):
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / (name + ".log")
    env = dict(os.environ)
    env["CEDAR_RAY_PLACEMENT_RESOURCE"] = "cedar_remote"
    cmd = [
        sys.executable,
        "-u",
        str(ENTRY),
        "evaluation/eval_cedar.py",
        "--dataset_file",
        DATA,
        "--batch_size",
        "1",
        "--num_total_samples",
        "9469",
        "--profiled_stats",
        str(PROFILE),
        "--master_feature_config",
        str(path),
        "--use_ray",
        "--ray_ip",
        ADDRESS,
        "--disable_controller",
        "--disable_caching",
    ]
    with log.open("w") as handle:
        subprocess.run(
            cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env
        )
    text = log.read_text(errors="replace")
    match = re.search(r"Total time: ([0-9.]+)s", text)
    per_sample = re.search(r"time per sample: ([0-9.]+)us", text)
    if not match:
        return {"plan": name, "error": "no summary", "log": str(log)}
    return {
        "plan": name,
        "total_time_sec": float(match.group(1)),
        "time_per_sample_us": (
            float(per_sample.group(1)) if per_sample else None
        ),
    }


def main():
    all_variants = variants()
    order = sys.argv[1:] or list(all_variants)
    results = []
    for name in order:
        plan = all_variants[name]
        path = PLANS / (name + ".yaml")
        write_plan(name, plan)
        print("[matrix] running " + name, flush=True)
        result = run(name, path)
        print("[matrix] " + json.dumps(result), flush=True)
        results.append(result)
    target = HERE / ("plan_matrix_" + "-".join(sys.argv[1:3] or ["all"]) + ".json")
    target.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

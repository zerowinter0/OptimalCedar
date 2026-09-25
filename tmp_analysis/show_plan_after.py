"""Print the plan a plan-only cell produced (used for the cost-rule comparison)."""

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1])
    data = json.loads(path.read_text())
    runs = data.get("runs") or []
    print("runs:", len(runs))
    for run in runs:
        plans = run.get("physical_plans_by_feature") or {}
        if not plans:
            continue
        key = "feature" if "feature" in plans else sorted(plans)[0]
        plan = plans[key]
        print(
            "optimizer=%s plan_cost=%s setup=%.1fs W=%s"
            % (
                run.get("optimizer"),
                run.get("plan_cost"),
                float(run.get("setup_time_sec") or 0.0),
                plan.get("n_local_workers"),
            )
        )
        print(
            "  fused:",
            [
                (p, v.get("fused_pipes"))
                for p, v in plan["pipes"].items()
                if v.get("fused_pipes")
            ],
        )
        print(
            "  stages:",
            [
                (v.get("name"), v.get("variant"), v.get("variant_ctx", {}).get("n_actors"))
                for v in plan["pipes"].values()
                if v.get("variant") not in (None, "INPROCESS")
            ],
        )
        print(
            "  graph:",
            {
                k: v
                for k, v in sorted(plan["graph"].items(), key=lambda kv: int(kv[0]))
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

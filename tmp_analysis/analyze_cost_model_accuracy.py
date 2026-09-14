"""Does the cost model rank plans the way execution does?

Every comparison cell stores the plan each planner produced together with the
score our model gives that plan (``pico_plan_costs_by_feature``).  Reading the
measured throughput next to that score answers the question directly:

  * model best == measured best  -> the model ranks these plans correctly;
  * model prefers a plan that measured slower -> the model is inaccurate;
  * model ranks the measured winner first but PICO chose another plan ->
    the search/representation, not the model, is at fault.

Usage (inside the container):
  python -u tmp_analysis/analyze_cost_model_accuracy.py
"""

import json
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
RUNS = [
    ROOT / "outputs/pico_drained_20260914/results",
    ROOT / "outputs/pico_ten_workloads_20260913b/results",
    ROOT / "outputs/pico_djpecan_20260914/results",
    ROOT / "outputs/pico_missing_20260914/results",
]

# Records in each workload's data set: a drained full pass processes each one
# exactly once (verified with per-operator call counters: parse_and_format
# calls == file lines), while the harness's sample counter can be inflated by
# its per-batch accounting, so the audit derives throughput from these counts.
WORKLOAD_RECORDS = {
    "simclr": 9469,
    "blip": 1000,
    "clip": 1000,
    "dino": 1000,
    "alpaca_cot": 74771,
    "pile_hackernews": 100000,
    "pile_pubmed_abstracts": 100000,
    "pile_uspto_backgrounds": 100000,
    "bloom_oscar": 50000,
}
DRAINED_RUN = "pico_drained_20260914"

WORKLOADS = [
    "simclr",
    "blip",
    "clip",
    "dino",
    "alpaca_cot",
    "pile_hackernews",
    "pile_pubmed_abstracts",
    "pile_uspto_backgrounds",
    "bloom_oscar",
]


def spearman(pairs):
    """Rank correlation between the model score and the measured throughput."""
    n = len(pairs)
    if n < 3:
        return float("nan")

    def rank(values):
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        index = 0
        while index < n:
            end = index
            while end + 1 < n and values[order[end + 1]] == values[order[index]]:
                end += 1
            average = (index + end) / 2.0 + 1.0
            for position in range(index, end + 1):
                ranks[order[position]] = average
            index = end + 1
        return ranks

    model_rank = rank([score for score, _ in pairs])
    measured_rank = rank([-throughput for _, throughput in pairs])
    mean = (n + 1) / 2.0
    numerator = sum(
        (model_rank[i] - mean) * (measured_rank[i] - mean) for i in range(n)
    )
    denominator = sum((value - mean) ** 2 for value in model_rank)
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def collect():
    data = {name: {} for name in WORKLOADS}
    for run in reversed(RUNS):
        if not run.is_dir():
            continue
        for path in sorted(run.glob("*.json")):
            workload, _, planner = path.stem.partition("__")
            if workload not in data:
                continue
            try:
                run_entry = json.loads(path.read_text())["runs"][0]
            except Exception:  # noqa: BLE001 - malformed/unfinished cell
                continue
            perf = run_entry.get("perf_time_sec")
            samples = run_entry.get("num_samples")
            if not perf or not samples or perf <= 0:
                continue
            costs = run_entry.get("pico_plan_costs_by_feature") or {}
            if not costs:
                continue
            model_cost = sum(costs.values()) / len(costs)
            plans = run_entry.get("physical_plans_by_feature") or {}
            workers = next(iter(plans.values()), {}).get("n_local_workers")
            records = WORKLOAD_RECORDS.get(workload)
            if DRAINED_RUN in str(path) and records:
                throughput = records / perf
                samples = records
            else:
                throughput = samples / perf
            data[workload][planner] = {
                "throughput": throughput,
                "samples": int(samples),
                "model_cost": model_cost,
                "workers": workers,
            }
    return data


def main():
    data = collect()
    print(
        f"{'workload':<22}{'n':>6}  {'measured best':<22}{'model best':<22}"
        f"{'match':<6}{'rho':>6}  PICO: measured# / model#"
    )
    mismatches = []
    for workload in WORKLOADS:
        entries = data[workload]
        if not entries:
            print(f"{workload:<22}{'-':>6}  (no data)")
            continue
        samples = max(entry["samples"] for entry in entries.values())
        measured_best = max(entries, key=lambda k: entries[k]["throughput"])
        model_best = min(entries, key=lambda k: entries[k]["model_cost"])
        pairs = [
            (entry["model_cost"], entry["throughput"])
            for entry in entries.values()
        ]
        rho = spearman(pairs)
        measured_order = sorted(
            entries, key=lambda k: -entries[k]["throughput"]
        )
        model_order = sorted(entries, key=lambda k: entries[k]["model_cost"])
        pico_rank = (
            (measured_order.index("dp_optimizer") + 1, model_order.index("dp_optimizer") + 1)
            if "dp_optimizer" in entries
            else ("-", "-")
        )
        print(
            f"{workload:<22}{samples:>6}  {measured_best:<22}{model_best:<22}"
            f"{str(measured_best == model_best):<6}{rho:>6.2f}  "
            f"{pico_rank[0]}/{len(entries)} / {pico_rank[1]}/{len(entries)}"
            if isinstance(pico_rank[0], int)
            else f"{workload:<22}{samples:>6}  {measured_best:<22}{model_best:<22}"
            f"{str(measured_best == model_best):<6}{rho:>6.2f}  n/a"
        )
        if measured_best != model_best:
            mismatches.append(
                (
                    workload,
                    measured_best,
                    entries[measured_best]["throughput"],
                    entries[measured_best]["model_cost"],
                    model_best,
                    entries[model_best]["throughput"],
                    entries[model_best]["model_cost"],
                )
            )
    print("\nWhere the model disagrees with execution:")
    for (
        workload,
        measured_best,
        measured_thr,
        measured_cost,
        model_best,
        model_thr,
        model_cost,
    ) in mismatches:
        print(
            f"  {workload:<22} measured best = {measured_best} "
            f"({measured_thr:.0f} rec/s, model {measured_cost:.2f} ms) | "
            f"model best = {model_best} ({model_thr:.0f} rec/s, "
            f"model {model_cost:.2f} ms)"
        )


if __name__ == "__main__":
    main()

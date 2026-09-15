"""Emit the comparison tables for the formal matrix in Markdown.

    python tmp_analysis/paper_table.py <matrix output dir>

External baselines and PICO's own ablation (``simple_dp_optimizer``) are
reported separately: the ablation shares the joint search space with PICO and
only swaps the cost model back to Cedar's original one, so it is not a
competing system.

Throughput is source records / ``perf_time_sec``; ``perf_time_sec`` is the
harness epoch time, which excludes setup and optimizer wall time.
"""

import json
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
SOURCE_RECORDS = {
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
EXTERNAL = [
    ("optimizer", "Cedar"),
    ("dj_optimizer", "DJ-Cedar"),
    ("pecan_optimizer", "Pecan-Cedar"),
    ("plumber_optimizer", "Plumber"),
    ("raydata_optimizer", "Ray-Data"),
]
ABLATION = ("simple_dp_optimizer", "simple-DP (ablation)")
WORKLOAD_ORDER = [
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


def load(out: Path):
    cells = {}
    for path in sorted((out / "results").glob("*__*.json")):
        workload, _, optimizer = path.stem.partition("__")
        try:
            entry = json.loads(path.read_text())["runs"][0]
        except Exception:
            continue
        perf = entry.get("perf_time_sec")
        if perf is None or perf != perf or perf == float("inf"):
            perf = None
        cells.setdefault(workload, {})[optimizer] = perf
    return cells


def fmt(value, digits=2):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def rate(perf, workload):
    records = SOURCE_RECORDS.get(workload)
    if perf is None or not records:
        return None
    return records / perf


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else
               "outputs/pico_formal_20260915")
    if not out.is_absolute():
        out = ROOT / out
    cells = load(out)

    print("### 表 1：PICO vs 外部系统（吞吐，源记录/秒，越高越好）\n")
    header = "| 负载 | 源记录 | " + " | ".join(
        name for _, name in EXTERNAL
    ) + " | **PICO** | 最优外部 | PICO 加速比 |"
    print(header)
    print("|" + "---|" * (len(EXTERNAL) + 5))
    for workload in WORKLOAD_ORDER:
        per = cells.get(workload)
        if not per:
            continue
        row = [workload, str(SOURCE_RECORDS.get(workload, "?"))]
        external_values = []
        for optimizer, _name in EXTERNAL:
            perf = per.get(optimizer)
            value = rate(perf, workload)
            row.append(fmt(value, 0))
            if perf:
                external_values.append((value, optimizer))
        pico = per.get("dp_optimizer")
        pico_rate = rate(pico, workload)
        row.append(f"**{fmt(pico_rate, 0)}**" if pico_rate else "—")
        if external_values:
            best_rate, best_optimizer = max(external_values)
            best_name = dict(EXTERNAL)[best_optimizer]
            row.append(f"{best_name} {fmt(best_rate, 0)}")
            row.append(
                f"**{pico_rate / best_rate:.2f}×**" if pico_rate else "—"
            )
        else:
            row.extend(["—", "—"])
        print("| " + " | ".join(row) + " |")

    print("\n### 表 2：ablation 单列（同一联合搜索 + Cedar 原始 cost model）\n")
    print("| 负载 | " + ABLATION[1] + " | 最优外部 | ablation 加速比 | PICO 加速比 |")
    print("|---|---|---|---|---|")
    for workload in WORKLOAD_ORDER:
        per = cells.get(workload)
        if not per:
            continue
        ablation = per.get(ABLATION[0])
        pico = per.get("dp_optimizer")
        external = [
            (rate(per[optimizer], workload), optimizer)
            for optimizer, _ in EXTERNAL
            if per.get(optimizer)
        ]
        if not external:
            continue
        best_rate, best_optimizer = max(external)
        best_name = dict(EXTERNAL)[best_optimizer]
        ablation_rate = rate(ablation, workload)
        pico_rate = rate(pico, workload)
        ratio_ablation = (
            f"{ablation_rate / best_rate:.2f}×" if ablation_rate else "—"
        )
        ratio_pico = (
            f"{pico_rate / ablation_rate:.2f}×"
            if (pico_rate and ablation_rate)
            else "—"
        )
        print(
            f"| {workload} | {fmt(ablation_rate, 0)} | {best_name} {fmt(best_rate, 0)} | "
            f"{ratio_ablation} | {ratio_pico} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

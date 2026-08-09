#!/usr/bin/env python3
"""Synthesize the newest valid optimizer artifacts with DP-Cedar baseline."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter


CORE = [
    ("coco", "COCO"),
    ("commonvoice", "CV"),
    ("commonvoice_cache", "CV [Cache]"),
    ("llava_pretrain", "LLaVA"),
    ("redpajama_c4", "RP-C4"),
    ("stackexchange", "StackEx"),
    ("simclrv2", "SimCLR"),
    ("simclrv2_cache", "SimCLR [Cache]"),
    ("wikitext103", "Wiki"),
    ("wikitext103_cache", "Wiki [Cache]"),
]
DATA_PIPELINES = [
    ("redpajama_code", "RP-Code"),
    ("pile_hackernews", "HN"),
    ("pile_pubmed_abstracts", "PubMed"),
    ("pile_uspto_backgrounds", "USPTO"),
    ("pile_europarl", "EuroParl"),
]
ALL = CORE + DATA_PIPELINES

# Logical Cedar operators in each Feature, excluding the source. Cache variants
# have the same logical pipeline as their non-cache counterparts. Keep this
# explicit so the paper ordering is reproducible and reviewable.
PIPELINE_OPERATOR_COUNTS = {
    "coco": 6,
    "commonvoice": 7,
    "commonvoice_cache": 7,
    "simclrv2": 9,
    "simclrv2_cache": 9,
    "wikitext103": 9,
    "wikitext103_cache": 9,
    "llava_pretrain": 16,
    "redpajama_c4": 17,
    "redpajama_code": 17,
    "pile_hackernews": 18,
    "stackexchange": 19,
    "pile_pubmed_abstracts": 19,
    "pile_uspto_backgrounds": 19,
    "pile_europarl": 19,
}

OPTIMIZERS = [
    ("dp_cedar_optimizer", "DP-Cedar", "#9E9E9E", ""),
    ("dj_optimizer", "Data-Juicer", "#E69F00", "---"),
    ("pecan_optimizer", "Pecan", "#009E73", "xxx"),
    ("dp_optimizer", "PICO", "#0072B2", "///"),
    ("optimizer", "Cedar", "#CC79A7", "\\\\\\"),
]
BASELINE = "dp_cedar_optimizer"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def positive(value) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def load_latest(args) -> dict:
    audit = read_json(args.candidate_report)
    data = {}
    replaced_dp_workloads = (
        {"coco", "simclrv2", "simclrv2_cache"}
        if args.dp_replacement_matrix is not None
        else set()
    )

    # The enlarged three-repeat run is the newest source for these workloads.
    for workload in ("coco", "commonvoice", "commonvoice_cache"):
        data[workload] = {}
        for optimizer, _, _, _ in OPTIMIZERS:
            if optimizer == "dp_optimizer" and workload in replaced_dp_workloads:
                continue
            files = sorted(
                (args.scaled_run / "cedar" / workload / optimizer).glob("*.json")
            )
            if not files:
                raise ValueError(f"No scaled results for {workload}/{optimizer}")
            times = []
            samples = []
            for path in files:
                result = read_json(path)
                times.append(float(result["epoch_run_times"][0]))
                samples.append(int(result["epoch_num_samples"][0]))
            plan = read_json(args.scaled_run / "plans" / workload / f"{optimizer}.json")
            data[workload][optimizer] = {
                "status": "success",
                "mean": statistics.mean(times),
                "sd": statistics.stdev(times) if len(times) > 1 else 0.0,
                "repeats": len(times),
                "samples": samples[0],
                "setup": float(plan["runs"][0]["setup_time_sec"]),
                "source": str(args.scaled_run),
                "protocol": "enlarged W=8; three round-robin repeats",
            }

    # These established workloads were rerun with the same W=8 profile and
    # three round-robin repetitions in the stable paper matrix.
    for workload, _ in CORE[3:]:
        data[workload] = {}
        for optimizer, _, _, _ in OPTIMIZERS:
            if optimizer == "dp_optimizer" and workload in replaced_dp_workloads:
                continue
            workload_root = args.paper_matrix / "workloads" / workload
            unavailable = workload_root / "plans" / f"{optimizer}.unavailable.json"
            result_files = sorted(
                (workload_root / "results").glob(f"round*__{optimizer}.json")
            )
            if unavailable.exists():
                unavailable_item = read_json(unavailable)
                data[workload][optimizer] = {
                    "status": unavailable_item["status"],
                    "mean": None,
                    "sd": 0.0,
                    "repeats": 0,
                    "samples": None,
                    "setup": float(unavailable_item["timeout_sec"]),
                    "source": str(args.paper_matrix),
                    "protocol": "fixed W=8; three round-robin repeats",
                }
                continue
            if len(result_files) != 3:
                raise ValueError(
                    f"Expected three results for {workload}/{optimizer}, "
                    f"found {len(result_files)}"
                )
            times = []
            samples = []
            for path in result_files:
                result = read_json(path)
                if result.get("num_epochs") != 1:
                    raise ValueError(f"Unexpected epoch count in {path}")
                times.append(float(result["epoch_run_times"][0]))
                samples.append(int(result["epoch_num_samples"][0]))
            if len(set(samples)) != 1:
                raise ValueError(
                    f"Inconsistent sample counts for {workload}/{optimizer}: "
                    f"{samples}"
                )
            plan_result = read_json(
                workload_root / "warmup_results" / f"plan_only__{optimizer}.json"
            )
            data[workload][optimizer] = {
                "status": "success",
                "mean": statistics.mean(times),
                "sd": statistics.stdev(times),
                "repeats": len(times),
                "samples": samples[0],
                "setup": float(plan_result["runs"][0]["setup_time_sec"]),
                "source": str(args.paper_matrix),
                "protocol": "fixed W=8; three round-robin repeats",
            }

    # New Data-Juicer/Pile pipelines have three round-robin repetitions.
    for workload, _ in DATA_PIPELINES:
        candidate = audit["candidates"][workload]
        data[workload] = {}
        for optimizer, _, _, _ in OPTIMIZERS:
            item = candidate["runs"][optimizer]
            setup = item.get("optimization_time_sec")
            statuses = item.get("statuses", [])
            timed_out = any(s.get("status") == "optimizer_timeout" for s in statuses)
            if timed_out and not positive(setup):
                setup = 300.0
            data[workload][optimizer] = {
                "status": "optimizer_timeout" if timed_out else item["outcome"],
                "mean": item.get("mean_execution_time_sec"),
                "sd": item.get("stddev_execution_time_sec") or 0.0,
                "repeats": len(item.get("execution_times_sec", [])),
                "samples": (
                    item.get("processed_samples", [None])[0]
                    if item.get("processed_samples")
                    else None
                ),
                "setup": setup,
                "source": str(args.candidate_report),
                "protocol": "candidate W=8; three round-robin repeats",
            }

    # An ablation may replace only the proposed optimizer while retaining all
    # unchanged baselines from the canonical three-repeat sources.
    if args.dp_replacement_matrix is not None:
        for workload in ("coco", "simclrv2", "simclrv2_cache"):
            workload_root = args.dp_replacement_matrix / workload
            result_files = sorted(
                (workload_root / "results").glob("round*__dp_optimizer.json")
            )
            if len(result_files) != 3:
                raise ValueError(
                    f"Expected three replacement results for {workload}/"
                    f"dp_optimizer, found {len(result_files)}"
                )
            times = []
            samples = []
            for path in result_files:
                result = read_json(path)
                if result.get("num_epochs") != 1:
                    raise ValueError(f"Unexpected epoch count in {path}")
                times.append(float(result["epoch_run_times"][0]))
                samples.append(int(result["epoch_num_samples"][0]))
            if len(set(samples)) != 1:
                raise ValueError(
                    f"Inconsistent replacement sample counts for {workload}: "
                    f"{samples}"
                )
            plan_result = read_json(
                workload_root / "warmup_results" / "plan_only__dp_optimizer.json"
            )
            data[workload]["dp_optimizer"] = {
                "status": "success",
                "mean": statistics.mean(times),
                "sd": statistics.stdev(times),
                "repeats": len(times),
                "samples": samples[0],
                "setup": float(plan_result["runs"][0]["setup_time_sec"]),
                "source": str(args.dp_replacement_matrix),
                "protocol": (
                    "fixed W=8; no wall-clock correction; three repeats"
                ),
            }
    return data


def ratio(data: dict, workload: str, optimizer: str):
    baseline = data[workload][BASELINE]
    item = data[workload][optimizer]
    if not positive(baseline["mean"]) or not positive(item["mean"]):
        return None, None
    value = baseline["mean"] / item["mean"]
    if optimizer == BASELINE:
        return 1.0, 0.0
    b_cv = baseline["sd"] / baseline["mean"] if baseline["repeats"] > 1 else 0.0
    i_cv = item["sd"] / item["mean"] if item["repeats"] > 1 else 0.0
    return value, value * math.sqrt(b_cv * b_cv + i_cv * i_cv)


def style(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def legend_handles():
    handles = [
        Patch(facecolor=color, edgecolor="#333333", hatch=hatch, label=label)
        for _, label, color, hatch in OPTIMIZERS
    ]
    handles.append(
        mpl.lines.Line2D(
            [], [], marker="x", linestyle="None", color="#D55E00", label="timeout"
        )
    )
    return handles


def overhead_legend_handles():
    handles = [
        Patch(facecolor=color, edgecolor="#333333", hatch=hatch, label=label)
        for _, label, color, hatch in OPTIMIZERS
    ]
    handles.append(
        Patch(
            facecolor="#FDE0DD",
            edgecolor="#C62828",
            hatch="////",
            label="timeout (300 s cap)",
        )
    )
    return handles


def workload_tick_labels(workloads):
    return [
        f"{label}\n({PIPELINE_OPERATOR_COUNTS[workload]} ops)"
        for workload, label in workloads
    ]


def plotted_workloads(data: dict):
    """Return valid workloads ordered by increasing logical operator count."""
    valid = [item for item in ALL if positive(data[item[0]][BASELINE]["mean"])]
    return sorted(valid, key=lambda item: PIPELINE_OPERATOR_COUNTS[item[0]])


def draw_execution_panel(ax, data: dict, workloads, title: str) -> None:
    width = 0.82 / len(OPTIMIZERS)
    centers = list(range(len(workloads)))
    plotted_upper_bounds = []
    for idx, (workload, _) in enumerate(workloads):
        if not positive(data[workload][BASELINE]["mean"]):
            ax.axvspan(idx - 0.45, idx + 0.45, facecolor="#EFEFEF", hatch="//", zorder=0)
            ax.text(idx, 1.5, "INV", ha="center", va="center", color="#777777", fontweight="bold")
            continue
        for series_idx, (optimizer, _, color, hatch) in enumerate(OPTIMIZERS):
            x = idx + (series_idx - (len(OPTIMIZERS) - 1) / 2) * width
            value, error = ratio(data, workload, optimizer)
            item = data[workload][optimizer]
            if positive(value):
                plotted_upper_bounds.append(value + (error or 0.0))
                ax.bar(
                    x,
                    value,
                    yerr=error if positive(error) else None,
                    error_kw={"elinewidth": 0.55, "capsize": 1.5, "capthick": 0.55},
                    width=width * 0.92,
                    color=color,
                    edgecolor="#333333",
                    linewidth=0.45,
                    hatch=hatch,
                    zorder=3,
                )
            elif "timeout" in item["status"] or item["status"] == "unavailable":
                ax.plot(x, 0.08, marker="x", color="#D55E00", markersize=4.5, zorder=4)
                ax.text(x, 0.14, "TO", ha="center", color="#D55E00", fontsize=5.5, fontweight="bold")
    ax.axhline(1, color="#333333", linewidth=0.8)
    ax.set_xlim(-0.55, len(workloads) - 0.45)
    data_upper = max(plotted_upper_bounds, default=0.0)
    ax.set_ylim(0, max(3.05, data_upper * 1.12))
    ax.set_xticks(centers, workload_tick_labels(workloads), rotation=32)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_ylabel("Execution speedup\nvs. DP-Cedar")
    ax.set_title(title, loc="left", pad=2)
    style(ax)


def execution_figure(data: dict, output_dir: Path) -> None:
    workloads = plotted_workloads(data)
    fig, ax = plt.subplots(figsize=(7.15, 3.05))
    draw_execution_panel(
        ax,
        data,
        workloads,
        "Optimizer execution, ordered by pipeline size (higher is better)",
    )
    ax.tick_params(axis="x", labelsize=6.2, pad=1)
    ax.legend(
        handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.3),
        ncol=7, frameon=False, columnspacing=0.85, handlelength=1.45,
    )
    fig.text(
        0.995, -0.075,
        "Error bars: sample SD propagated from three round-robin repeats",
        ha="right", va="bottom", fontsize=6,
    )
    save(fig, output_dir, "latest_optimizer_execution_dp_cedar_baseline")


def draw_overhead_panel(ax, data: dict, workloads, title: str) -> None:
    width = 0.82 / len(OPTIMIZERS)
    centers = list(range(len(workloads)))
    for idx, (workload, _) in enumerate(workloads):
        for series_idx, (optimizer, _, color, hatch) in enumerate(OPTIMIZERS):
            item = data[workload][optimizer]
            setup = item["setup"]
            x = idx + (series_idx - (len(OPTIMIZERS) - 1) / 2) * width
            if positive(setup):
                timeout = (
                    "timeout" in item["status"]
                    or item["status"] == "unavailable"
                )
                ax.bar(
                    x,
                    setup,
                    width=width * 0.92,
                    color="#FDE0DD" if timeout else color,
                    edgecolor="#C62828" if timeout else "#333333",
                    linewidth=1.0 if timeout else 0.45,
                    hatch="////" if timeout else hatch,
                    zorder=3,
                )
                if timeout:
                    ax.text(x, setup * 1.08, "TO", ha="center", color="#8E0000", fontsize=5.5, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1, 520)
    ax.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _p: f"{y:g}"))
    ax.set_xlim(-0.55, len(workloads) - 0.45)
    ax.set_xticks(centers, workload_tick_labels(workloads), rotation=32)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_ylabel("Optimization/setup (s)")
    ax.set_title(title, loc="left", pad=2)
    style(ax)


def overhead_figure(data: dict, output_dir: Path) -> None:
    workloads = plotted_workloads(data)
    fig, ax = plt.subplots(figsize=(7.15, 3.05))
    draw_overhead_panel(
        ax,
        data,
        workloads,
        "Optimizer overhead, ordered by pipeline size (lower is better)",
    )
    ax.tick_params(axis="x", labelsize=6.2, pad=1)
    ax.legend(
        handles=overhead_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.3),
        ncol=6, frameon=False, columnspacing=1.0, handlelength=1.5,
    )
    fig.text(
        0.995, -0.075,
        "TO values are capped at the 300 s optimizer limit",
        ha="right", va="bottom", fontsize=6,
    )
    save(fig, output_dir, "latest_optimizer_overhead_dp_cedar_baseline")


def geomean(values):
    values = [value for value in values if positive(value)]
    return math.exp(sum(math.log(value) for value in values) / len(values))


def aggregate_figure(data: dict, output_dir: Path) -> dict:
    valid = [workload for workload, _ in ALL if positive(data[workload][BASELINE]["mean"])]
    common = [workload for workload in valid if positive(data[workload]["optimizer"]["mean"])]
    summary = {}
    for optimizer, label, color, hatch in OPTIMIZERS:
        all_values = [ratio(data, workload, optimizer)[0] for workload in valid]
        common_values = [ratio(data, workload, optimizer)[0] for workload in common]
        summary[optimizer] = {
            "label": label,
            "valid_baselines": len(valid),
            "valid_optimizer_runs": sum(positive(value) for value in all_values),
            "geomean_all_valid": geomean(all_values),
            "geomean_common": geomean(common_values),
            "common_workloads": len(common),
            "total_setup": sum(
                data[workload][optimizer]["setup"]
                for workload in valid
                if positive(data[workload][optimizer]["setup"])
            ),
        }

    labels = [item[1] for item in OPTIMIZERS]
    colors = [item[2] for item in OPTIMIZERS]
    hatches = [item[3] for item in OPTIMIZERS]
    keys = [item[0] for item in OPTIMIZERS]
    speed = [summary[key]["geomean_common"] for key in keys]
    setup = [summary[key]["total_setup"] for key in keys]
    x = list(range(len(keys)))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.3))
    bars = axes[0].bar(x, speed, color=colors, edgecolor="#333333", linewidth=0.5, width=0.7)
    for bar, hatch, value in zip(bars, hatches, speed):
        bar.set_hatch(hatch)
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}×", ha="center", fontsize=6.5)
    axes[0].axhline(1, color="#333333", linewidth=0.8)
    axes[0].set_ylim(0, max(speed) * 1.25)
    axes[0].set_ylabel("Geomean speedup\nvs. DP-Cedar")
    axes[0].set_title(f"(a) Common {len(common)} workloads with valid Cedar runs", loc="left", pad=3)
    style(axes[0])

    bars = axes[1].bar(x, setup, color=colors, edgecolor="#333333", linewidth=0.5, width=0.7)
    for bar, hatch, value in zip(bars, hatches, setup):
        bar.set_hatch(hatch)
        axes[1].text(bar.get_x() + bar.get_width() / 2, value * 1.08, f"{value:.1f}", ha="center", fontsize=6.5)
    axes[1].set_yscale("log")
    axes[1].set_ylim(20, 3000)
    axes[1].set_ylabel("Total optimization/setup (s)")
    axes[1].set_title(f"(b) {len(valid)} valid-baseline workloads; timeouts capped", loc="left", pad=3)
    style(axes[1])
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
    fig.subplots_adjust(wspace=0.32)
    save(fig, output_dir, "latest_optimizer_aggregate_dp_cedar_baseline")
    return summary


def save(fig, output_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
        if suffix == "svg":
            # Matplotlib emits trailing spaces in path data. Normalize the
            # tracked vector artifact so repository whitespace checks pass.
            lines = path.read_text().splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n")
    plt.close(fig)


def export(data: dict, summary: dict, output_dir: Path) -> None:
    fields = [
        "workload", "optimizer", "status", "mean_execution_sec", "sd_execution_sec",
        "repeats", "samples", "speedup_vs_dp_cedar", "speedup_sd", "optimization_setup_sec",
        "protocol", "source",
    ]
    with (output_dir / "latest_optimizer_data.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for workload, _ in ALL:
            for optimizer, _, _, _ in OPTIMIZERS:
                item = data[workload][optimizer]
                value, error = ratio(data, workload, optimizer)
                writer.writerow(
                    {
                        "workload": workload, "optimizer": optimizer, "status": item["status"],
                        "mean_execution_sec": item["mean"], "sd_execution_sec": item["sd"],
                        "repeats": item["repeats"], "samples": item["samples"],
                        "speedup_vs_dp_cedar": value, "speedup_sd": error,
                        "optimization_setup_sec": item["setup"], "protocol": item["protocol"],
                        "source": item["source"],
                    }
                )
    with (output_dir / "latest_optimizer_aggregate.tsv").open("w", newline="") as handle:
        fields = ["optimizer"] + list(next(iter(summary.values())).keys())
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for optimizer, values in summary.items():
            writer.writerow({"optimizer": optimizer, **values})


def write_readme(args, summary: dict, output_dir: Path) -> None:
    dp = summary["dp_optimizer"]
    valid = dp["valid_baselines"]
    replacement_source = ""
    replacement_arg = ""
    if args.dp_replacement_matrix is not None:
        replacement_source = (
            "- DP results for COCO and SimCLR(v2) variants: "
            f"`{args.dp_replacement_matrix}` (wall-clock correction removed; "
            "three repetitions)."
        )
        replacement_arg = (
            f"  --dp-replacement-matrix {args.dp_replacement_matrix} "
            + chr(92)
            + "\n"
        )
    text = f"""# Latest optimizer figures with DP-Cedar baseline

This is a source-tracked synthesis of the newest valid W=8 artifacts. Every
successful plotted cell contains three round-robin measured executions:

- COCO, CommonVoice, and CommonVoice-cache: `{args.scaled_run}`
  (enlarged inputs, three round-robin repetitions).
- LLaVA, RP-C4, StackExchange, SimCLR(v2), and WikiText-103 variants:
  `{args.paper_matrix}` (fixed W=8, three round-robin repetitions).
- RP-Code and the Pile pipelines: `{args.candidate_report}`
  (20,000 outputs by default; EuroParl uses 2,500 retained outputs; three
  round-robin repetitions).
{replacement_source}

{valid} workloads have a valid DP-Cedar execution baseline. Pile FreeLaw is
excluded because no valid formal profile was produced within the three-hour
limit. Invalid workloads are not plotted.
The per-workload x-axis is ordered by increasing logical Cedar operator count
(excluding the source); ties retain the suite order.
Cedar has valid execution plans on {summary['optimizer']['valid_optimizer_runs']}/{valid}
workloads and optimizer-timeout outcomes on the other valid-baseline pipelines.

Headline values:

- DP geomean speedup over DP-Cedar across all {valid} valid-baseline workloads:
  **{dp['geomean_all_valid']:.3f}x**.
- On the common {dp['common_workloads']} workloads where Cedar also completes,
  DP achieves **{dp['geomean_common']:.3f}x** over DP-Cedar.

Use the per-workload figures as the primary paper evidence. Error bars are
propagated sample standard deviations of the normalized ratio.

## Reproduction

Run inside the project container after `source env/bin/activate`:

```bash
python evaluation/chapter6_experiments/plot_latest_optimizer_dp_cedar_baseline.py \\
  --candidate-report {args.candidate_report} \\
  --scaled-run {args.scaled_run} \\
  --paper-matrix {args.paper_matrix} \\
{replacement_arg}  --output-dir {output_dir}
```
"""
    (output_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--scaled-run", type=Path, required=True)
    parser.add_argument("--paper-matrix", type=Path, required=True)
    parser.add_argument("--dp-replacement-matrix", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure()
    data = load_latest(args)
    execution_figure(data, args.output_dir)
    overhead_figure(data, args.output_dir)
    summary = aggregate_figure(data, args.output_dir)
    export(data, summary, args.output_dir)
    write_readme(args, summary, args.output_dir)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()

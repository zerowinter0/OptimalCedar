#!/usr/bin/env python3
"""
Benchmark reorder optimization time on independent linear operators.

The script constructs a synthetic pipeline with one source and N logical
operators. The N operators are arranged in a linear input pipeline, but they
have no semantic dependency constraints between them, so Cedar's native
enumeration-based reorder pass can consider every permutation. DP-based
optimizers use the same synthetic pipeline/profile so their reorder time can
be compared on the same input sizes and operator costs.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from cedar.compose.optimizer import Optimizer, OptimizerOptions
from cedar.pipes.noop import NoopPipe
from cedar.pipes.pipe import Pipe


OPTIMIZER_SELECTORS: Dict[str, int] = {
    "cedar": 0,
    "my_optimizer": 1,
    "dp_optimizer": 2,
    "dp_separate_optimizer": 4,
    "dp_cedar_optimizer": 5,
}

REORDER_PASS_RUNNERS: Dict[str, str] = {
    "cedar": "logical_reorder_pass",
    "my_optimizer": "physical_dp_pass",
    "dp_optimizer": "physical_dp_pass",
    "dp_separate_optimizer": "physical_dp_pass",
    "dp_cedar_optimizer": "logical_reorder_pass",
}


def optimizer_label(optimizer_name: str) -> str:
    labels = {
        "cedar": "Cedar Native",
        "my_optimizer": "MyOptimizer",
        "dp_optimizer": "DP Optimizer",
        "dp_separate_optimizer": "DP Separate Optimizer",
        "dp_cedar_optimizer": "DP Cedar Optimizer",
    }
    return labels.get(optimizer_name, optimizer_name)


def plot_title(rows: List[Dict[str, Any]]) -> str:
    optimizer_name = str(rows[0].get("optimizer", "cedar"))
    return f"{optimizer_label(optimizer_name)} Reorder Optimization Time"


class SyntheticSourcePipe(Pipe):
    def __init__(self) -> None:
        super().__init__("SyntheticSourcePipe", [], tag="source")

    def _to_inprocess(self, variant_ctx):
        raise NotImplementedError

    def _to_multiprocess(self, variant_ctx):
        raise NotImplementedError

    def _to_multithreaded(self, variant_ctx):
        raise NotImplementedError

    def _to_smp(self, variant_ctx):
        raise NotImplementedError

    def _to_ray(self, variant_ctx):
        raise NotImplementedError

    def _to_tf(self, variant_ctx):
        raise NotImplementedError

    def _to_tf_ray(self, variant_ctx):
        raise NotImplementedError

    def _to_ray_ds(self, variant_ctx):
        raise NotImplementedError


def build_pipeline(num_ops: int) -> Tuple[Dict[int, Pipe], Dict[int, Set[int]]]:
    source = SyntheticSourcePipe()
    pipes: Dict[int, Pipe] = {0: source}
    graph: Dict[int, Set[int]] = {0: set()}
    source.id = 0

    prev: Pipe = source
    for p_id in range(1, num_ops + 1):
        pipe = NoopPipe(prev, tag=f"op{p_id}")
        pipe.id = p_id
        pipes[p_id] = pipe
        graph[p_id - 1] = {p_id}
        graph[p_id] = set()
        prev = pipe

    return pipes, graph


def build_profile(num_ops: int) -> Dict[str, Any]:
    input_sizes = {0: 1024.0}
    output_sizes = {0: 1024.0}
    latencies = {0: 1000.0}

    curr_size = 1024.0
    for p_id in range(1, num_ops + 1):
        input_sizes[p_id] = curr_size
        # Use different selectivities and latencies so Cedar still performs
        # realistic cost comparisons among the generated reorder candidates.
        ratio = 0.72 + 0.035 * ((p_id - 1) % 5)
        curr_size *= ratio
        output_sizes[p_id] = curr_size
        latencies[p_id] = 600.0 + 137.0 * p_id

    return {
        "baseline": {
            "input_sizes": input_sizes,
            "latencies": latencies,
            "output_sizes": output_sizes,
            "throughput": 100.0,
        },
        # DP optimizers read disk_info even when caching is disabled.
        "disk_info": {"read_latency": 0.0},
    }


def get_optimizer_class(optimizer_name: str) -> Type[Optimizer]:
    if optimizer_name == "cedar":
        return Optimizer
    if optimizer_name == "my_optimizer":
        from cedar.compose.my_optimizer import MyOptimizer

        return MyOptimizer
    if optimizer_name == "dp_optimizer":
        from cedar.compose.dp_optimizer import DpOptimizer

        return DpOptimizer
    if optimizer_name == "dp_separate_optimizer":
        from cedar.compose.dp_seperate_optimizer import DpSeperateOptimizer

        return DpSeperateOptimizer
    if optimizer_name == "dp_cedar_optimizer":
        from cedar.compose.dp_cedar_optimizer import DpCedarOptimizer

        return DpCedarOptimizer
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def make_optimizer(
    num_ops: int, timeout_sec: Optional[float], optimizer_name: str
) -> Optimizer:
    pipes, graph = build_pipeline(num_ops)
    opt = get_optimizer_class(optimizer_name)()
    opt.init(pipes, graph)
    opt.profiled_stats = build_profile(num_ops)
    opt.options = OptimizerOptions(
        enable_prefetch=False,
        enable_offload=False,
        enable_reorder=True,
        enable_local_parallelism=False,
        enable_fusion=False,
        enable_caching=False,
        disable_physical_opt=True,
        use_my_optimizer=OPTIMIZER_SELECTORS[optimizer_name],
        reorder_timeout_sec=timeout_sec,
    )
    opt._validate_stats()
    opt._init_stats()
    return opt


def get_reorder_runner(optimizer_name: str) -> Callable[[Optimizer], None]:
    runner = REORDER_PASS_RUNNERS[optimizer_name]
    if runner == "logical_reorder_pass":
        return lambda opt: opt._pass_reordering()
    if runner == "physical_dp_pass":
        return lambda opt: opt._physical_opt()
    raise ValueError(f"Unsupported reorder runner: {runner}")


def time_reorder_once(
    num_ops: int, timeout_sec: Optional[float], optimizer_name: str
) -> Tuple[float, str]:
    opt = make_optimizer(num_ops, timeout_sec, optimizer_name)
    run_reorder_pass = get_reorder_runner(optimizer_name)
    start = time.perf_counter()
    try:
        run_reorder_pass(opt)
    except TimeoutError:
        return time.perf_counter() - start, "timeout"
    elapsed = time.perf_counter() - start
    return elapsed, "ok"


def benchmark(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for num_ops in range(args.min_ops, args.max_ops + 1):
        times: List[float] = []
        status = "ok"
        for _ in range(args.repeats):
            elapsed, run_status = time_reorder_once(
                num_ops, args.timeout_sec, args.optimizer
            )
            times.append(elapsed)
            status = run_status
            if run_status == "timeout":
                break

        rows.append(
            {
                "optimizer": args.optimizer,
                "num_independent_ops": num_ops,
                "candidate_orders": math.factorial(num_ops),
                "dp_states": 1 << num_ops,
                "repeats_completed": len(times),
                "mean_seconds": statistics.mean(times),
                "median_seconds": statistics.median(times),
                "min_seconds": min(times),
                "max_seconds": max(times),
                "status": status,
            }
        )
    return rows


def write_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_with_matplotlib(rows: List[Dict[str, Any]], figure_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    x = [row["num_independent_ops"] for row in rows]
    y = [row["mean_seconds"] for row in rows]

    plt.figure(figsize=(7, 4.2))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xlabel("Number of independent operators")
    plt.ylabel("Mean reorder time (s)")
    plt.title(plot_title(rows))
    plt.xticks(x)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def plot_svg(rows: List[Dict[str, Any]], figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 460
    left, right, top, bottom = 78, 24, 42, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [row["num_independent_ops"] for row in rows]
    ys = [row["mean_seconds"] for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(ys)
    if y_max == y_min:
        y_max = 1.0

    def sx(x: float) -> float:
        if x_max == x_min:
            return left + plot_w / 2
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))
    y_ticks = [y_max * i / 4 for i in range(5)]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">{plot_title(rows)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
    ]

    for tick in y_ticks:
        y = sy(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#ddd" stroke-dasharray="4 4"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:.3g}</text>'
        )

    for x in xs:
        px = sx(x)
        parts.append(
            f'<line x1="{px:.2f}" y1="{top + plot_h}" x2="{px:.2f}" y2="{top + plot_h + 5}" stroke="#222"/>'
        )
        parts.append(
            f'<text x="{px:.2f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{x}</text>'
        )

    parts.extend(
        [
            f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="3"/>',
            *[
                f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="#2563eb"/>'
                for x, y in zip(xs, ys)
            ],
            f'<text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="14">Number of independent operators</text>',
            f'<text x="20" y="{top + plot_h / 2}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 20 {top + plot_h / 2})">Mean reorder time (s)</text>',
            "</svg>",
        ]
    )
    figure_path.write_text("\n".join(parts))


def plot(rows: List[Dict[str, Any]], figure_path: Path) -> Path:
    try:
        plot_with_matplotlib(rows, figure_path)
        return figure_path
    except ModuleNotFoundError:
        svg_path = figure_path
        if figure_path.suffix.lower() != ".svg":
            svg_path = figure_path.with_suffix(".svg")
        plot_svg(rows, svg_path)
        return svg_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure reorder optimization time on independent operators."
        )
    )
    parser.add_argument("--min-ops", type=int, default=1)
    parser.add_argument("--max-ops", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--optimizer",
        choices=sorted(OPTIMIZER_SELECTORS.keys()),
        default="cedar",
        help=(
            "Optimizer to benchmark. cedar and dp_cedar_optimizer run "
            "_pass_reordering(); the others run the DP physical pass with "
            "offload/cache/fusion/local parallelism disabled."
        ),
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=None,
        help="Optional timeout for each native reorder run.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("evaluation/plots/cedar_reorder_time.csv"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("evaluation/plots/cedar_reorder_time.svg"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_ops < 0 or args.max_ops < args.min_ops:
        raise ValueError("Require 0 <= min_ops <= max_ops.")
    if args.repeats < 1:
        raise ValueError("Require repeats >= 1.")

    rows = benchmark(args)
    write_csv(rows, args.csv)
    figure_path = plot(rows, args.figure)

    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote figure: {figure_path}")
    for row in rows:
        print(
            "{num_independent_ops} ops: mean={mean_seconds:.6f}s, "
            "orders={candidate_orders}, status={status}".format(**row)
        )


if __name__ == "__main__":
    main()

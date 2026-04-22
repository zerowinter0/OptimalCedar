import argparse
import io
import logging
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib  # type: ignore[reportMissingImports]
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.cedar_utils import CedarEvalSpec
from evaluation.run_optimizer_cost import get_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]


LOG_COST_PATTERN = re.compile(
    r"Optimized plan cost \(calculate_cost\)\s*=\s*([0-9eE+\-\.]+)"
)
PROFILE_PATH = REPO_ROOT / "cedar" / "compose" / "simple_five_ops_profile.yml"


def _random_size(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 6)


def _random_disk_latency(rng: random.Random, lo: float, hi: float) -> float:
    """
    磁盘 read/write 延迟为每字节秒级极小量（~1e-9）。
    若用 _random_size 的 round(..., 6)，凡 < 1e-6 的值都会变成 0.0。
    """
    return round(rng.uniform(lo, hi), 12)


def generate_fake_profile(rng: random.Random) -> Dict:
    """
    生成合法 5-op profile，格式与 simple_five_ops_profile.yml 一致。
    约束：
    - input_sizes[4] = 0
    - input_sizes[x] = output_sizes[x+1], x=0..3
    """
    output_sizes = {
        0: _random_size(rng, 20.0, 300.0),
        1: _random_size(rng, 20.0, 300.0),
        2: _random_size(rng, 20.0, 300.0),
        3: _random_size(rng, 20.0, 300.0),
        4: _random_size(rng, 20.0, 300.0),
    }
    input_sizes = {
        0: output_sizes[1],
        1: output_sizes[2],
        2: output_sizes[3],
        3: output_sizes[4],
        4: 0.0,
    }
    latencies = {i: _random_size(rng, 5.0, 150.0) for i in range(5)}
    baseline_throughput = _random_size(rng, 5.0, 200.0)

    def offload_tput(mult_lo: float, mult_hi: float) -> float:
        return round(baseline_throughput * rng.uniform(mult_lo, mult_hi), 6)

    offloads = {"RAY": {}, "SMP": {}, "TF_RAY": {}}
    for p_id in range(5):
        offloads["RAY"][p_id] = {
            "input_sizes": dict(input_sizes),
            "latencies": dict(latencies),
            "output_sizes": dict(output_sizes),
            "throughput": offload_tput(0.7, 1.8),
        }
        offloads["SMP"][p_id] = {
            "input_sizes": dict(input_sizes),
            "latencies": dict(latencies),
            "output_sizes": dict(output_sizes),
            "throughput": offload_tput(0.6, 1.6),
        }

    return {
        "baseline": {
            "input_sizes": input_sizes,
            "latencies": latencies,
            "output_sizes": output_sizes,
            "throughput": baseline_throughput,
        },
        "disk_info": {
            "read_latency": _random_disk_latency(rng, 1e-10, 2e-8),
            "write_latency": _random_disk_latency(rng, 1e-10, 2e-8),
        },
        "offloads": offloads,
    }


def _extract_cost_from_logs(log_text: str, cls_name: str) -> float:
    marker = f"[{cls_name}] Optimized plan cost (calculate_cost)"
    lines = [line for line in log_text.splitlines() if marker in line]
    if not lines:
        raise RuntimeError(f"日志中未找到 {cls_name} 的 cost 行。")
    match = LOG_COST_PATTERN.search(lines[-1])
    if not match:
        raise RuntimeError(f"无法从日志行解析 cost: {lines[-1]}")
    return float(match.group(1))


def _run_like_run_optimizer_cost(use_my_optimizer: bool) -> float:
    # 与 evaluation/run_optimizer_cost.py 对齐，除了 use_my_optimizer
    spec = CedarEvalSpec(
        1,
        None,
        1,
        run_profiling=False,
        use_ray=True,
        profiled_stats=str(PROFILE_PATH),
        disable_offload=False,
        use_my_optimizer=use_my_optimizer,
        disable_prefetch=True,
        disable_fusion=False,
        disable_caching=False,
    )

    compose_logger = logging.getLogger("cedar.compose")
    old_level = compose_logger.level
    old_handlers = list(compose_logger.handlers)
    old_propagate = compose_logger.propagate

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    compose_logger.handlers = [handler]
    compose_logger.setLevel(logging.INFO)
    compose_logger.propagate = False

    try:
        _ = get_dataset(spec)
        log_text = stream.getvalue()
    finally:
        compose_logger.handlers = old_handlers
        compose_logger.setLevel(old_level)
        compose_logger.propagate = old_propagate

    cls_name = "MyOptimizer" if use_my_optimizer else "Optimizer"
    return _extract_cost_from_logs(log_text, cls_name)


def _save_cost_comparison_chart(
    case_costs: List[Tuple[int, float, float]], output_path: Path
) -> None:
    if not case_costs:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    case_ids = [case_idx for case_idx, _, _ in case_costs]
    optimizer_costs = [base_cost for _, base_cost, _ in case_costs]
    my_optimizer_costs = [my_cost for _, _, my_cost in case_costs]

    x = list(range(len(case_ids)))
    width = 0.4
    x_optimizer = [i - width / 2 for i in x]
    x_my_optimizer = [i + width / 2 for i in x]

    fig_width = max(12, len(case_ids) * 0.5)
    plt.figure(figsize=(fig_width, 6))
    plt.bar(x_optimizer, optimizer_costs, width=width, label="optimizer.py")
    plt.bar(x_my_optimizer, my_optimizer_costs, width=width, label="my_optimizer.py")

    plt.xlabel("Case Index")
    plt.ylabel("Cost (calculate_cost)")
    plt.title("Cost Comparison: optimizer.py vs my_optimizer.py")
    plt.xticks(x, [str(i) for i in case_ids], rotation=90, fontsize=8)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="复用 run_optimizer_cost.py 的 pipeline，批量对比 cost。"
    )
    parser.add_argument("--num-cases", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260331)
    parser.add_argument(
        "--keep-generated-profiles",
        action="store_true",
        help="保留每轮生成的 profile 到 evaluation/generated_profiles/",
    )
    parser.add_argument(
        "--save-failed-profiles",
        action="store_true",
        help="仅保存失败 case 的 profile 到 evaluation/failed_profiles/",
    )
    parser.add_argument(
        "--chart-output",
        type=str,
        default=str(
            REPO_ROOT / "evaluation" / "plots" / "optimizer_cost_compare.png"
        ),
        help="柱状图输出路径（PNG）。",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    profile_backup = PROFILE_PATH.read_text(encoding="utf-8")

    generated_dir = REPO_ROOT / "evaluation" / "generated_profiles"
    if args.keep_generated_profiles:
        generated_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = REPO_ROOT / "evaluation" / "failed_profiles"
    if args.save_failed_profiles:
        failed_dir.mkdir(parents=True, exist_ok=True)

    violations = []
    case_costs: List[Tuple[int, float, float]] = []
    try:
        for case_idx in range(args.num_cases):
            profile = generate_fake_profile(rng)

            # 覆盖 run_optimizer_cost.py 默认读取的 profile 文件
            with PROFILE_PATH.open("w", encoding="utf-8") as f:
                yaml.safe_dump(profile, f, sort_keys=False)

            if args.keep_generated_profiles:
                with (generated_dir / f"profile_case_{case_idx:04d}.yml").open(
                    "w", encoding="utf-8"
                ) as f:
                    yaml.safe_dump(profile, f, sort_keys=False)

            base_cost = _run_like_run_optimizer_cost(use_my_optimizer=False)
            my_cost = _run_like_run_optimizer_cost(use_my_optimizer=True)
            case_costs.append((case_idx, base_cost, my_cost))

            status = "PASS" if my_cost <= base_cost else "FAIL"
            print(
                f"[case {case_idx:03d}] {status} "
                f"optimizer={base_cost:.12f} my_optimizer={my_cost:.12f}"
            )

            if my_cost > base_cost:
                violations.append((case_idx, base_cost, my_cost))
                if args.save_failed_profiles:
                    failed_path = failed_dir / f"profile_case_{case_idx:04d}.yml"
                    with failed_path.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(profile, f, sort_keys=False)
                    print(f"  -> saved failed profile: {failed_path}")
    finally:
        # 还原原始 profile，避免污染仓库默认文件
        PROFILE_PATH.write_text(profile_backup, encoding="utf-8")

    print("\n===== SUMMARY =====")
    print(f"total_cases={args.num_cases}")
    print(f"not_higher_cases(my_cost<=optimizer_cost)={args.num_cases - len(violations)}")
    print(f"violations(my_cost>optimizer_cost)={len(violations)}")

    chart_path = Path(args.chart_output)
    _save_cost_comparison_chart(case_costs, chart_path)
    print(f"saved_cost_chart={chart_path}")

    if violations:
        print("\n以下 case 违背“my_optimizer cost 不高于 optimizer”：")
        for case_idx, base_cost, my_cost in violations[:20]:
            print(
                f"- case={case_idx} optimizer={base_cost:.12f} "
                f"my_optimizer={my_cost:.12f}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

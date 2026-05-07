import argparse
import io
import logging
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.cedar_utils import CedarEvalSpec
from evaluation.run_optimizer_cost import get_dataset


MY_COST_PATTERN = re.compile(
    r"\[MyOptimizer\] Optimized plan cost \(calculate_cost\)\s*=\s*([0-9eE+\-\.]+)"
)
DP_COST_PATTERN = re.compile(
    r"\[DpOptimizer\] Optimized plan cost\s*=\s*([0-9eE+\-\.]+)"
)
PROFILE_PATH = REPO_ROOT / "cedar" / "compose" / "simple_five_ops_profile.yml"


def _random_size(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 6)


def _random_disk_latency(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 12)


def generate_fake_profile(rng: random.Random) -> Dict:
    """
    生成合法 5-op 数值 profile，格式与 simple_five_ops_profile.yml 一致。
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


def _extract_cost_from_logs(log_text: str, optimizer_selector: int) -> float:
    pattern = MY_COST_PATTERN if optimizer_selector == 1 else DP_COST_PATTERN
    matches = pattern.findall(log_text)
    if not matches:
        optimizer_name = "MyOptimizer" if optimizer_selector == 1 else "DpOptimizer"
        raise RuntimeError(f"日志中未找到 {optimizer_name} 的 cost 行。")
    return float(matches[-1])


def _run_like_run_optimizer_cost(optimizer_selector: int) -> float:
    spec = CedarEvalSpec(
        1,
        None,
        1,
        run_profiling=False,
        use_ray=True,
        profiled_stats=str(PROFILE_PATH),
        disable_offload=False,
        use_my_optimizer=optimizer_selector,
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

    return _extract_cost_from_logs(log_text, optimizer_selector)


def _costs_match(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "在多个自动生成的数值 profile 下，检查 my_optimizer 与 "
            "dp_optimizer 的 calculate_cost 是否始终一致。"
        )
    )
    parser.add_argument("--num-cases", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--rel-tol", type=float, default=1e-9)
    parser.add_argument("--abs-tol", type=float, default=1e-9)
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
    args = parser.parse_args()

    rng = random.Random(args.seed)
    profile_backup = PROFILE_PATH.read_text(encoding="utf-8")

    generated_dir = REPO_ROOT / "evaluation" / "generated_profiles"
    if args.keep_generated_profiles:
        generated_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = REPO_ROOT / "evaluation" / "failed_profiles"
    if args.save_failed_profiles:
        failed_dir.mkdir(parents=True, exist_ok=True)

    mismatches: List[Tuple[int, float, float]] = []
    try:
        for case_idx in range(args.num_cases):
            profile = generate_fake_profile(rng)
            with PROFILE_PATH.open("w", encoding="utf-8") as f:
                yaml.safe_dump(profile, f, sort_keys=False)

            if args.keep_generated_profiles:
                generated_path = generated_dir / f"profile_case_{case_idx:04d}.yml"
                with generated_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(profile, f, sort_keys=False)

            my_cost = _run_like_run_optimizer_cost(optimizer_selector=1)
            dp_cost = _run_like_run_optimizer_cost(optimizer_selector=2)
            ok = _costs_match(my_cost, dp_cost, args.rel_tol, args.abs_tol)
            status = "PASS" if ok else "FAIL"
            print(
                f"[case {case_idx:03d}] {status} "
                f"my_optimizer={my_cost:.12f} dp_optimizer={dp_cost:.12f}"
            )

            if not ok:
                mismatches.append((case_idx, my_cost, dp_cost))
                if args.save_failed_profiles:
                    failed_path = failed_dir / f"profile_case_{case_idx:04d}.yml"
                    with failed_path.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(profile, f, sort_keys=False)
                    print(f"  -> saved failed profile: {failed_path}")
    finally:
        PROFILE_PATH.write_text(profile_backup, encoding="utf-8")

    print("\n===== SUMMARY =====")
    print(f"total_cases={args.num_cases}")
    print(f"matched_cases={args.num_cases - len(mismatches)}")
    print(f"mismatches={len(mismatches)}")

    if mismatches:
        print("\n以下 case 中两种 DP 优化器 cost 不一致：")
        for case_idx, my_cost, dp_cost in mismatches[:20]:
            print(
                f"- case={case_idx} my_optimizer={my_cost:.12f} "
                f"dp_optimizer={dp_cost:.12f}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from argparse import Namespace

from evaluation.compare_optimizer_perf import _make_spec
from cedar.compose.simple_dp_optimizer import SimpleDpOptimizer


def test_make_spec_uses_current_cedar_eval_spec_fields() -> None:
    args = Namespace(
        batch_size=4,
        num_total_samples=2000,
        data_num_total_samples=1000,
        full_data_run=False,
        num_epochs=3,
        dataset_kwargs="dataset_path=/data/input.jsonl",
        use_ray=True,
        ray_ip="ray.example:10001",
        iteration_time=0.25,
        profiled_stats="/profiles/workload.yaml",
        enable_controller=False,
        disable_offload=False,
        enable_local_parallelism=True,
        disable_caching=True,
    )

    spec = _make_spec(args, use_my_optimizer=2, reorder_timeout_sec=60.0)

    assert spec.num_total_samples == 1000
    assert spec.ray_runtime_env is None
    assert spec.iteration_time == 0.25
    assert spec.profiled_stats == "/profiles/workload.yaml"
    assert spec.disable_optimizer is False
    assert spec.disable_controller is True
    assert spec.disable_parallelism is False
    assert spec.disable_caching is True
    assert spec.use_my_optimizer == 2
    assert spec.reorder_timeout_sec == 60.0


def test_simple_dp_widths_are_normalized_by_shared_resource_policy() -> None:
    assert SimpleDpOptimizer.joint_actor_allocation is False
    assert SimpleDpOptimizer.preserve_optimizer_widths is False

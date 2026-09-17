import copy
from types import SimpleNamespace
import pytest
from cedar.compose.optimizer import OptimizerOptions
from cedar.compose.simple_dp_optimizer import SimpleDpOptimizer
from cedar.compose.simple_dp_ablation_optimizer import (
    SimpleDpWorkersOptimizer, SimpleDpBoundaryOptimizer,
    SimpleDpVariantOptimizer, SimpleDpWidthOptimizer, UnoptimizedOptimizer,
    SimpleDpWorkersBoundaryOptimizer,
    SimpleDpMaxWorkersBoundaryOptimizer,
)
from cedar.compose.plumber_optimizer import PlumberOptimizer
from cedar.compose.raydata_optimizer import RayDataOptimizer
from cedar.compose.dp_optimizer import DpObjectiveCost
from cedar.pipes import PipeVariantType
from cedar.sources import IterSource
from tests.test_dp_cache_fusion_optimizer import TwoMapFeature, _ray_profile_for


def make(cls, monkeypatch):
    monkeypatch.setenv('CEDAR_MATCH_PROFILE_RESOURCES', '1')
    monkeypatch.setenv('CEDAR_PROFILE_MATCH_CPU_BUDGET', '8')
    monkeypatch.setenv('CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET', '8')
    f = TwoMapFeature()
    f.apply(IterSource([1, 2, 3]))
    opt = cls()
    opt.init(f.logical_pipes, f.logical_adj_list)
    profile = _ray_profile_for(f)
    profile['resource_config'] = dict(schema_version=1,
        profile_scope='single_local_worker', profile_local_workers=1,
        ray_actors_per_stage=1, smp_procs_per_stage=1)
    plan = opt.run(profile, OptimizerOptions(
        enable_prefetch=False, enable_offload=True, enable_reorder=True,
        enable_local_parallelism=True, available_local_cpus=8,
        enable_fusion=True, enable_caching=False))
    return opt, plan


@pytest.mark.parametrize('cls', [SimpleDpOptimizer, SimpleDpWorkersOptimizer,
    SimpleDpBoundaryOptimizer, SimpleDpVariantOptimizer, SimpleDpWidthOptimizer,
    SimpleDpWorkersBoundaryOptimizer, PlumberOptimizer, RayDataOptimizer,
    SimpleDpMaxWorkersBoundaryOptimizer, UnoptimizedOptimizer])
def test_valid_plans(cls, monkeypatch):
    opt, plan = make(cls, monkeypatch)
    assert plan.validate()
    if cls in (PlumberOptimizer, RayDataOptimizer, UnoptimizedOptimizer):
        assert plan.n_local_workers == 1
    if cls is SimpleDpOptimizer:
        assert not opt._dp_worker_search_enabled()
        assert not hasattr(opt, '_dp_selected_workers')


def test_variant_objective_sums_within_backend_then_max():
    opt = SimpleDpVariantOptimizer()
    obj = DpObjectiveCost(local_serial=3)
    for variant, cost in [(PipeVariantType.RAY, 4),
                          (PipeVariantType.RAY, 5),
                          (PipeVariantType.SMP, 7)]:
        obj = opt._dp_accumulate_objective_cost(obj, cost,
            SimpleNamespace(variant=variant), 0)
    assert obj.score == 9
    assert obj.ray_serial == 9
    assert obj.smp_serial == 7


def test_boundary_removes_io_discount(monkeypatch):
    opt, plan = make(SimpleDpBoundaryOptimizer, monkeypatch)
    assert opt._fusion_cost_ratio((0, 1)) == 1
    assert opt._cedar_fusion_io_ratio((0, 1)) < 1


def test_widths_preserve_baseline_worker_choice(monkeypatch):
    from cedar.compose.feature import apply_profile_matched_resources
    base, base_plan = make(SimpleDpOptimizer, monkeypatch)
    apply_profile_matched_resources(base_plan, base.profiled_stats, 8)
    width, width_plan = make(SimpleDpWidthOptimizer, monkeypatch)
    assert width_plan.n_local_workers == base_plan.n_local_workers
    assert width.preserve_optimizer_widths


def test_worker_boundary_conditions_dp_on_each_resource_slice(monkeypatch):
    opt, plan = make(SimpleDpWorkersBoundaryOptimizer, monkeypatch)
    assert plan.n_local_workers == opt._dp_selected_workers
    assert 1 <= plan.n_local_workers <= 4
    assert opt.preserve_optimizer_widths
    evaluated = [row for row in opt._worker_search_evidence
                 if row['status'] == 'evaluated']
    assert evaluated
    winner = min(evaluated,
                 key=lambda row: (row['score'], row['plan_cost'],
                                  row['workers']))
    assert winner['workers'] == plan.n_local_workers
    assert winner['plan_resource_usage']['ray_cpus'] <= winner['limits']['ray_cpus']
    assert winner['plan_resource_usage']['smp_cpus'] <= winner['limits']['smp_cpus']
    for desc in plan.pipe_descs.values():
        if desc.variant_type in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
            assert desc.variant_ctx.n_actors == 1
        elif desc.variant_type == PipeVariantType.SMP:
            assert desc.variant_ctx.n_procs == 1


def test_worker_boundary_respects_workload_worker_cap(monkeypatch):
    monkeypatch.setenv('CEDAR_MATCH_PROFILE_RESOURCES', '1')
    monkeypatch.setenv('CEDAR_PROFILE_MATCH_CPU_BUDGET', '64')
    monkeypatch.setenv('CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET', '64')
    opt = SimpleDpWorkersBoundaryOptimizer()
    opt.options = SimpleNamespace(available_local_cpus=1)
    groups = opt._worker_resource_groups()
    assert [workers for workers, _ in groups] == [1]


def test_max_worker_boundary_uses_largest_feasible_w(monkeypatch):
    opt, plan = make(SimpleDpMaxWorkersBoundaryOptimizer, monkeypatch)
    assert plan.n_local_workers == 4
    selected = [row for row in opt._worker_search_evidence
                if row['status'] == 'selected_largest_feasible_workers']
    assert len(selected) == 1
    assert selected[0]['workers'] == 4

import copy
from types import SimpleNamespace
import pytest
from cedar.compose.optimizer import OptimizerOptions
from cedar.compose.simple_dp_optimizer import SimpleDpOptimizer
from cedar.compose.simple_dp_ablation_optimizer import (
    SimpleDpBoundaryOptimizer, UnoptimizedOptimizer,
    SimpleDpWorkersBoundaryOptimizer,
    SimpleDpWorkersWidthBoundaryOptimizer,
    OldDpBoundaryOptimizer,
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
    add_layered_measurements(profile)
    profile['resource_config'] = dict(schema_version=1,
        profile_scope='single_local_worker', profile_local_workers=1,
        ray_actors_per_stage=1, smp_procs_per_stage=1)
    plan = opt.run(profile, OptimizerOptions(
        enable_prefetch=False, enable_offload=True, enable_reorder=True,
        enable_local_parallelism=True, available_local_cpus=8,
        enable_fusion=True, enable_caching=False))
    return opt, plan


def add_layered_measurements(profile):
    baseline = profile['baseline']
    baseline['wall_latencies'] = copy.deepcopy(baseline['latencies'])
    operators = {}
    for p_id, input_size in baseline['input_sizes'].items():
        local_ms = float(baseline['wall_latencies'][p_id]) / 1e6
        reference = max(float(input_size), 1.0)
        operators[str(p_id)] = {
            'k_ms_per_byte': local_ms * 0.8 / reference,
            'b_ms': local_ms * 0.2,
            'fixed_fraction': 0.2,
            'x_reference_bytes': reference,
        }
    scaling = {}
    for variant, entries in profile.get('offloads', {}).items():
        if variant not in PipeVariantType.__members__:
            continue
        scaling[variant] = {}
        for p_id, entry in entries.items():
            entry['backend_compute'] = {
                'count': 32,
                'mean_ms_per_sample': 0.1 + int(p_id) * 0.01,
                'adaptive_profile': {'converged': True},
            }
            scaling[variant][str(p_id)] = {
                'widths': {
                    1: {'count': 32, 'mean_ms_per_sample': 0.1,
                        'adaptive_profile': {'converged': True}},
                    2: {'count': 32, 'mean_ms_per_sample': 0.12,
                        'adaptive_profile': {'converged': True}},
                }
            }
    profile['physical_model'] = {
        'operator_affine': {
            'schema_version': 1,
            'operators': operators,
        },
        'scaling': scaling,
    }
    return profile


@pytest.mark.parametrize('cls', [
    SimpleDpBoundaryOptimizer,
    SimpleDpWorkersBoundaryOptimizer, PlumberOptimizer, RayDataOptimizer,
    SimpleDpWorkersWidthBoundaryOptimizer, OldDpBoundaryOptimizer,
    UnoptimizedOptimizer])
def test_valid_plans(cls, monkeypatch):
    opt, plan = make(cls, monkeypatch)
    assert plan.validate()
    if cls in (PlumberOptimizer, RayDataOptimizer, UnoptimizedOptimizer):
        assert plan.n_local_workers == 1


def test_boundary_removes_io_discount(monkeypatch):
    opt, plan = make(SimpleDpBoundaryOptimizer, monkeypatch)
    assert opt._fusion_cost_ratio((0, 1)) == 1
    assert opt._cedar_fusion_io_ratio((0, 1)) < 1


def test_worker_boundary_conditions_dp_on_each_resource_slice(monkeypatch):
    opt, plan = make(SimpleDpWorkersBoundaryOptimizer, monkeypatch)
    assert plan.n_local_workers == opt._dp_selected_workers
    assert 1 <= plan.n_local_workers <= 8
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


def test_worker_width_boundary_tracks_summed_stage_widths(monkeypatch):
    opt, plan = make(SimpleDpWorkersWidthBoundaryOptimizer, monkeypatch)
    assert plan.n_local_workers == opt._dp_selected_workers
    assert opt.joint_actor_allocation
    evaluated = [row for row in opt._worker_search_evidence
                 if row['status'] == 'evaluated']
    assert evaluated
    selected = min(evaluated,
                   key=lambda row: (row['score'], row['plan_cost'],
                                    row['workers']))
    assert selected['workers'] == plan.n_local_workers
    assert selected['plan_resource_usage']['ray_cpus'] <= selected['limits']['ray_cpus']
    assert selected['plan_resource_usage']['smp_cpus'] <= selected['limits']['smp_cpus']


def test_worker_width_boundary_resource_usage_sums_widths():
    result = SimpleNamespace(
        blocks=[[0], [1], [2]],
        variants_by_idx={
            0: PipeVariantType.RAY,
            1: PipeVariantType.SMP,
            2: PipeVariantType.RAY,
        },
        parallelism_by_idx={0: 3, 1: 4, 2: 2},
    )
    usage = SimpleDpWorkersWidthBoundaryOptimizer._result_resource_usage(
        result
    )
    assert usage.ray_cpus == 5
    assert usage.smp_cpus == 4


def _initialized_cost_optimizer(cls):
    feature = TwoMapFeature()
    feature.apply(IterSource([1, 2, 3]))
    optimizer = cls()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    profile = add_layered_measurements(_ray_profile_for(feature))
    optimizer.profiled_stats = profile
    optimizer.options = OptimizerOptions(enable_offload=True)
    optimizer._init_stats()
    return optimizer, profile


def test_new_simple_dp_reads_backend_compute_not_cedar_throughput():
    from cedar.compose.optimizer import PipeDesc

    optimizer, profile = _initialized_cost_optimizer(
        SimpleDpBoundaryOptimizer
    )
    p_id = max(profile['offloads']['RAY'])
    desc = PipeDesc(name=None, variant_type=PipeVariantType.RAY)
    size = profile['baseline']['input_sizes'][p_id]
    original = optimizer._calculate_pipe_cost(p_id, size, desc)
    profile['offloads']['RAY'][p_id]['throughput'] *= 1000
    assert optimizer._calculate_pipe_cost(p_id, size, desc) == original
    profile['offloads']['RAY'][p_id][
        'backend_compute'
    ]['mean_ms_per_sample'] *= 2
    assert optimizer._calculate_pipe_cost(p_id, size, desc) == original * 2


@pytest.mark.parametrize("cls", [OldDpBoundaryOptimizer, PlumberOptimizer, RayDataOptimizer])
def test_old_dp_boundary_reads_only_cedar_offload_profile(cls):
    from cedar.compose.optimizer import PipeDesc

    optimizer, profile = _initialized_cost_optimizer(cls)
    p_id = max(profile['offloads']['RAY'])
    profile['offloads']['RAY'][p_id]['throughput'] = 120.0
    desc = PipeDesc(name=None, variant_type=PipeVariantType.RAY)
    size = profile['baseline']['input_sizes'][p_id]
    original = optimizer._calculate_pipe_cost(p_id, size, desc)
    profile['offloads']['RAY'][p_id][
        'backend_compute'
    ]['mean_ms_per_sample'] *= 1000
    assert optimizer._calculate_pipe_cost(p_id, size, desc) == original
    profile['offloads']['RAY'][p_id]['throughput'] *= 1.1
    assert optimizer._calculate_pipe_cost(p_id, size, desc) != original


def test_layered_simple_dp_has_no_boundary(monkeypatch):
    from cedar.compose.simple_dp_ablation_optimizer import LayeredSimpleDpOptimizer
    opt, plan = make(LayeredSimpleDpOptimizer, monkeypatch)
    assert plan.validate()
    block = SimpleNamespace(cost=3.0)
    assert opt._dp_regular_transition_cost(0, block) == 3.0
    assert opt._fusion_cost_ratio((0, 1)) == 1.0

def test_worker_budget_has_no_runtime_cpu_reserve(monkeypatch):
    monkeypatch.setenv("CEDAR_MATCH_PROFILE_RESOURCES", "1")
    monkeypatch.setenv("CEDAR_PROFILE_MATCH_CPU_BUDGET", "64")
    monkeypatch.setenv("CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET", "64")
    monkeypatch.setenv("CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER", "1")
    monkeypatch.setenv("CEDAR_DP_RAY_CPU_RESERVE_PER_WORKER", "1")
    opt = SimpleDpWorkersBoundaryOptimizer()
    opt.options = SimpleNamespace(available_local_cpus=64)
    assert opt._dp_worker_budget() == (64, 64, 0, 0)
    groups = opt._worker_resource_groups()
    assert groups[0][0] == 64
    assert groups[0][1].ray_cpus == 1
    assert groups[0][1].smp_cpus == 0

@pytest.mark.parametrize("cls", [
    SimpleDpWorkersBoundaryOptimizer, SimpleDpWorkersWidthBoundaryOptimizer])
def test_shared_ray_bandwidth_is_not_divided_by_workers(cls, monkeypatch):
    opt = cls()
    opt.profiled_stats = {"baseline": {"output_sizes": {7: 100.0}}}
    opt._dp_r_prod = {0: 1.0, 1: 2.0, 3: 3.0}
    opt._dp_cardinality_prod = {0: 1.0, 1: 1.0, 3: 1.0}
    monkeypatch.setattr(opt, "_get_source_p_id", lambda: 7)
    monkeypatch.setattr(opt, "_dp_boundary_throughput", lambda variant: 1000.0)
    monkeypatch.setattr(opt, "_dp_boundary_cost_ms", lambda *args: 300.0)
    block = SimpleNamespace(variant=PipeVariantType.RAY, mask=1)
    opt._dp_selected_workers = 1
    one = opt._dp_accumulate_objective_cost(DpObjectiveCost(), 500.0, block, 0)
    opt._dp_selected_workers = 8
    many = opt._dp_accumulate_objective_cost(DpObjectiveCost(), 500.0, block, 0)
    assert one.local_serial == many.local_serial == 200.0
    assert many.smp_serial == 0.0
    assert one.ray_serial == 300.0
    assert many.ray_serial / 8 == 300.0
    assert many.score / 8 == 325.0
    second = SimpleNamespace(variant=PipeVariantType.RAY, mask=2)
    combined = opt._dp_accumulate_objective_cost(many, 500.0, second, 1)
    assert combined.ray_serial / 8 == 800.0


@pytest.mark.parametrize("cls", [
    SimpleDpWorkersBoundaryOptimizer, SimpleDpWorkersWidthBoundaryOptimizer])
def test_smp_uses_measured_aggregate_capacity_without_double_charging(cls, monkeypatch):
    opt = cls()
    opt._dp_selected_workers = 8
    opt.profiled_stats = {"baseline": {"output_sizes": {7: 100.0}}}
    opt._dp_r_prod = {0: 1.0, 1: 2.0}
    monkeypatch.setattr(opt, "_get_source_p_id", lambda: 7)
    monkeypatch.setattr(opt, "_dp_boundary_cost_ms", lambda *args: 100.0)
    monkeypatch.setattr(opt, "_unscaled_boundary_transfer_ms", lambda *args: 100.0)
    monkeypatch.setattr(opt, "_dp_boundary_throughput", lambda *args: 100.0)
    monkeypatch.setattr(opt, "_dp_boundary_profile", lambda *args: {
        "aggregate_transport": {"max_inflight": 10, "points": [
            {"workers": 8, "throughput_bytes_per_sec": 400.0}]}})
    block = SimpleNamespace(variant=PipeVariantType.SMP, mask=1)
    objective = opt._dp_accumulate_objective_cost(
        DpObjectiveCost(local_serial=10.0), 500.0, block, 0)
    assert objective.local_serial == 410.0
    assert objective.smp_serial == 0.0
    assert objective.ray_serial == 200.0
    assert objective.score / 8 == 76.25


@pytest.mark.parametrize("cls", [
    SimpleDpWorkersBoundaryOptimizer, SimpleDpWorkersWidthBoundaryOptimizer])
def test_worker_boundary_searches_each_w_with_its_own_objective(cls, monkeypatch):
    opt, _ = make(cls, monkeypatch)
    evaluated = [row for row in opt._worker_search_evidence
                 if row["status"] == "evaluated"]
    assert len(evaluated) == len(opt._worker_resource_groups())
    assert all(row["source"] == "conditioned_search" for row in evaluated)
    assert all("unscaled_boundary_ms_per_sample" in row for row in evaluated)

@pytest.mark.parametrize("cls", [
    SimpleDpWorkersBoundaryOptimizer, SimpleDpWorkersWidthBoundaryOptimizer])
def test_shared_bandwidth_floor_changes_dp_placement(cls, monkeypatch):
    # A poor shared link must affect the DP's variant decision, not just the
    # score attached to an already selected Ray plan.
    monkeypatch.setattr(
        cls, "_unscaled_boundary_transfer_ms",
        lambda self, mask, block: 1e6 if block.variant in
        (PipeVariantType.RAY, PipeVariantType.TF_RAY) else 0.0)
    # Make Ray attractive on the replica coordinate. Only the independent
    # shared coordinate should rule it out.
    monkeypatch.setattr(
        cls, "_dp_regular_transition_cost",
        lambda self, mask, block: 1e6 + 1e-6 if block.variant in
        (PipeVariantType.RAY, PipeVariantType.TF_RAY) else 1.0)
    opt, plan = make(cls, monkeypatch)
    result = opt._last_dp_search_result
    assert opt._result_resource_usage(result).ray_cpus == 0
    assert result.objective.ray_serial == 0.0
    assert opt.calculate_dp_objective_cost(plan=plan) == pytest.approx(result.cost)


def test_shared_objective_ignores_unrelated_lane_exposure(monkeypatch):
    from cedar.compose import dp_optimizer
    monkeypatch.setitem(dp_optimizer._LANE_EXPOSURE, "local", 1.0)
    opt = SimpleDpWorkersBoundaryOptimizer()
    opt._base_cost_map = {7: 10.0}
    monkeypatch.setattr(opt, "_get_source_p_id", lambda: 7)
    monkeypatch.setattr(opt, "_unscaled_boundary_transfer_ms", lambda mask, block: 30.0)
    opt.profiled_stats = {"baseline": {"output_sizes": {7: 100.0}}}
    opt._dp_r_prod = {0: 1.0, 1: 2.0}
    monkeypatch.setattr(opt, "_dp_boundary_cost_ms", lambda *args: 20.0)
    opt._dp_selected_workers = 4
    result = opt._dp_accumulate_objective_cost(
        opt._dp_initial_objective_cost(), 40.0,
        SimpleNamespace(variant=PipeVariantType.RAY, mask=1), 0)
    assert result.score / 4 == 35.0

@pytest.mark.parametrize("cls", [
    SimpleDpWorkersBoundaryOptimizer, SimpleDpWorkersWidthBoundaryOptimizer])
def test_boundary_cost_adds_bytes_without_w_discount(cls, monkeypatch):
    opt = cls()
    opt.profiled_stats = {"baseline": {"output_sizes": {7: 100.0}}}
    opt._dp_r_prod = {0: 1.0, 1: 2.0}
    opt._dp_cardinality_prod = {0: 1.0, 1: 1.0}
    monkeypatch.setattr(opt, "_get_source_p_id", lambda: 7)
    monkeypatch.setattr(opt, "_dp_boundary_throughput", lambda variant: 1000.0)
    monkeypatch.setattr(opt, "_dp_boundary_cost_ms", lambda *args: 400.0)
    block = SimpleNamespace(variant=PipeVariantType.RAY, mask=1)
    opt._dp_selected_workers = 1
    one = opt._dp_accumulate_objective_cost(DpObjectiveCost(), 800.0, block, 0)
    opt._dp_selected_workers = 8
    many = opt._dp_accumulate_objective_cost(DpObjectiveCost(), 800.0, block, 0)
    # 400 compute + 100 fixed + 300 byte service.
    assert one.score == 800.0
    assert many.score / 8 == 362.5

@pytest.mark.parametrize("workers,ray_width,smp_width", [(1,64,63),(8,8,7)])
def test_width_boundary_candidates_use_the_worker_resource_slice(monkeypatch,workers,ray_width,smp_width):
    from cedar.pipes import PipeExecutionResource
    monkeypatch.setenv("CEDAR_MATCH_PROFILE_RESOURCES","1")
    monkeypatch.setenv("CEDAR_PROFILE_MATCH_CPU_BUDGET","64")
    monkeypatch.setenv("CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET","64")
    opt=SimpleDpWorkersWidthBoundaryOptimizer()
    opt._dp_selected_workers=workers
    # The measured width ladder plus the slice limit, not every integer: the
    # profile only measures 1/2/4/8, and enumerating all of them made the joint
    # W search quadratic in the CPU budget.
    ladder=(1,2,4,8)
    def expected(limit):
        return tuple(sorted({w for w in ladder if w<=limit} | {1,limit}))
    assert opt._dp_candidate_parallelisms(PipeVariantType.RAY,PipeExecutionResource.CPU)==expected(ray_width)
    assert opt._dp_candidate_parallelisms(PipeVariantType.SMP,PipeExecutionResource.CPU)==expected(smp_width)
    assert opt._dp_candidate_parallelisms(PipeVariantType.RAY,PipeExecutionResource.CUDA)==(1,)

def test_width_search_selects_and_materializes_a_wide_ray_stage(monkeypatch):
    monkeypatch.setenv('CEDAR_MATCH_PROFILE_RESOURCES','1')
    monkeypatch.setenv('CEDAR_PROFILE_MATCH_CPU_BUDGET','8')
    monkeypatch.setenv('CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET','8')
    feature=TwoMapFeature()
    feature.apply(IterSource([1,2,3]))
    profile=add_layered_measurements(_ray_profile_for(feature))
    for model in profile['physical_model']['operator_affine']['operators'].values():
        model.update(k_ms_per_byte=0.0,b_ms=100.0)
    profile['physical_model']['scaling']={}
    opt=SimpleDpWorkersWidthBoundaryOptimizer()
    opt.init(feature.logical_pipes,feature.logical_adj_list)
    plan=opt.run(profile,OptimizerOptions(
        available_local_cpus=1,enable_prefetch=False,enable_caching=False,
        enable_fusion=True,enable_offload=True,enable_reorder=True,
        enable_local_parallelism=True))
    assert plan.n_local_workers==1
    active_ray=[plan.pipe_descs[pid] for pid in plan.graph
                if plan.pipe_descs[pid].variant_type==PipeVariantType.RAY]
    assert len(active_ray)==1
    assert active_ray[0].variant_ctx.n_actors==8
    assert opt.calculate_dp_objective_cost(plan=plan)==pytest.approx(opt._last_dp_state_cost)

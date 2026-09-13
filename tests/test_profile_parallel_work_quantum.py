from types import SimpleNamespace

import pytest

from cedar.client import dataset as dataset_module
from cedar.client import DataSet
from cedar.client.dataset import _minimum_parallel_epoch_records
from cedar.pipes import PipeExecutionResource, PipeVariantType


def test_ray_epoch_rounds_per_worker_floor_to_complete_batches():
    assert (
        _minimum_parallel_epoch_records(
            PipeVariantType.RAY,
            width=48,
            ray_batch_size=10,
            minimum_records_per_worker=101,
        )
        == 48 * 110
    )


def test_batch_one_ray_epoch_gives_every_actor_sustained_work():
    assert (
        _minimum_parallel_epoch_records(
            PipeVariantType.RAY,
            width=48,
            ray_batch_size=1,
            minimum_records_per_worker=100,
        )
        == 4800
    )


def test_smp_epoch_uses_same_per_process_record_floor():
    assert (
        _minimum_parallel_epoch_records(
            PipeVariantType.SMP,
            width=48,
            ray_batch_size=1,
            minimum_records_per_worker=100,
        )
        == 4800
    )


def test_layered_ray_profile_reserves_one_gpu_across_cuda_actors():
    cuda_pipe = SimpleNamespace(execution_resource=PipeExecutionResource.CUDA)
    cpu_pipe = SimpleNamespace(execution_resource=PipeExecutionResource.CPU)

    assert dataset_module._ray_profile_gpu_fraction(cuda_pipe, 1) == 1.0
    assert dataset_module._ray_profile_gpu_fraction(cuda_pipe, 8) == 0.125
    assert dataset_module._ray_profile_gpu_fraction(cpu_pipe, 8) == 0.0


def test_adaptive_cuda_ray_benchmark_materializes_gpu_fraction():
    class FakeService:
        def reset_backend_compute_stats(self):
            return None

        def get_backend_compute_stats(self):
            return {
                "count": 2,
                "mean_ms_per_sample": 1.0,
                "stddev_ms_per_sample": 0.0,
                "stderr_ms_per_sample": 0.0,
            }

    class FakeVariant:
        def __init__(self, replay):
            self.replay = replay
            self.service = FakeService()

        def __iter__(self):
            return iter(range(self.replay.record_count))

        def shutdown(self):
            return None

    predecessor = SimpleNamespace(id=7, pipe_variant="original")
    observed = {}

    def create_variant(variant_type, variant_ctx):
        observed["variant_type"] = variant_type
        observed["variant_ctx"] = variant_ctx
        return FakeVariant(predecessor.pipe_variant)

    pipe = SimpleNamespace(
        id=8,
        input_pipes=[predecessor],
        pipe_spec=SimpleNamespace(),
        execution_resource=PipeExecutionResource.CUDA,
        _create_pipe_variant=create_variant,
    )
    dataset = object.__new__(DataSet)

    dataset._adaptive_operator_benchmark(
        pipe=pipe,
        snapshots=[b"fixed-input"],
        variant_type=PipeVariantType.RAY,
        width=8,
        min_duration=0.0,
        max_duration=1.0,
        target_rse=0.1,
        min_observations=2,
        minimum_records_per_worker=1,
    )

    assert observed["variant_type"] == PipeVariantType.RAY
    assert observed["variant_ctx"].num_gpus == 0.125
    assert predecessor.pipe_variant == "original"


@pytest.mark.parametrize("width,batch", [(0, 1), (1, 0)])
def test_parallel_epoch_rejects_nonpositive_dimensions(width, batch):
    with pytest.raises(ValueError):
        _minimum_parallel_epoch_records(
            PipeVariantType.RAY, width, batch, 100
        )

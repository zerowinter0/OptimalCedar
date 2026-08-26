import pytest

from cedar.client.dataset import _minimum_parallel_epoch_records
from cedar.pipes import PipeVariantType


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


@pytest.mark.parametrize("width,batch", [(0, 1), (1, 0)])
def test_parallel_epoch_rejects_nonpositive_dimensions(width, batch):
    with pytest.raises(ValueError):
        _minimum_parallel_epoch_records(
            PipeVariantType.RAY, width, batch, 100
        )

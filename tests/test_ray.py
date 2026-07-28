import pytest
import ray
import functools
import threading
import time
from cedar.config import CedarContext
from cedar.sources import IterSource
from cedar.pipes import (
    PipeVariantType,
    NoopPipe,
    RayPipeVariantContext,
    MapperPipe,
)


@pytest.mark.parametrize("use_threads", [False, True])
def test_basic_noop(setup_ray, use_threads):
    data = [1, 2, 3]
    source = IterSource(data)

    ctx = CedarContext()
    # ctx.init_ray()
    variant_ctx = RayPipeVariantContext(n_actors=1, use_threads=use_threads)

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    noop2 = NoopPipe(noop1)

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop1.mutate(ctx, PipeVariantType.RAY, variant_ctx)
    noop2.mutate(ctx, PipeVariantType.INPROCESS)
    out = noop2.pipe_variant

    result = []
    for x in out:
        result.append(x.data)
    assert result == data

    result = []
    for x in out:
        result.append(x.data)
    assert result == data
    variant_ctx.service.shutdown()


@pytest.mark.parametrize("use_threads", [False, True])
def test_multiple_actors(setup_ray, use_threads):
    data = range(1000)
    source = IterSource(data)

    ctx = CedarContext()
    variant_ctx = RayPipeVariantContext(n_actors=3, use_threads=use_threads)

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    noop2 = NoopPipe(noop1)

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop1.mutate(ctx, PipeVariantType.RAY, variant_ctx)
    noop2.mutate(ctx, PipeVariantType.INPROCESS)
    out = noop2.pipe_variant

    result = []
    for x in out:
        result.append(x.data)
    variant_ctx.service.shutdown()
    assert set(result) == set(data)


def _slow_identity(x):
    time.sleep(0.002)
    return x


def _add(x, y):
    return x + y


@pytest.mark.parametrize("use_threads", [False, True])
def test_map(setup_ray, use_threads):
    add_one = functools.partial(_add, y=1)

    data = range(500)
    source = IterSource(data)

    ctx = CedarContext()

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop = NoopPipe(source_pipe)
    map = MapperPipe(noop, add_one)

    noop.id = 0
    map.id = 1

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop.mutate(
        ctx,
        PipeVariantType.RAY,
        RayPipeVariantContext(4, use_threads=use_threads),
    )
    map.mutate(
        ctx,
        PipeVariantType.RAY,
        RayPipeVariantContext(4, use_threads=use_threads),
    )

    out = map.pipe_variant

    result = []
    for x in out:
        result.append(x.data)

    assert noop.pipe_variant.issued_tasks == 500
    assert noop.pipe_variant.completed_tasks == 500
    assert map.pipe_variant.issued_tasks == 500
    assert map.pipe_variant.completed_tasks == 500

    noop.pipe_variant.variant_ctx.service.shutdown()
    map.pipe_variant.variant_ctx.service.shutdown()

    assert set(result) == set((range(1, 501)))  # ordering not guaranteed


def test_batch_submit(setup_ray):
    data = range(1002)  # test fractional batch
    source = IterSource(data)

    ctx = CedarContext()
    variant_ctx = RayPipeVariantContext(
        n_actors=3, use_threads=True, submit_batch_size=8
    )

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    noop2 = NoopPipe(noop1)

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop1.mutate(ctx, PipeVariantType.RAY, variant_ctx)
    noop2.mutate(ctx, PipeVariantType.INPROCESS)
    out = noop2.pipe_variant

    result = []
    for x in out:
        result.append(x.data)
    variant_ctx.service.shutdown()
    assert set(result) == set(data)


def test_tail_batch_submit(setup_ray):
    data = range(100)  # test fractional batch
    source = IterSource(data)

    ctx = CedarContext()
    variant_ctx = RayPipeVariantContext(
        n_actors=3, use_threads=True, submit_batch_size=128
    )

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    noop2 = NoopPipe(noop1)

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop1.mutate(ctx, PipeVariantType.RAY, variant_ctx)
    noop2.mutate(ctx, PipeVariantType.INPROCESS)
    out = noop2.pipe_variant

    result = []
    for x in out:
        result.append(x.data)
    variant_ctx.service.shutdown()
    assert set(result) == set(data)


def test_profile_backend_compute_records_actor_time(setup_ray):
    data = range(33)
    source_pipe = IterSource(data).to_pipe()
    mapped = MapperPipe(source_pipe, functools.partial(_add, y=1))
    ctx = CedarContext()
    variant_ctx = RayPipeVariantContext(
        n_actors=2,
        use_threads=True,
        submit_batch_size=8,
        profile_backend_compute=True,
    )
    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    mapped.mutate(ctx, PipeVariantType.RAY, variant_ctx)

    assert [x.data for x in mapped.pipe_variant]
    stats = variant_ctx.service.get_backend_compute_stats()
    variant_ctx.service.shutdown()

    # Four full actor batches plus one tail batch.
    assert stats["count"] == 5
    assert stats["mean_ms_per_sample"] >= 0


def test_partial_batch_does_not_livelock_producer(setup_ray):
    data = range(200)
    source = IterSource(data)
    ctx = CedarContext()
    variant_ctx = RayPipeVariantContext(
        n_actors=3,
        use_threads=True,
        submit_batch_size=16,
        max_prefetch=16,
    )

    source_pipe = source.to_pipe()
    slow_map = MapperPipe(source_pipe, _slow_identity)
    ray_noop = NoopPipe(slow_map)

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    slow_map.mutate(ctx, PipeVariantType.INPROCESS)
    ray_noop.mutate(ctx, PipeVariantType.RAY, variant_ctx)

    result = []
    errors = []

    def consume():
        try:
            result.extend(x.data for x in ray_noop.pipe_variant)
        except BaseException as exc:
            errors.append(exc)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    consumer.join(timeout=15)

    if consumer.is_alive():
        variant_ctx.service.shutdown()
        consumer.join(timeout=5)
        pytest.fail("partial Ray batch livelocked")

    variant_ctx.service.shutdown()
    assert not errors
    assert set(result) == set(data)


def test_scale_sync(setup_ray):
    data = range(2000)
    source = IterSource(data)

    ctx = CedarContext()
    variant_ctx = RayPipeVariantContext(
        n_actors=1, use_threads=True, submit_batch_size=1, max_prefetch=10
    )
    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    noop2 = NoopPipe(noop1)

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop1.mutate(ctx, PipeVariantType.RAY, variant_ctx)
    noop2.mutate(ctx, PipeVariantType.INPROCESS)
    out = noop2.pipe_variant

    it = iter(out)

    result = []
    for _ in range(100):
        x = next(it)
        result.append(x.data)

    noop1.get_variant().set_scale(3)

    for _ in range(600):
        x = next(it)
        result.append(x.data)

    assert noop1.get_variant().get_scale() == 3

    service = noop1.get_variant().variant_ctx.service
    for actor in service._actors:
        # statistically, random load balance should mean it's very very
        # unlikely that an actor gets zero requests...
        assert ray.get(actor._get_n_proc.remote()) > 0

    noop1.get_variant().set_scale(2)

    assert noop1.get_variant().get_scale() == 2
    assert len(service._actors) == 2 and len(service._inflight_tasks) == 2
    assert (
        len(service._retired_actors) == 1
        and len(service._retired_inflight_tasks) == 1
    )

    for _ in range(300):
        x = next(it)
        result.append(x.data)

    noop1.get_variant().set_scale(1)
    assert noop1.get_variant().get_scale() == 1

    for _ in range(1000):
        x = next(it)
        result.append(x.data)

    assert len(service._actors) == 1 and len(service._inflight_tasks) == 1

    variant_ctx.service.shutdown()
    assert set(result) == set(data)


def test_scale_async(setup_ray):
    def _test_thread(pipe):
        time.sleep(0.5)
        pipe.get_variant().set_scale(2)
        time.sleep(0.5)
        pipe.get_variant().set_scale(1)
        time.sleep(0.5)
        pipe.get_variant().set_scale(4)
        time.sleep(0.5)
        pipe.get_variant().set_scale(2)

    data = range(2000)
    source = IterSource(data)

    ctx = CedarContext()
    variant_ctx = RayPipeVariantContext(
        n_actors=1, use_threads=True, submit_batch_size=1, max_prefetch=10
    )
    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    noop2 = NoopPipe(noop1)

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop1.mutate(ctx, PipeVariantType.RAY, variant_ctx)
    noop2.mutate(ctx, PipeVariantType.INPROCESS)
    out = noop2.pipe_variant

    thread = threading.Thread(target=_test_thread, args=(noop1,))

    result = []
    thread.start()
    for x in out:
        time.sleep(0.002)
        result.append(x.data)
    thread.join()

    service = noop1.get_variant().variant_ctx.service
    for actor in service._actors:
        assert ray.get(actor._get_n_proc.remote()) > 0
        print(ray.get(actor._get_n_proc.remote()))

    variant_ctx.service.shutdown()
    assert set(result) == set(data)

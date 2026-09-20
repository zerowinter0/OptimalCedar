import gc
import pickle

import pytest

from cedar.client import DataSet
import cedar.client.dataset as dataset_module


def _mutating_growth_operator(observed_sizes):
    def grow(value):
        observed_sizes.append(len(value["payload"]))
        value["payload"] *= 4
        return value

    return grow


def test_snapshot_affine_timer_uses_fresh_input_for_every_call():
    dataset = DataSet.__new__(DataSet)
    snapshot = pickle.dumps({"payload": bytearray(b"abcd")})
    observed_sizes = []

    elapsed_ms = dataset._time_operator_on_snapshots(
        _mutating_growth_operator(observed_sizes),
        [snapshot],
        min_calls=2,
        max_calls=2,
        repeats=2,
        target_sec=0.001,
        max_batch_bytes=1024,
    )

    assert elapsed_ms is not None
    assert len(observed_sizes) > 1
    assert set(observed_sizes) == {4}


def test_rescaled_affine_timer_uses_fresh_input_for_every_call():
    dataset = DataSet.__new__(DataSet)
    value = {"payload": bytearray(b"abcd")}
    observed_sizes = []

    elapsed_ms = dataset._time_operator_value(
        _mutating_growth_operator(observed_sizes),
        value,
        min_calls=2,
        max_calls=2,
        repeats=2,
        target_sec=0.001,
        max_batch_bytes=1024,
        payload_bytes=len(pickle.dumps(value)),
    )

    assert elapsed_ms is not None
    assert len(observed_sizes) > 1
    assert set(observed_sizes) == {4}
    assert value["payload"] == bytearray(b"abcd")


def test_affine_timer_restores_gc_after_operator_failure():
    dataset = DataSet.__new__(DataSet)
    snapshot = pickle.dumps({"payload": bytearray(b"abcd")})
    calls = 0

    def fail_during_measured_repeat(value):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("expected failure")
        return value

    gc.enable()
    with pytest.raises(RuntimeError, match="expected failure"):
        dataset._time_operator_on_snapshots(
            fail_during_measured_repeat,
            [snapshot],
            min_calls=1,
            max_calls=1,
            repeats=1,
            target_sec=0.001,
            max_batch_bytes=1024,
        )
    assert gc.isenabled()


def test_batcher_without_a_callable_is_measured_per_record():
    from cedar.pipes import BatcherPipe, NoopPipe
    from cedar.sources import IterSource

    dataset = DataSet.__new__(DataSet)
    batcher = BatcherPipe(NoopPipe(IterSource([1])), batch_size=4)
    fn, tag, records_per_call = DataSet._operator_affine_measurement(batcher)

    assert tag == "batcher"
    assert records_per_call == 4

    snapshots = [pickle.dumps({"payload": index}) for index in range(4)]
    elapsed_ms = dataset._time_operator_on_snapshots(
        fn,
        snapshots,
        min_calls=2,
        max_calls=4,
        repeats=1,
        target_sec=0.001,
        max_batch_bytes=1024,
        records_per_call=records_per_call,
    )
    assert elapsed_ms is not None and elapsed_ms > 0.0


def test_image_reader_without_a_callable_is_measured_on_real_files(tmp_path):
    import torch
    from torchvision.io import write_png

    from cedar.pipes import ImageReaderPipe
    from cedar.sources import IterSource

    dataset = DataSet.__new__(DataSet)
    path = tmp_path / "tiny.png"
    write_png(
        torch.randint(0, 255, (3, 8, 8), dtype=torch.uint8),
        str(path),
    )
    reader = ImageReaderPipe(IterSource([str(path)]))
    fn, tag, records_per_call = DataSet._operator_affine_measurement(reader)

    assert tag == "image_reader"
    assert records_per_call == 1

    elapsed_ms = dataset._time_operator_on_snapshots(
        fn,
        [pickle.dumps(str(path))],
        min_calls=2,
        max_calls=4,
        repeats=1,
        target_sec=0.001,
        max_batch_bytes=1024,
    )
    assert elapsed_ms is not None and elapsed_ms > 0.0


def test_affine_usable_snapshots_drops_records_the_operator_rejects():
    dataset = DataSet.__new__(DataSet)
    good = pickle.dumps({"payload": b"ok"})
    bad = pickle.dumps({"payload": None})

    def reject_missing_payload(value):
        if value["payload"] is None:
            raise ValueError("no payload")
        return value

    kept = dataset._affine_usable_snapshots(
        reject_missing_payload, [good, bad, good]
    )

    assert kept == [good, good]


def test_reader_measurement_ignores_directory_entries(tmp_path):
    import torch
    from torchvision.io import write_png

    path = tmp_path / "tiny.png"
    write_png(
        torch.randint(0, 255, (3, 4, 4), dtype=torch.uint8),
        str(path),
    )
    snapshots = [
        pickle.dumps(str(tmp_path)),
        pickle.dumps(str(path)),
    ]

    kept = DataSet._readable_affine_snapshots(snapshots)

    assert kept == [snapshots[1]]


def test_unsupported_pipe_reports_why_it_has_no_measurement():
    from cedar.pipes import NoopPipe
    from cedar.sources import IterSource

    fn, reason, records_per_call = DataSet._operator_affine_measurement(
        NoopPipe(IterSource([1]))
    )

    assert fn is None
    assert reason == "unsupported_pipe_type:NoopPipe"
    assert records_per_call == 1


def test_layered_snapshot_pass_is_scoped_and_discarded(monkeypatch):
    dataset = DataSet.__new__(DataSet)
    active = None
    calls = []

    def set_reservoir(value):
        nonlocal active
        active = value

    def profile_feature(*args, **kwargs):
        calls.append(active)
        return {"throughput": 123.0, "call": len(calls)}

    monkeypatch.setattr(
        dataset_module, "set_profile_input_reservoir", set_reservoir
    )
    monkeypatch.setattr(dataset, "_profile_feature", profile_feature)
    reservoir = object()

    result = dataset._collect_profile_input_reservoir(
        "feature", object(), None, reservoir
    )

    assert calls == [reservoir]
    assert result is None
    assert active is None

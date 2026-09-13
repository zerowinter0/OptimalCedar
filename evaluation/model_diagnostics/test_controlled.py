import importlib.util
from pathlib import Path


def implementation():
    path = Path(__file__).with_name('controlled.py')
    assert path.exists(), 'controlled diagnostic implementation is missing'
    spec = importlib.util.spec_from_file_location('controlled', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_partition_preserves_work_and_resources():
    m = implementation()
    for stages, widths, work in [(1, [6], [1200]), (2, [3, 3], [600, 600]),
                                (3, [2, 2, 2], [400, 400, 400])]:
        assert m.partition(stages, 1200) == (widths, work)


def test_stage_partition_rejects_lost_work():
    import pytest
    m = implementation()
    with pytest.raises(ValueError):
        m.partition(3, 100)


def test_compute_preserves_record_identity_payload_and_total_work():
    m = implementation()
    record = (7, b'x' * 512, 0)
    a = m.Work(12)(record)
    b = m.Work(6)(m.Work(6)(record))
    assert a == b == (7, b'x' * 512, 12)
    assert m.Work(0)(record) == record


def test_compute_accepts_cedar_local_tuple_unpacking():
    m = implementation()
    assert m.Work(12)(7, b'x' * 512, 0) == (7, b'x' * 512, 12)


def test_round_robin_balances_stage_positions():
    m = implementation()
    assert m.rotation([1, 2, 3], 0) == [1, 2, 3]
    assert m.rotation([1, 2, 3], 1) == [2, 3, 1]
    assert m.rotation([1, 2, 3], 2) == [3, 1, 2]


def test_ray_initialization_serializes_shared_package_writes(tmp_path, monkeypatch):
    import concurrent.futures
    import threading
    import time
    import ray
    m = implementation()
    package = tmp_path / 'shared.zip'
    def fake_init(**kwargs):
        with package.open('x'):
            time.sleep(.01)
        package.unlink()
    monkeypatch.setattr(ray, 'init', fake_init)
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        jobs = [pool.submit(m.initialize_ray, lock, address='test') for _ in range(8)]
        for job in jobs:
            job.result()

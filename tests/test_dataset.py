from typing import List
from cedar.config import CedarContext, RayConfig
from cedar.compose import Feature, OptimizerOptions
from cedar.client import DataSet
from cedar.pipes import (
    NoopPipe,
    Pipe,
    DataSample,
    MapperPipe,
    ImageReaderPipe,
    PipeVariantType,
    SMPPipeVariantContext,
    BatcherPipe,
)
from cedar.sources import IterSource, LocalFSSource, LocalLineSource
from cedar.sources.iterable import InProcessIterSourcePipeVariant
from cedar.client.profiler import FeatureProfiler
from .utils import (
    read_data_from_checkpoint_file,
    remove_checkpoint_dir,
    check_sealed_partition_dict_correctness,
    check_full_dict_correctness,
)

import functools
import os
import pathlib
import pytest
import shutil


def test_mp_dataset_epochs():
    def add(x, y):
        return x + y

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            add_two = functools.partial(add, y=2)
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = MapperPipe(ft, add_two)
            ft = MapperPipe(ft, add_two)

            return ft

    data = range(1000)

    feats = {}
    for i in range(2):
        source = IterSource(data)
        test_feature = TestFeature()
        test_feature.apply(source)

        feats[str(i)] = test_feature

    print(feats)
    ctx = CedarContext()

    dataset = DataSet(
        ctx,
        feats,
        enable_controller=False,
        iter_mode="mp",
        prefetch=False,
        enable_optimizer=False,
    )

    for _ in range(5):
        out = []
        for x in dataset:
            out.append(x)
        expected = [x + 4 for x in data] * 2
        assert sorted(expected) == sorted(out)
    # dataset._exit()


def test_mp_dataset_drains_truncated_epoch_before_next_epoch():
    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            return NoopPipe(source_pipes[0])

    data = range(200)
    feats = {}
    for i in range(2):
        feature = TestFeature()
        feature.apply(IterSource(data))
        feats[str(i)] = feature

    dataset = DataSet(
        CedarContext(),
        feats,
        enable_controller=False,
        iter_mode="mp",
        prefetch=False,
        enable_optimizer=False,
    )

    first_epoch = iter(dataset)
    for _ in range(25):
        next(first_epoch)

    # Starting a new epoch must discard the unfinished epoch, including all
    # of its sentinels, rather than letting stale sentinels end this epoch.
    out = list(dataset)
    assert sorted(out) == sorted(list(data) * 2)


def test_epoch():
    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)

            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    dataset = DataSet(ctx, {"feature": test_feature}, enable_controller=False)
    out = []

    for _ in range(5):
        for x in dataset:
            out.append(x)

    assert out == data * 5


def test_check_helper_iter_source():
    def add(x, y):
        return x + y

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            add_two = functools.partial(add, y=2)
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = MapperPipe(ft, add_two)
            ft = MapperPipe(ft, add_two)

            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    dataset = DataSet(ctx, {"feature": test_feature}, enable_controller=False)
    out = []

    for x in dataset:
        out.append(x)

    # We should have not have any remaining samples left if we have gone
    # through entire dataset
    assert dataset.check_remaining_samples()

    for x in dataset:
        break

    # We should have remaining samples if we did not go through entire dataset
    assert not dataset.check_remaining_samples()


def test_check_helper_file_list_source():
    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]) -> Pipe:
            fp = source_pipes[0]
            fp = ImageReaderPipe(fp)
            return fp

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    test_dir = pathlib.Path(test_dir) / "data/images/"
    test_feature = TestFeature()
    test_feature.apply(LocalFSSource(str(test_dir)))

    ctx = CedarContext()

    dataset = DataSet(ctx, {"feature": test_feature}, enable_controller=False)
    out = []

    for x in dataset:
        out.append(x)

    assert dataset.check_remaining_samples()

    for x in dataset:
        break

    assert not dataset.check_remaining_samples()


def test_check_helper_line_reader():
    def add_suffix(word):
        return word + " added suffix"

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = MapperPipe(ft, add_suffix)
            ft = MapperPipe(ft, add_suffix)

            return ft

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    test_file = pathlib.Path(test_dir) / "data/test_text.txt"
    source = LocalLineSource(str(test_file))

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    dataset = DataSet(ctx, {"feature": test_feature}, enable_controller=False)
    out = []
    expected = [
        "hello added suffix added suffix",
        "world added suffix added suffix",
    ]

    for x in dataset:
        out.append(x)

    assert out == expected

    assert dataset.check_remaining_samples()

    new_out = []
    for x in dataset:
        new_out.append(x)
        break

    assert new_out == [expected[0]]

    assert not dataset.check_remaining_samples()


def test_profiler(basic_noop_feature):
    basic_noop_feature.load(CedarContext())
    profiler = FeatureProfiler(basic_noop_feature)

    ds = DataSample([])

    profiler.update_ds(ds)

    assert profiler.calculate_avg_latency_per_sample() == {}

    ds.trace_dict = {
        -1: 10,
        3: 20,
        2: 25,
        1: 45,
        0: 50,
    }
    ds.trace_order = [-1, 3, 2, 1, 0]
    ds.size_dict = {
        -1: 1,
        3: 1,
        2: 1,
        1: 1,
        0: 2,
    }
    ds.do_trace = True
    ds.buffer_size_dict = {}

    profiler.update_ds(ds)

    d = profiler.calculate_avg_latency_per_sample()
    assert d == {3: 10.0, 2: 5.0, 1: 20.0, 0: 2.5}


def test_profiler_missing_traced_data_size_falls_back(basic_noop_feature):
    basic_noop_feature.load(CedarContext())
    profiler = FeatureProfiler(basic_noop_feature, profile_mode=True)

    ds = DataSample([])
    ds.trace_dict = {
        -1: 10,
        3: 20,
        2: 30,
        1: 40,
        0: 50,
    }
    ds.wall_trace_dict = {
        -1: 100,
        3: 120,
        2: 150,
        1: 190,
        0: 250,
    }
    ds.wall_trace_resume_dict = {
        -1: 100,
        3: 120,
        2: 150,
        1: 190,
        0: 250,
    }
    ds.trace_order = [-1, 3, 2, 1, 0]
    ds.size_dict = {
        -1: 1,
        3: 1,
        2: 1,
        1: 1,
        0: 1,
    }
    ds.data_size_dict = {
        3: 32,
        2: 64,
        # Pipe 1 intentionally missing, mirroring get_sizeof_data failures.
        0: 128,
    }
    ds.do_trace = True
    ds.buffer_size_dict = {}

    profiler.update_ds(ds)

    input_sizes, output_sizes = profiler.calculate_avg_data_size()
    assert input_sizes[1] == 64
    assert output_sizes[1] == 64
    assert profiler.calculate_avg_wall_latency_per_sample() == {
        3: 20.0,
        2: 30.0,
        1: 40.0,
        0: 60.0,
    }


def test_buffer_sizes(monkeypatch, noop_sleep_dataset):
    monkeypatch.setattr(
        InProcessIterSourcePipeVariant, "_should_trace", lambda self: True
    )
    _, phys_plan = noop_sleep_dataset.get_plan()["feature"]

    # Rewrite plan to have noop be multithreaded
    phys_plan["pipes"][1]["variant"] = "MULTITHREADED"
    noop_sleep_dataset.reset_feature("feature")
    noop_sleep_dataset.load_feature_from_dict("feature", phys_plan)
    noop_sleep_dataset._return_datasample = True

    for x in noop_sleep_dataset:
        assert len(x.buffer_size_dict) == 1 and 1 in x.buffer_size_dict

    buf_sizes = noop_sleep_dataset.profilers[
        "feature"
    ].calculate_avg_buffer_size()
    assert len(buf_sizes) == 1


def test_buffer_sizes_smp(monkeypatch, noop_sleep_dataset):
    monkeypatch.setattr(
        InProcessIterSourcePipeVariant, "_should_trace", lambda self: True
    )
    _, phys_plan = noop_sleep_dataset.get_plan()["feature"]

    # Rewrite plan to have noop be multithreaded
    phys_plan["pipes"][1]["variant"] = "SMP"
    noop_sleep_dataset.reset_feature("feature")
    noop_sleep_dataset.load_feature_from_dict("feature", phys_plan)
    noop_sleep_dataset._return_datasample = True

    for x in noop_sleep_dataset:
        assert len(x.buffer_size_dict) == 1 and 1 in x.buffer_size_dict

    buf_sizes = noop_sleep_dataset.profilers[
        "feature"
    ].calculate_avg_buffer_size()
    assert len(buf_sizes) == 1


def test_buffer_sizes_multiprocess(monkeypatch, noop_sleep_dataset):
    monkeypatch.setattr(
        InProcessIterSourcePipeVariant, "_should_trace", lambda self: True
    )
    _, phys_plan = noop_sleep_dataset.get_plan()["feature"]

    # Rewrite plan to have noop be multithreaded
    phys_plan["pipes"][1]["variant"] = "MULTIPROCESS"
    noop_sleep_dataset.reset_feature("feature")
    noop_sleep_dataset.load_feature_from_dict("feature", phys_plan)
    noop_sleep_dataset._return_datasample = True

    for x in noop_sleep_dataset:
        assert len(x.buffer_size_dict) == 1 and 1 in x.buffer_size_dict

    buf_sizes = noop_sleep_dataset.profilers[
        "feature"
    ].calculate_avg_buffer_size()
    assert len(buf_sizes) == 1


def test_buffer_sizes_prefetch(monkeypatch, prefetch_dataset):
    monkeypatch.setattr(
        InProcessIterSourcePipeVariant, "_should_trace", lambda self: True
    )
    prefetch_dataset._return_datasample = True

    out = []
    for x in prefetch_dataset:
        out.append(x.data)
        assert len(x.buffer_size_dict) == 2 and 1 in x.buffer_size_dict

    buf_sizes = prefetch_dataset.profilers[
        "feature"
    ].calculate_avg_buffer_size()
    assert len(buf_sizes) == 2
    assert out == list(range(5))


@pytest.mark.parametrize(
    "noop_dataset_with_variable_input", [100, 10000, 10005], indirect=True
)
def test_remaining_samples(noop_dataset_with_variable_input):
    # Check whether we fully process normal input
    out = []
    for x in noop_dataset_with_variable_input:
        out.append(x)

    assert noop_dataset_with_variable_input.check_remaining_samples()

    # Check whether we correctly seal last partition when iteration stops
    new_out = []
    counter = 0
    for y in noop_dataset_with_variable_input:
        new_out.append(y)
        if counter == 50:
            break

    assert noop_dataset_with_variable_input.check_remaining_samples()

    # Check for remaining when sent from source, but not received
    source_pipes = noop_dataset_with_variable_input._get_source_pipes()
    feature_names = noop_dataset_with_variable_input._get_feature_names()
    variant = source_pipes[feature_names[0]][0].pipe_variant
    variant.reset_for_new_epoch()
    counter = 0
    for z in variant:
        if counter == 10:
            break

    assert not noop_dataset_with_variable_input.check_remaining_samples()

    # NOTE: Potential test case: iterate, stop and then resume execution?


@pytest.mark.parametrize(
    "noop_dataset_with_variable_input", [10000], indirect=True
)
def test_checkpointing_only_sealed_basic(noop_dataset_with_variable_input):
    """
    Basic check for checkpointing only sealed partitions.
    """
    out = []
    for x in noop_dataset_with_variable_input:
        out.append(x)

    noop_dataset_with_variable_input.checkpoint(True)

    loaded_data = read_data_from_checkpoint_file()
    check_sealed_partition_dict_correctness(loaded_data, 1000, 10000)
    remove_checkpoint_dir()


@pytest.mark.parametrize(
    "noop_dataset_with_variable_input", [10999], indirect=True
)
def test_checkpointing_only_sealed_uneven(noop_dataset_with_variable_input):
    """
    Test for only checkpointing sealed partitions with data that has
    uneven last partition.
    """
    out = []
    for x in noop_dataset_with_variable_input:
        out.append(x)

    noop_dataset_with_variable_input.checkpoint(True)

    loaded_data = read_data_from_checkpoint_file()
    check_sealed_partition_dict_correctness(loaded_data, 1000, 10999)
    remove_checkpoint_dir()


@pytest.mark.parametrize(
    "noop_dataset_with_variable_input", [10000], indirect=True
)
def test_checkpointing_all(noop_dataset_with_variable_input):
    """
    Tests checkpointing all data.
    """
    counter = 0
    out = []
    for x in noop_dataset_with_variable_input:
        out.append(x)
        if counter == 1000:
            break
        counter += 1

    noop_dataset_with_variable_input.checkpoint(False)
    loaded_data = read_data_from_checkpoint_file()
    check_full_dict_correctness(loaded_data, 10005, 1000, 1, 0, 1)
    remove_checkpoint_dir()


@pytest.mark.parametrize(
    "noop_dataset_with_variable_input", [10000], indirect=True
)
def test_checkpointing_overwrite(noop_dataset_with_variable_input):
    """
    Test checkpointing with overwriting checkpoint file.
    """
    # First run
    counter = 0
    for x in noop_dataset_with_variable_input:
        if counter == 999:
            break
        counter += 1

    noop_dataset_with_variable_input.checkpoint(True)
    loaded_data = read_data_from_checkpoint_file()
    check_sealed_partition_dict_correctness(loaded_data, 1000, 1000)

    # Second (overwrite) run
    counter = 0
    out = []
    for y in noop_dataset_with_variable_input:
        out.append(y)
        counter += 1

    noop_dataset_with_variable_input.checkpoint(True)
    loaded_data = read_data_from_checkpoint_file()
    check_sealed_partition_dict_correctness(loaded_data, 1000, 10000)

    remove_checkpoint_dir()


@pytest.mark.parametrize(
    "v_type",
    [
        (PipeVariantType.INPROCESS),
        (PipeVariantType.MULTIPROCESS),
        (PipeVariantType.MULTITHREADED),
        (PipeVariantType.SMP),
        (PipeVariantType.RAY),
    ],
)
def test_dynamic_mutate_basic(v_type):
    data = [1, 2, 3, 4, 5]
    source = IterSource(data)
    ctx = CedarContext(RayConfig(n_cpus=16))
    if v_type == PipeVariantType.RAY:
        ctx.init_ray()

    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    noop2 = NoopPipe(noop1)
    source_pipe.id = 0
    noop1.id = 1
    noop2.id = 2

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop1.mutate(ctx, PipeVariantType.INPROCESS)
    noop2.mutate(ctx, PipeVariantType.INPROCESS)

    noop1_pipe_variant_old = noop1.pipe_variant

    output_it = iter(noop2.pipe_variant)

    assert next(output_it).data == 1
    assert next(output_it).data == 2

    noop1.dynamic_mutate(v_type, noop2)

    assert not (noop1_pipe_variant_old is noop1.pipe_variant)

    assert noop2.pipe_variant.input_pipe_variant == noop1.pipe_variant

    assert next(output_it).data == 3
    assert next(output_it).data == 4
    assert next(output_it).data == 5

    with pytest.raises(StopIteration):
        next(output_it)


def test_dynamic_mutate_async_drain():
    data = [1, 2, 3, 4, 5, 6, 7]
    source = IterSource(data)
    ctx = CedarContext()

    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    noop2 = NoopPipe(noop1)
    source_pipe.id = 0
    noop1.id = 1
    noop2.id = 2

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop1.mutate(ctx, PipeVariantType.SMP, SMPPipeVariantContext(1, 5))
    noop2.mutate(ctx, PipeVariantType.INPROCESS)

    output_it = iter(noop2.pipe_variant)

    assert next(output_it).data == 1
    assert next(output_it).data == 2

    noop1.dynamic_mutate(PipeVariantType.INPROCESS, noop2)

    res = []
    while True:
        try:
            ds = next(output_it)
        except StopIteration:
            break
        if not ds.dummy:
            res.append(ds.data)

    assert noop1.pipe_variant_type == PipeVariantType.INPROCESS
    assert res == [3, 4, 5, 6, 7]


def test_from_yaml_prefetch():
    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)

            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_config_file = pathlib.Path(test_dir) / "data/config_ref_prefetch.yml"

    ctx = CedarContext()

    ds = DataSet(
        ctx,
        {"feature": test_feature},
        feature_config={"feature": str(ref_config_file)},
        enable_controller=False,
    )

    out = []

    for x in ds:
        out.append(x)

    assert out == data

    print(test_feature.physical_adj_list)
    print(test_feature.logical_pipes[3].name)

    assert len(test_feature.physical_adj_list) == 5

    assert test_feature.physical_pipes[4].name == "PrefetcherPipe"


def test_from_auto_prefetch():
    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)

            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    ds = DataSet(
        ctx, {"feature": test_feature}, prefetch=True, enable_controller=False
    )

    out = []

    for x in ds:
        out.append(x)

    assert out == data

    print(test_feature.physical_adj_list)
    print(test_feature.logical_pipes[3].name)

    assert len(test_feature.physical_adj_list) == 5

    assert test_feature.physical_pipes[4].name == "PrefetcherPipe"


def test_threaded_dataset_iter():
    def add(x, y):
        return x + y

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            add_two = functools.partial(add, y=2)
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = MapperPipe(ft, add_two)
            ft = MapperPipe(ft, add_two)

            return ft

    data = [1, 2, 3, 4, 5]

    feats = {}
    for i in range(2):
        source = IterSource(data)
        test_feature = TestFeature()
        test_feature.apply(source)

        feats[str(i)] = test_feature

    print(feats)
    ctx = CedarContext()

    dataset = DataSet(
        ctx, feats, enable_controller=False, iter_mode="thread", prefetch=False
    )
    out = []

    for x in dataset:
        out.append(x)
    expected = [x + 4 for x in data] * 2
    assert sorted(expected) == sorted(out)


def test_mp_dataset_iter():
    def add(x, y):
        return x + y

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            add_two = functools.partial(add, y=2)
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = MapperPipe(ft, add_two)
            ft = MapperPipe(ft, add_two)

            return ft

    data = range(1000)

    feats = {}
    for i in range(3):
        source = IterSource(data)
        test_feature = TestFeature()
        test_feature.apply(source)

        feats[str(i)] = test_feature

    print(feats)
    ctx = CedarContext()

    dataset = DataSet(
        ctx, feats, enable_controller=False, iter_mode="mp", prefetch=False
    )
    out = []

    for x in dataset:
        out.append(x)
    expected = [x + 4 for x in data] * 3
    assert sorted(expected) == sorted(out)
    dataset._exit()


def test_static_profile(trace_every_sample):
    class TestFeature(Feature):
        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = BatcherPipe(ft, batch_size=3)

            return ft

    data = range(1, 101, 1)
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    dataset = DataSet(
        ctx,
        {"feature": test_feature},
        prefetch=False,
        enable_controller=False,
        enable_optimizer=False,  # Set to false to avoid auto running profiler
    )

    dataset.reset_feature("feature")
    d = dataset._profile("feature", n_samples=10)
    assert "disk_info" in d
    assert "read_latency" in d["disk_info"]
    assert "write_latency" in d["disk_info"]
    d = d["baseline"]
    print(d)

    assert len(d["latencies"]) == 5
    assert d["input_sizes"] == {4: 0, 3: 32, 2: 32, 1: 32, 0: 32}
    assert d["output_sizes"] == {4: 32, 3: 32, 2: 32, 1: 32, 0: 32}
    assert "throughput" in d

    # Reset the feature after profiling and make sure it returns right results
    dataset._init_features()

    res = []
    for x in dataset:
        for y in x:
            res.append(y)

    assert res == list(data)


def test_optimizer_api(trace_every_sample):
    class TestFeature(Feature):
        def __init__(self, batch_size: int):
            super().__init__()
            self.batch_size = batch_size

        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = BatcherPipe(ft, batch_size=self.batch_size)

            return ft

    data = range(1, 101, 1)
    source = IterSource(data)

    test_feature = TestFeature(3)
    test_feature.apply(source)

    ctx = CedarContext()

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_file = pathlib.Path(test_dir) / "data/test_optimizer_stats.yml"

    dataset = DataSet(
        ctx,
        {"feature": test_feature},
        prefetch=False,
        enable_controller=False,
        enable_optimizer=True,
        profiled_data=str(ref_file),
        optimizer_options=OptimizerOptions(
            enable_offload=False, available_local_cpus=1
        ),
        generate_plan=False,
    )

    res = []
    for x in dataset:
        for y in x:
            res.append(y)
    assert len(res) == 100
    assert set(res) == set(data)


def test_mp_from_config():
    class TestFeature(Feature):
        def __init__(self, batch_size: int):
            super().__init__()
            self.batch_size = batch_size

        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = BatcherPipe(ft, batch_size=self.batch_size)

            return ft

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_config_file = pathlib.Path(test_dir) / "data/config_ref_mp.yml"

    data = range(1, 100, 1)
    source = IterSource(data)

    test_feature = TestFeature(3)
    test_feature.apply(source)

    ctx = CedarContext()

    ds = DataSet(
        ctx,
        {"feature": test_feature},
        feature_config={"feature": str(ref_config_file)},
        enable_controller=False,
    )

    assert ds._iter_mode == "mp"
    assert len(ds.features) == 3

    res = []
    for x in ds:
        assert len(x) == 3
        for y in x:
            res.append(y)

    assert ds.features[ds.feature_names[0]].batch_size == 3

    assert len(res) == 99
    assert set(res) == set(data)
    ds._exit()


def _add_one(x):
    return x + 1


def test_optimizer_full_pass(setup_ray, set_ray_parallelism):
    class TestFeature(Feature):
        def __init__(self, batch_size: int):
            super().__init__()
            self.batch_size = batch_size

        def _compose(self, source_pipes):
            ft = source_pipes[0]  # 6
            ft = MapperPipe(ft, _add_one)  # 5
            ft = MapperPipe(ft, _add_one)  # 4
            ft = MapperPipe(ft, _add_one)  # 3
            ft = NoopPipe(ft)  # 2
            ft = MapperPipe(ft, _add_one)  # 1
            ft = BatcherPipe(ft, batch_size=self.batch_size).fix()  # 0

            return ft

    data = range(100)
    source = IterSource(data)

    test_feature = TestFeature(3)
    test_feature.apply(source)

    ctx = CedarContext()

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_file = pathlib.Path(test_dir) / "data/test_full_optimizer_stats.yml"

    dataset = DataSet(
        ctx,
        {"feature": test_feature},
        prefetch=False,
        enable_controller=False,
        enable_optimizer=True,
        profiled_data=str(ref_file),
        optimizer_options=OptimizerOptions(available_local_cpus=1),
        generate_plan=False,
    )

    res = []
    for x in dataset:
        for y in x:
            res.append(y)

    # Reorder 1 to the front, 5 to the back
    # 6, 1, 4, 3, 2, 5, 0, 7
    # Now, 3 is super slow to offload, so fuse 1, 4 (pid 8) and offload
    # Also offload 5
    # Check graph
    print(test_feature.physical_adj_list)
    assert test_feature.physical_adj_list == {
        6: {8},
        8: {3},
        3: {2},
        2: {5},
        5: {0},
        0: {7},
        7: set(),  # Check prefetch insertion
    }
    # Check fusion
    assert not test_feature.is_active_pipe(1)
    assert not test_feature.is_active_pipe(4)
    # Check offloading
    assert (
        test_feature.physical_pipes[8].pipe_variant_type == PipeVariantType.RAY
    )
    assert (
        test_feature.physical_pipes[5].pipe_variant_type == PipeVariantType.RAY
    )
    # Check prefetch
    assert (
        test_feature.physical_pipes[7].get_logical_name() == "PrefetcherPipe"
    )

    assert len(res) == 100
    assert set(res) == set(range(4, 104, 1))

    res = []
    for x in dataset:
        for y in x:
            res.append(y)
    assert len(res) == 100
    assert set(res) == set(range(4, 104, 1))


def test_optimizer_caching_general(setup_ray, set_ray_parallelism):
    class TestFeature(Feature):
        def __init__(self, batch_size: int):
            super().__init__()
            self.batch_size = batch_size

        def _compose(self, source_pipes):
            ft = source_pipes[0]  # 6
            ft = MapperPipe(ft, _add_one)  # 5
            ft = MapperPipe(ft, _add_one)  # 4
            ft = MapperPipe(ft, _add_one)  # 3
            ft = NoopPipe(ft)  # 2
            ft = MapperPipe(ft, _add_one)  # 1
            ft = BatcherPipe(ft, batch_size=self.batch_size).fix()  # 0

            return ft

    data = range(100)
    source = IterSource(data)

    test_feature = TestFeature(3)
    test_feature.apply(source)

    ctx = CedarContext()

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_file = pathlib.Path(test_dir) / "data/test_cache_optimizer_stats.yml"

    optimizer_options = OptimizerOptions(
        enable_prefetch=False,
        available_local_cpus=1,
        enable_offload=False,
        enable_reorder=False,
        enable_local_parallelism=False,
        enable_fusion=False,
        enable_caching=True,
        disable_physical_opt=True,
        num_samples=100,
    )

    dataset = DataSet(
        ctx,
        {"feature": test_feature},
        prefetch=False,
        enable_controller=False,
        enable_optimizer=True,
        profiled_data=str(ref_file),
        optimizer_options=optimizer_options,
        generate_plan=False,
    )

    res = []
    for x in dataset:
        for y in x:
            res.append(y)

    # Order should stay the same with cache pipe inserted
    print(test_feature.physical_adj_list)
    assert test_feature.physical_adj_list == {
        6: {5},
        5: {4},
        4: {3},
        3: {2},
        2: {1},
        1: {0},
        0: {7},
        7: set(),
    }
    for key in test_feature.physical_pipes:
        assert (
            test_feature.physical_pipes[key].pipe_variant_type
            == PipeVariantType.INPROCESS
        )

    assert len(res) == 100
    assert set(res) == set(range(4, 104, 1))

    res = []
    for x in dataset:
        for y in x:
            res.append(y)
    assert len(res) == 100
    assert set(res) == set(range(4, 104, 1))

    # Delete caching directory that was created
    pid = os.getpid()
    cache_dir = f"/tmp/cedar_{pid}"
    shutil.rmtree(cache_dir)


def test_optimizer_caching_randomness(setup_ray, set_ray_parallelism):
    class TestFeature(Feature):
        def __init__(self, batch_size: int):
            super().__init__()
            self.batch_size = batch_size

        def _compose(self, source_pipes):
            ft = source_pipes[0]  # 6
            ft = MapperPipe(ft, _add_one)  # 5
            # declare pipe as "random" for testing purposes
            ft = MapperPipe(ft, _add_one, is_random=True)  # 4
            ft = MapperPipe(ft, _add_one)  # 3
            ft = NoopPipe(ft)  # 2
            # declare pipe as "random" for testing purposes
            ft = MapperPipe(ft, _add_one, is_random=True)  # 1
            ft = BatcherPipe(ft, batch_size=self.batch_size).fix()  # 0

            return ft

    data = range(100)
    source = IterSource(data)

    test_feature = TestFeature(3)
    test_feature.apply(source)

    ctx = CedarContext()

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_file = pathlib.Path(test_dir) / "data/test_cache_optimizer_stats.yml"

    optimizer_options = OptimizerOptions(
        enable_prefetch=False,
        available_local_cpus=1,
        enable_offload=False,
        enable_reorder=False,
        enable_local_parallelism=False,
        enable_fusion=False,
        enable_caching=True,
        disable_physical_opt=True,
        num_samples=100,
    )

    dataset = DataSet(
        ctx,
        {"feature": test_feature},
        prefetch=False,
        enable_controller=False,
        enable_optimizer=True,
        profiled_data=str(ref_file),
        optimizer_options=optimizer_options,
        generate_plan=False,
    )

    res = []
    for x in dataset:
        for y in x:
            res.append(y)

    # Order should stay the same with cache pipe inserted
    print(test_feature.physical_adj_list)
    assert test_feature.physical_adj_list == {
        6: {5},
        5: {7},
        7: {4},
        4: {3},
        3: {2},
        2: {1},
        1: {0},
        0: set(),
    }
    for key in test_feature.physical_pipes:
        assert (
            test_feature.physical_pipes[key].pipe_variant_type
            == PipeVariantType.INPROCESS
        )

    assert len(res) == 100
    assert set(res) == set(range(4, 104, 1))

    res = []
    for x in dataset:
        for y in x:
            res.append(y)
    assert len(res) == 100
    assert set(res) == set(range(4, 104, 1))

    # Delete caching directory that was created
    pid = os.getpid()
    cache_dir = f"/tmp/cedar_{pid}"
    shutil.rmtree(cache_dir)


def test_optimizer_caching_with_reordering(setup_ray, set_ray_parallelism):
    class TestFeature(Feature):
        def __init__(self, batch_size: int):
            super().__init__()
            self.batch_size = batch_size

        def _compose(self, source_pipes):
            ft = source_pipes[0]  # 6
            ft = MapperPipe(ft, _add_one)  # 5
            ft = MapperPipe(ft, _add_one)  # 4
            ft = MapperPipe(ft, _add_one)  # 3
            ft = NoopPipe(ft)  # 2
            ft = MapperPipe(ft, _add_one)  # 1
            ft = BatcherPipe(ft, batch_size=self.batch_size).fix()  # 0

            return ft

    data = range(100)
    source = IterSource(data)

    test_feature = TestFeature(3)
    test_feature.apply(source)

    ctx = CedarContext()

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_file = pathlib.Path(test_dir) / "data/test_cache_optimizer_stats.yml"

    optimizer_options = OptimizerOptions(
        enable_prefetch=False,
        available_local_cpus=1,
        enable_offload=False,
        enable_reorder=True,
        enable_local_parallelism=False,
        enable_fusion=False,
        enable_caching=True,
        disable_physical_opt=True,
        num_samples=100,
    )

    dataset = DataSet(
        ctx,
        {"feature": test_feature},
        prefetch=False,
        enable_controller=False,
        enable_optimizer=True,
        profiled_data=str(ref_file),
        optimizer_options=optimizer_options,
        generate_plan=False,
    )

    res = []
    for x in dataset:
        for y in x:
            res.append(y)

    # Order should stay the same with cache pipe inserted
    print(test_feature.physical_adj_list)

    assert test_feature.physical_adj_list == {
        6: {1},
        1: {4},
        4: {3},
        3: {2},
        2: {5},
        5: {0},
        0: {7},
        7: set(),
    }

    for key in test_feature.physical_pipes:
        assert (
            test_feature.physical_pipes[key].pipe_variant_type
            == PipeVariantType.INPROCESS
        )

    assert len(res) == 100
    assert set(res) == set(range(4, 104, 1))

    res = []
    for x in dataset:
        for y in x:
            res.append(y)
    assert len(res) == 100
    assert set(res) == set(range(4, 104, 1))

    # Delete caching directory that was created
    pid = os.getpid()
    cache_dir = f"/tmp/cedar_{pid}"
    shutil.rmtree(cache_dir)


def test_optimizer_caching_expensive_io(setup_ray, set_ray_parallelism):
    class TestFeature(Feature):
        def __init__(self, batch_size: int):
            super().__init__()
            self.batch_size = batch_size

        def _compose(self, source_pipes):
            ft = source_pipes[0]  # 6
            ft = MapperPipe(ft, _add_one)  # 5
            ft = MapperPipe(ft, _add_one)  # 4
            ft = MapperPipe(ft, _add_one)  # 3
            ft = NoopPipe(ft)  # 2
            ft = MapperPipe(ft, _add_one)  # 1
            ft = BatcherPipe(ft, batch_size=self.batch_size).fix()  # 0

            return ft

    data = range(100)
    source = IterSource(data)

    test_feature = TestFeature(3)
    test_feature.apply(source)

    ctx = CedarContext()

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_file = (
        pathlib.Path(test_dir)
        / "data/test_cache_optimizer_stats_expensive_io.yml"
    )

    optimizer_options = OptimizerOptions(
        enable_prefetch=False,
        available_local_cpus=1,
        enable_offload=False,
        enable_reorder=False,
        enable_local_parallelism=False,
        enable_fusion=False,
        enable_caching=True,
        disable_physical_opt=True,
    )

    dataset = DataSet(
        ctx,
        {"feature": test_feature},
        prefetch=False,
        enable_controller=False,
        enable_optimizer=True,
        profiled_data=str(ref_file),
        optimizer_options=optimizer_options,
        generate_plan=False,
    )

    res = []
    for x in dataset:
        for y in x:
            res.append(y)

    # Order should stay the same with cache pipe inserted
    print(test_feature.physical_adj_list)

    # NOTE: no cache should be inserted, since IO is expensive
    assert test_feature.physical_adj_list == {
        6: {5},
        5: {4},
        4: {3},
        3: {2},
        2: {1},
        1: {0},
        0: set(),
    }
    for key in test_feature.physical_pipes:
        assert (
            test_feature.physical_pipes[key].pipe_variant_type
            == PipeVariantType.INPROCESS
        )

    assert len(res) == 100
    assert set(res) == set(range(4, 104, 1))

    res = []
    for x in dataset:
        for y in x:
            res.append(y)
    assert len(res) == 100
    assert set(res) == set(range(4, 104, 1))

import filecmp
import functools
import pathlib
import pytest
import threading
import time
import logging
from typing import List
from cedar.config import CedarContext
from cedar.compose import Feature
from cedar.client import DataSet
from cedar.pipes import NoopPipe, Pipe, PipeVariantType, MapperPipe, FilterPipe
from cedar.sources import IterSource
from cedar.pipes.optimize import FusedOptimizerPipe


logger = logging.getLogger(__name__)


def test_basic_optimizer_pipe():
    from cedar.pipes.optimize import NoopOptimizerPipe

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopOptimizerPipe(ft)
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

    for x in dataset:
        out.append(x)

    assert out == data

    assert len(test_feature.logical_pipes) == 5


def test_registry():
    from cedar.pipes.optimize import OptimizerPipeRegistry

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            noop = OptimizerPipeRegistry.get_pipe("NoopOptimizerPipe")

            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = noop(ft)
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

    for x in dataset:
        out.append(x)

    assert out == data

    assert len(test_feature.logical_pipes) == 5


def test_insert():
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

    dataset = DataSet(
        ctx, {"feature": test_feature}, prefetch=False, enable_controller=False
    )

    _, phys_plan = dataset.get_plan()["feature"]

    # insert a NoopOptimizer pipe
    phys_plan["pipes"][4] = {
        "name": "NoopOptimizerPipe",
        "variant": "INPROCESS",
    }
    phys_plan["graph"][4] = "1"
    phys_plan["graph"][2] = "4"

    dataset.reset_feature("feature")
    dataset.load_feature_from_dict("feature", phys_plan)

    out = []
    for x in dataset:
        out.append(x)
    assert out == data

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_config_file = pathlib.Path(test_dir) / "data/insert_config_ref.yml"
    test_config_file = pathlib.Path(test_dir) / "data/insert_config_output.yml"

    test_feature.to_yaml(str(test_config_file))
    assert filecmp.cmp(str(ref_config_file), str(test_config_file))

    # dataset.features["feature"].viz_logical_plan(
    #     str(test_dir / "data/log_plan.png")
    # )
    # dataset.features["feature"].viz_physical_plan(
    #     str(test_dir / "data/phys_plan.png")
    # )

    # Make sure that the feature's attributes are still ok
    assert len(dataset.features["feature"].logical_pipes) == 4
    assert len(dataset.features["feature"].physical_pipes) == 5
    for i in range(4):
        assert (
            dataset.features["feature"].logical_pipes[i]
            == dataset.features["feature"].physical_pipes[i]
        )


# -- CHECK FOR SINGLE PROCESS CACHE --

"""
Checks for basic caching operations using pkl files.
Concretely, checks for correct output from iteration and
correct write to cache files.
"""


@pytest.mark.parametrize(
    "setup_cache_helper_class",
    [
        ("pkl", [i for i in range(5)], True),
        ("pkl", [i for i in range(100000)], True),
    ],
    indirect=True,
)
def test_optimizer_cache_pkl_writes(setup_cache_helper_class):
    setup_cache_helper_class.check_regular()


"""
Checks for basic caching operations using pt files.
Concretely, checks for correct output from iteration and
correct write to cache files.
"""


@pytest.mark.parametrize(
    "setup_cache_helper_class",
    [
        ("pt", [i for i in range(5)], True),
        ("pt", [i for i in range(100000)], True),
    ],
    indirect=True,
)
def test_optimizer_cache_pt_writes(setup_cache_helper_class):
    setup_cache_helper_class.check_regular()


"""
Checks for cache consistency if pipes after cache change
the data sample. Cached data should be different from
returned result. Checks for pkl files.
"""


@pytest.mark.parametrize(
    "setup_cache_helper_class",
    [("pkl", [i for i in range(100000)], False)],
    indirect=True,
)
def test_optimizer_cache_later_ops_modify_input_pkl(setup_cache_helper_class):
    setup_cache_helper_class.check_non_regular(
        [i for i in range(100000)], [i + 1000 for i in range(100000)]
    )


"""
Checks for cache consistency if pipes after cache change
the data sample. Cached data should be different from
returned result. Checks for pt files.
"""


@pytest.mark.parametrize(
    "setup_cache_helper_class",
    [("pt", [i for i in range(100000)], False)],
    indirect=True,
)
def test_optimizer_cache_later_ops_modify_input_pt(setup_cache_helper_class):
    setup_cache_helper_class.check_non_regular(
        [i for i in range(100000)], [i + 1000 for i in range(100000)]
    )


"""
Checks for correct reads by the cache pipe in later iterations
when data is saved to .pkl files.
"""


@pytest.mark.parametrize(
    "setup_cache_helper_class",
    [
        ("pkl", [i for i in range(5)], True),
        ("pkl", [i for i in range(100000)], True),
    ],
    indirect=True,
)
def test_optimizer_cache_pkl_reads(setup_cache_helper_class):
    setup_cache_helper_class.cache()
    setup_cache_helper_class.setup(True)
    setup_cache_helper_class.check_read()


"""
Checks for correct reads by the cache pipe in later iterations
when data is saved to .pt files.
"""


@pytest.mark.parametrize(
    "setup_cache_helper_class",
    [
        ("pt", [i for i in range(5)], True),
        ("pt", [i for i in range(100000)], True),
    ],
    indirect=True,
)
def test_optimizer_cache_pt_reads(setup_cache_helper_class):
    setup_cache_helper_class.cache()
    setup_cache_helper_class.setup(True)
    setup_cache_helper_class.check_read()


@pytest.mark.skip()  # TODO: FIXME
@pytest.mark.parametrize(
    "setup_cache_helper_class",
    [
        ("pkl", [i for i in range(5)], True),
        ("pkl", [i for i in range(1006)], True),
        ("pkl", [i for i in range(100000)], True),
    ],
    indirect=True,
)
def test_optimizer_cache_pkl_break_writes(setup_cache_helper_class):
    setup_cache_helper_class.check_break()


@pytest.mark.skip()  # TODO: FIXME
@pytest.mark.parametrize(
    "setup_cache_helper_class",
    [
        ("pkl", [i for i in range(5)], True),
        ("pkl", [i for i in range(100000)], True),
    ],
    indirect=True,
)
def test_optimizer_cache_pkl_reads_break(setup_cache_helper_class):
    setup_cache_helper_class.check_break()
    setup_cache_helper_class.cache()
    setup_cache_helper_class.setup()
    setup_cache_helper_class.check_read()


# -- CHECK MULTIPROCESS CACHE --


@pytest.mark.parametrize(
    "setup_cache_helper_class_mp",
    [
        ("pkl", [i for i in range(5)], True, 1),
        ("pkl", [i for i in range(100000)], True, 1),
        ("pkl", [i for i in range(5)], True, 3),
        ("pkl", [i for i in range(100000)], True, 3),
        ("pkl", [i for i in range(5)], True, 5),
        ("pkl", [i for i in range(100000)], True, 5),
    ],
    indirect=True,
)
def test_mp_cache_caching_pkl(setup_cache_helper_class_mp):
    setup_cache_helper_class_mp.cache(clean=True)


@pytest.mark.parametrize(
    "setup_cache_helper_class_mp",
    [
        ("pt", [i for i in range(5)], True, 1),
        ("pt", [i for i in range(100000)], True, 1),
        ("pt", [i for i in range(5)], True, 3),
        ("pt", [i for i in range(100000)], True, 3),
        ("pt", [i for i in range(5)], True, 5),
        ("pt", [i for i in range(100000)], True, 5),
    ],
    indirect=True,
)
def test_mp_cache_caching_pt(setup_cache_helper_class_mp):
    setup_cache_helper_class_mp.cache(clean=True)


@pytest.mark.parametrize(
    "setup_cache_helper_class_mp",
    [
        ("pkl", [i for i in range(100000)], False, 3),
        ("pt", [i for i in range(100000)], False, 3),
    ],
    indirect=True,
)
def test_mp_cache_caching_changed_data(setup_cache_helper_class_mp):
    setup_cache_helper_class_mp.cache(
        clean=True, expected_data=[i + 1000 for i in range(100000)]
    )


@pytest.mark.parametrize(
    "setup_cache_helper_class_mp",
    [
        ("pkl", [i for i in range(100000)], True, 3),
        ("pt", [i for i in range(100000)], True, 3),
    ],
    indirect=True,
)
def test_mp_cache_write_regular(setup_cache_helper_class_mp):
    setup_cache_helper_class_mp.cache(clean=False)
    setup_cache_helper_class_mp.check_write(clean=True)


@pytest.mark.parametrize(
    "setup_cache_helper_class_mp",
    [
        ("pkl", [i for i in range(100000)], False, 3),
        ("pt", [i for i in range(100000)], False, 3),
    ],
    indirect=True,
)
def test_mp_cache_write_changed_data(setup_cache_helper_class_mp):
    setup_cache_helper_class_mp.cache(
        clean=False, expected_data=[i + 1000 for i in range(100000)]
    )
    setup_cache_helper_class_mp.check_write(clean=True)


@pytest.mark.parametrize(
    "setup_cache_helper_class_mp",
    [
        ("pkl", [i for i in range(100000)], True, 3),
        ("pt", [i for i in range(100000)], True, 3),
    ],
    indirect=True,
)
def test_mp_cache_read_regular(setup_cache_helper_class_mp):
    setup_cache_helper_class_mp.cache(clean=False)
    setup_cache_helper_class_mp.check_write(clean=False)
    setup_cache_helper_class_mp.check_read(clean=True)


def test_prefetcher():
    from cedar.pipes.optimize import PrefetcherPipe

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = PrefetcherPipe(ft)
            ft = NoopPipe(ft)
            ft = PrefetcherPipe(ft)
            ft = NoopPipe(ft)

            return ft

    data = range(100)
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    dataset = DataSet(ctx, {"feature": test_feature}, enable_controller=False)
    out = []

    for x in dataset:
        out.append(x)

    assert out == list(data)


def test_fusion_basic():
    data = range(100)
    source = IterSource(data)
    ctx = CedarContext()

    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    noop2 = NoopPipe(noop1)
    source_pipe.id = 0
    noop1.id = 1
    noop2.id = 2

    fused_pipe = FusedOptimizerPipe([noop1, noop2])

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    fused_pipe.mutate(ctx, PipeVariantType.SMP)

    out = fused_pipe.pipe_variant
    res = []

    for x in out:
        res.append(x.data)

    assert set(res) == set(data)

    for x in out:
        res.append(x.data)


def _add(x, y):
    return x + y


def _is_even(x):
    return x % 2 == 0


def test_fusion_map():
    add_one = functools.partial(_add, y=1)

    data = range(100)
    source = IterSource(data)
    ctx = CedarContext()

    source_pipe = source.to_pipe()
    p1 = MapperPipe(source_pipe, add_one)
    p2 = MapperPipe(p1, add_one)
    p3 = MapperPipe(p2, add_one)

    source_pipe.id = 0
    p1.id = 1
    p2.id = 2
    p3.id = 3

    fused_pipe = FusedOptimizerPipe([p1, p2, p3])

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    fused_pipe.mutate(ctx, PipeVariantType.SMP)

    out = fused_pipe.pipe_variant
    res = []
    for x in out:
        res.append(x.data)

    assert set(res) == set(range(3, 103))


def test_fusion_filter_between_maps():
    add_one = functools.partial(_add, y=1)

    data = range(10)
    source = IterSource(data)
    ctx = CedarContext()

    source_pipe = source.to_pipe()
    p1 = MapperPipe(source_pipe, add_one)
    p2 = FilterPipe(p1, _is_even)
    p3 = MapperPipe(p2, add_one)

    source_pipe.id = 0
    p1.id = 1
    p2.id = 2
    p3.id = 3

    fused_pipe = FusedOptimizerPipe([p1, p2, p3])

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    fused_pipe.mutate(ctx, PipeVariantType.SMP)

    out = fused_pipe.pipe_variant
    res = []
    for x in out:
        res.append(x.data)

    assert set(res) == {3, 5, 7, 9, 11}


def test_fusion_feature(basic_map_dataset_long):
    # Start off by reading some elements
    it = iter(basic_map_dataset_long)
    out = []
    for _ in range(300):
        out.append(next(it))

    # Trigger a fusion to merge 2 and 1
    # 3 (source) -> 2 (map) -> 1 (map) -> 0 (noop)
    feature = basic_map_dataset_long.features["feature"]
    feature.dynamic_fusion(
        [2, 1], wait_for_mutation=False
    )  # needs to be in order

    for _ in range(700):
        out.append(next(it))

    assert set(out) == set(range(2, 1002))
    assert feature.physical_adj_list[3] == {4}
    assert (
        2 not in feature.physical_adj_list
        and 1 not in feature.physical_adj_list
    )
    assert (
        feature.physical_pipes[4].input_pipes[0] == feature.physical_pipes[3]
    )
    assert (
        feature.physical_pipes[0].pipe_variant.input_pipe_variant
        == feature.physical_pipes[4].pipe_variant
    )

    # Try another epoch
    out = []
    for x in basic_map_dataset_long:
        out.append(x)

    assert set(out) == set(range(2, 1002))
    assert feature.physical_adj_list[3] == {4}
    assert (
        feature.physical_pipes[4].input_pipes[0] == feature.physical_pipes[3]
    )
    assert (
        feature.physical_pipes[0].pipe_variant.input_pipe_variant
        == feature.physical_pipes[4].pipe_variant
    )


def test_fusion_threaded(basic_map_dataset_long):
    # Test triggering a fusion from a separate thread
    def _test_thread(feature: Feature):
        time.sleep(0.5)
        feature.dynamic_fusion([2, 1])
        time.sleep(1)

    feature = basic_map_dataset_long.features["feature"]

    thread = threading.Thread(target=_test_thread, args=(feature,))
    it = iter(basic_map_dataset_long)
    out = []

    thread.start()
    for _ in range(1000):
        out.append(next(it))
        time.sleep(0.001)

    assert set(out) == set(range(2, 1002))
    assert feature.physical_adj_list[3] == {4}
    assert (
        feature.physical_pipes[4].input_pipes[0] == feature.physical_pipes[3]
    )
    assert (
        feature.physical_pipes[0].pipe_variant.input_pipe_variant
        == feature.physical_pipes[4].pipe_variant
    )
    thread.join()


def test_fusion_split(basic_3map_dataset_long):
    # Start off by reading some elements
    it = iter(basic_3map_dataset_long)
    out = []
    for _ in range(300):
        out.append(next(it))

    # Trigger a fusion to merge 2 and 1
    # 4 (source) -> 3 (map) -> 2 (map) -> 1 (map) -> 0 (noop)
    feature = basic_3map_dataset_long.features["feature"]
    feature.dynamic_fusion(
        [2, 1], wait_for_mutation=False
    )  # needs to be in order

    for _ in range(300):
        out.append(next(it))

    assert feature.physical_adj_list[3] == {5}
    assert feature.physical_adj_list[5] == {0}
    assert (
        2 not in feature.physical_adj_list
        and 1 not in feature.physical_adj_list
    )
    assert (
        feature.physical_pipes[5].input_pipes[0] == feature.physical_pipes[3]
    )
    assert (
        feature.physical_pipes[0].input_pipes[0] == feature.physical_pipes[5]
    )
    assert (
        feature.physical_pipes[0].pipe_variant.input_pipe_variant
        == feature.physical_pipes[5].pipe_variant
    )

    feature.dynamic_split(5, wait_for_mutation=False)

    for _ in range(400):
        out.append(next(it))

    assert set(out) == set(range(3, 1003))
    assert feature.physical_pipes[5].pipe_variant is None
    assert feature.physical_adj_list[3] == {2}
    assert feature.physical_adj_list[2] == {1}
    assert feature.physical_adj_list[1] == {0}
    assert 5 not in feature.physical_adj_list
    assert (
        feature.physical_pipes[2].input_pipes[0] == feature.physical_pipes[3]
    )
    assert (
        feature.physical_pipes[1].input_pipes[0] == feature.physical_pipes[2]
    )
    assert (
        feature.physical_pipes[0].input_pipes[0] == feature.physical_pipes[1]
    )
    assert (
        feature.physical_pipes[0].pipe_variant.input_pipe_variant
        == feature.physical_pipes[1].pipe_variant
    )

    # Try another epoch
    it = iter(basic_3map_dataset_long)
    out = []
    for _ in range(300):
        out.append(next(it))

    feature.dynamic_fusion([3, 2, 1], wait_for_mutation=False)
    for _ in range(300):
        out.append(next(it))

    assert feature.physical_adj_list[4] == {6}
    assert feature.physical_adj_list[6] == {0}
    assert (
        3 not in feature.physical_adj_list
        and 2 not in feature.physical_adj_list
        and 1 not in feature.physical_adj_list
    )
    assert (
        feature.physical_pipes[6].input_pipes[0] == feature.physical_pipes[4]
    )
    assert (
        feature.physical_pipes[0].input_pipes[0] == feature.physical_pipes[6]
    )
    assert (
        feature.physical_pipes[0].pipe_variant.input_pipe_variant
        == feature.physical_pipes[6].pipe_variant
    )
    feature.dynamic_split(6, wait_for_mutation=False)
    for _ in range(400):
        out.append(next(it))
    assert set(out) == set(range(3, 1003))
    assert feature.physical_pipes[6].pipe_variant is None
    assert feature.physical_adj_list[4] == {3}
    assert feature.physical_adj_list[3] == {2}
    assert feature.physical_adj_list[2] == {1}
    assert feature.physical_adj_list[1] == {0}
    assert 5 not in feature.physical_adj_list
    assert (
        feature.physical_pipes[2].input_pipes[0] == feature.physical_pipes[3]
    )
    assert (
        feature.physical_pipes[1].input_pipes[0] == feature.physical_pipes[2]
    )
    assert (
        feature.physical_pipes[0].input_pipes[0] == feature.physical_pipes[1]
    )
    assert (
        feature.physical_pipes[0].pipe_variant.input_pipe_variant
        == feature.physical_pipes[1].pipe_variant
    )


def test_fusion_split_threaded(basic_3map_dataset_long):
    def _test_thread(feature: Feature):
        time.sleep(0.25)
        feature.dynamic_fusion([2, 1])
        time.sleep(0.25)
        feature.dynamic_split(5)
        time.sleep(0.25)
        feature.dynamic_fusion([3, 2, 1])
        time.sleep(0.25)
        feature.dynamic_split(6)
        time.sleep(0.25)

    feature = basic_3map_dataset_long.features["feature"]

    thread = threading.Thread(target=_test_thread, args=(feature,))
    out = []

    thread.start()

    for i in range(2):
        for x in basic_3map_dataset_long:
            out.append(x + i * 1000)
            time.sleep(0.001)

    assert (
        feature.physical_pipes[6].pipe_variant is None
        and feature.physical_pipes[5].pipe_variant is None
    )
    assert feature.physical_adj_list[4] == {3}
    assert feature.physical_adj_list[3] == {2}
    assert feature.physical_adj_list[2] == {1}
    assert feature.physical_adj_list[1] == {0}
    assert 5 not in feature.physical_adj_list
    assert 6 not in feature.physical_adj_list
    assert (
        feature.physical_pipes[3].input_pipes[0] == feature.physical_pipes[4]
    )
    assert (
        feature.physical_pipes[2].input_pipes[0] == feature.physical_pipes[3]
    )
    assert (
        feature.physical_pipes[1].input_pipes[0] == feature.physical_pipes[2]
    )
    assert (
        feature.physical_pipes[0].input_pipes[0] == feature.physical_pipes[1]
    )
    assert (
        feature.physical_pipes[0].pipe_variant.input_pipe_variant
        == feature.physical_pipes[1].pipe_variant
    )
    assert set(out) == set(range(3, 2003))
    thread.join()


def test_fusion_reset_sync(basic_3map_dataset_long):
    # Start off by reading some elements
    it = iter(basic_3map_dataset_long)
    out = []
    for _ in range(300):
        out.append(next(it))

    # Trigger a fusion to merge 2 and 1
    # 4 (source) -> 3 (map) -> 2 (map) -> 1 (map) -> 0 (noop)
    feature = basic_3map_dataset_long.features["feature"]
    feature.dynamic_fusion(
        [2, 1], wait_for_mutation=False
    )  # needs to be in order

    for _ in range(300):
        out.append(next(it))

    assert feature.physical_adj_list[3] == {5}
    assert feature.physical_adj_list[5] == {0}
    assert (
        2 not in feature.physical_adj_list
        and 1 not in feature.physical_adj_list
    )
    assert (
        feature.physical_pipes[5].input_pipes[0] == feature.physical_pipes[3]
    )
    assert (
        feature.physical_pipes[0].input_pipes[0] == feature.physical_pipes[5]
    )
    assert (
        feature.physical_pipes[0].pipe_variant.input_pipe_variant
        == feature.physical_pipes[5].pipe_variant
    )
    assert 5 in feature.physical_adj_list

    feature.reset_pipes([5], wait_for_mutation=False)

    for _ in range(200):
        out.append(next(it))
    feature.wait_for_mutation([5])
    feature.dynamic_fusion([3, 2, 1], wait_for_mutation=False)

    for _ in range(150):
        out.append(next(it))

    feature.reset_pipes([6], wait_for_mutation=False)
    for _ in range(50):
        out.append(next(it))

    with pytest.raises(StopIteration):
        next(it)

    assert set(out) == set(range(3, 1003))

    # Try another epoch
    it = iter(basic_3map_dataset_long)
    out = []
    for _ in range(100):
        out.append(next(it))

    feature.wait_for_mutation([6])
    assert feature.physical_adj_list[4] == {3}
    assert feature.physical_adj_list[3] == {2}
    assert feature.physical_adj_list[2] == {1}
    assert feature.physical_adj_list[1] == {0}
    assert (
        feature.physical_pipes[2].input_pipes[0] == feature.physical_pipes[3]
    )
    assert (
        feature.physical_pipes[1].input_pipes[0] == feature.physical_pipes[2]
    )
    assert (
        feature.physical_pipes[0].input_pipes[0] == feature.physical_pipes[1]
    )
    assert (
        feature.physical_pipes[0].pipe_variant.input_pipe_variant
        == feature.physical_pipes[1].pipe_variant
    )

    assert 6 not in feature.physical_adj_list
    for _ in range(900):
        out.append(next(it))
    assert set(out) == set(range(3, 1003))


def test_fusion_reset_async(basic_3map_dataset_long):
    # Start off by reading some elements
    it = iter(basic_3map_dataset_long)
    out = []
    for _ in range(100):
        out.append(next(it))

    # Trigger a fusion to merge 2 and 1
    # 4 (source) -> 3 (map) -> 2 (map) -> 1 (map) -> 0 (noop)
    feature = basic_3map_dataset_long.features["feature"]

    feature.dynamic_mutate(
        2, PipeVariantType.MULTITHREADED, wait_for_mutation=False
    )
    feature.dynamic_mutate(3, PipeVariantType.SMP, wait_for_mutation=False)

    for _ in range(200):
        out.append(next(it))

    feature.reset_pipes([2], wait_for_mutation=False)

    for _ in range(200):
        out.append(next(it))
    feature.wait_for_mutation([2])
    feature.dynamic_fusion(
        [2, 1], wait_for_mutation=False
    )  # needs to be in order

    for _ in range(200):
        out.append(next(it))

    assert feature.physical_adj_list[3] == {5}
    assert feature.physical_adj_list[5] == {0}
    assert (
        2 not in feature.physical_adj_list
        and 1 not in feature.physical_adj_list
    )
    assert (
        feature.physical_pipes[5].input_pipes[0] == feature.physical_pipes[3]
    )
    assert (
        feature.physical_pipes[0].input_pipes[0] == feature.physical_pipes[5]
    )
    assert (
        feature.physical_pipes[0].pipe_variant.input_pipe_variant
        == feature.physical_pipes[5].pipe_variant
    )

    feature.reset_pipes([5, 3], wait_for_mutation=False)

    for _ in range(200):
        out.append(next(it))
    feature.wait_for_mutation([5, 3])
    feature.dynamic_fusion([3, 2, 1], wait_for_mutation=False)

    for _ in range(100):
        out.append(next(it))

    feature.reset_pipes([6], wait_for_mutation=False)

    with pytest.raises(StopIteration):
        next(it)

    assert set(out) == set(range(3, 1003))

    # Try another epoch
    it = iter(basic_3map_dataset_long)
    out = []
    for _ in range(200):
        out.append(next(it))

    feature.wait_for_mutation([6])
    assert feature.physical_adj_list[4] == {3}
    assert feature.physical_adj_list[3] == {2}
    assert feature.physical_adj_list[2] == {1}
    assert feature.physical_adj_list[1] == {0}
    assert (
        feature.physical_pipes[2].input_pipes[0] == feature.physical_pipes[3]
    )
    assert (
        feature.physical_pipes[1].input_pipes[0] == feature.physical_pipes[2]
    )
    assert (
        feature.physical_pipes[0].input_pipes[0] == feature.physical_pipes[1]
    )
    assert (
        feature.physical_pipes[0].pipe_variant.input_pipe_variant
        == feature.physical_pipes[1].pipe_variant
    )

    assert 6 not in feature.physical_adj_list
    for _ in range(800):
        out.append(next(it))
    assert set(out) == set(range(3, 1003))

    with pytest.raises(StopIteration):
        next(it)


def test_fusion_reset_threaded(basic_3map_dataset_long):
    def _test_thread(feature: Feature):
        time.sleep(0.25)
        feature.dynamic_mutate(2, PipeVariantType.SMP)
        feature.wait_for_mutation([2])
        feature.dynamic_mutate(3, PipeVariantType.SMP)
        feature.wait_for_mutation([3])
        feature.dynamic_mutate(2, PipeVariantType.MULTITHREADED)
        feature.wait_for_mutation([2])
        time.sleep(0.25)
        feature.reset_pipes([2])
        feature.wait_for_mutation([2])
        feature.dynamic_fusion([2, 1])
        time.sleep(0.25)
        feature.wait_for_mutation([5])
        time.sleep(0.25)
        feature.reset_pipes([5, 3])
        time.sleep(0.25)
        feature.wait_for_mutation([5, 3])
        feature.dynamic_fusion([3, 2, 1])
        time.sleep(0.25)
        feature.reset_pipes([6])
        time.sleep(0.25)
        feature.wait_for_mutation([6])

    feature = basic_3map_dataset_long.features["feature"]

    thread = threading.Thread(target=_test_thread, args=(feature,))
    out = []

    thread.start()
    # Start off by reading some elements

    for i in range(3):
        for x in basic_3map_dataset_long:
            out.append(x + i * 1000)
            time.sleep(0.001)

    assert (
        feature.physical_pipes[6].pipe_variant is None
        and feature.physical_pipes[5].pipe_variant is None
    )
    assert feature.physical_adj_list[4] == {3}
    assert feature.physical_adj_list[3] == {2}
    assert feature.physical_adj_list[2] == {1}
    assert feature.physical_adj_list[1] == {0}
    assert 5 not in feature.physical_adj_list
    assert 6 not in feature.physical_adj_list
    assert (
        feature.physical_pipes[3].input_pipes[0] == feature.physical_pipes[4]
    )
    assert (
        feature.physical_pipes[2].input_pipes[0] == feature.physical_pipes[3]
    )
    assert (
        feature.physical_pipes[1].input_pipes[0] == feature.physical_pipes[2]
    )
    assert (
        feature.physical_pipes[0].input_pipes[0] == feature.physical_pipes[1]
    )
    assert (
        feature.physical_pipes[0].pipe_variant.input_pipe_variant
        == feature.physical_pipes[1].pipe_variant
    )
    assert set(out) == set(range(3, 3003))
    thread.join()


def _add_one(x):
    return x + 1


def test_load_fuse_ray(setup_ray):
    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = MapperPipe(ft, _add_one)
            ft = MapperPipe(ft, _add_one)
            ft = NoopPipe(ft)
            return ft

    data = range(100)
    source = IterSource(data)
    test_feature = TestFeature()
    test_feature.apply(source)

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_config_file = pathlib.Path(test_dir) / "data/config_fuse_ray.yml"
    ctx = CedarContext()

    ds = DataSet(
        ctx,
        {"feature": test_feature},
        feature_config=str(ref_config_file),
        enable_controller=False,
        enable_optimizer=False,
    )

    assert ds.feature_plans["feature"] is not None
    assert 6 in test_feature.physical_adj_list
    assert 2 not in test_feature.physical_adj_list
    assert 1 not in test_feature.physical_adj_list

    assert (
        test_feature.physical_pipes[6].pipe_variant_type == PipeVariantType.RAY
    )
    assert test_feature.physical_pipes[2].pipe_variant is None
    assert test_feature.physical_pipes[1].pipe_variant is None

    res = []
    for x in ds:
        res.append(x)

    assert set(res) == set(range(2, 102))

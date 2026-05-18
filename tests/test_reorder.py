import pytest
from typing import List
from cedar.config import CedarContext
from cedar.compose import Feature
from cedar.client import DataSet
from cedar.pipes import NoopPipe, Pipe, MapperPipe
from cedar.sources import IterSource


def _add_one(x):
    return x + 1


def _double(x):
    return x * 2


def test_reordering_api_basic():
    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft, tag="noop1").fix()
            ft = MapperPipe(ft, _add_one, tag="add_one").depends_on(["noop1"])
            ft = MapperPipe(ft, _double, tag="double").fix()
            ft = NoopPipe(ft, tag="noop2").depends_on(["double", "add_one"])
            ft = NoopPipe(ft, tag="noop3").depends_on(["noop2"])
            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    dataset = DataSet(
        ctx, {"feature": test_feature}, prefetch=False, enable_controller=False
    )
    out = []

    for x in dataset:
        out.append(x)

    assert out == [4, 6, 8]

    assert test_feature.get_pipe(0)._depends_on_tags == ["noop2"]
    assert test_feature.get_pipe(1)._depends_on_tags == ["double", "add_one"]
    assert test_feature.get_pipe(2)._fix_order is True
    assert test_feature.get_pipe(3)._depends_on_tags == ["noop1"]
    assert test_feature.get_pipe(4)._fix_order is True
    assert test_feature.get_pipe(5)._fix_order is True


def test_cycle():
    from cedar.compose.utils import derive_constraint_graph

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft, tag="noop1").depends_on(["noop3"])
            ft = NoopPipe(ft, tag="noop2").depends_on(["noop1"])
            ft = NoopPipe(ft, tag="noop3").depends_on(["noop2"])
            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)
    with pytest.raises(ValueError):
        derive_constraint_graph(test_feature.logical_pipes)


def test_calculate_reorderings_timeout():
    from cedar.compose.utils import calculate_reorderings

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            return ft

    test_feature = TestFeature()
    test_feature.apply(IterSource([1, 2, 3]))

    with pytest.raises(TimeoutError):
        calculate_reorderings(
            test_feature.logical_pipes,
            test_feature.logical_adj_list,
            timeout_sec=0,
        )


def test_constraint_graph():
    from cedar.compose.utils import derive_constraint_graph

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft, tag="noop1").fix()
            ft = MapperPipe(ft, _add_one, tag="add_one").depends_on(["noop1"])
            ft = MapperPipe(ft, _double, tag="double").fix()
            ft = NoopPipe(ft, tag="noop2").depends_on(["double", "add_one"])
            ft = NoopPipe(ft, tag="noop3").depends_on(["noop2"])
            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    constraint_graph = derive_constraint_graph(test_feature.logical_pipes)

    assert constraint_graph == {
        0: set(),
        1: {0},
        2: {1},
        3: {1},
        4: {3},
        5: set(),
    }


def _get_dict_key(d):
    return tuple((k, tuple(sorted(v))) for k, v in sorted(d.items()))


def test_reordering_algorithm_basic():
    from cedar.compose.utils import calculate_reorderings

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft, tag="noop1")
            ft = NoopPipe(ft, tag="noop2")
            ft = NoopPipe(ft, tag="noop3")
            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    reorderings = calculate_reorderings(
        test_feature.logical_pipes, test_feature.logical_adj_list
    )

    correct_reorderings = [
        {3: {2}, 2: {1}, 1: {0}, 0: set()},  # 3210
        {3: {2}, 2: {0}, 1: set(), 0: {1}},  # 3201
        {3: {1}, 2: {0}, 1: {2}, 0: set()},  # 3120
        {3: {1}, 2: set(), 1: {0}, 0: {2}},  # 3102
        {3: {0}, 2: set(), 1: {2}, 0: {1}},  # 3012
        {3: {0}, 2: {1}, 1: set(), 0: {2}},  # 3021
    ]

    assert sorted(
        correct_reorderings, key=lambda d: _get_dict_key(d)
    ) == sorted(reorderings, key=lambda d: _get_dict_key(d))


def test_reordering_algorithm_with_dependency():
    from cedar.compose.utils import calculate_reorderings

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft, tag="noop1")
            ft = NoopPipe(ft, tag="noop2")
            ft = NoopPipe(ft, tag="noop3").depends_on(["noop1"])
            ft = NoopPipe(ft, tag="noop4").depends_on(["noop1"])
            ft = NoopPipe(ft, tag="noop5").depends_on(["noop3"])
            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    reorderings = calculate_reorderings(
        test_feature.logical_pipes, test_feature.logical_adj_list
    )

    correct_reorderings = [
        {5: {4}, 4: {3}, 3: {2}, 2: {1}, 1: {0}, 0: set()},  # 543210
        {5: {4}, 4: {2}, 3: {1}, 2: {3}, 1: {0}, 0: set()},  # 542310
        {5: {4}, 4: {2}, 3: {0}, 2: {1}, 1: {3}, 0: set()},  # 542130
        {5: {4}, 4: {2}, 3: set(), 2: {1}, 1: {0}, 0: {3}},  # 542103
        {5: {3}, 4: {2}, 3: {4}, 2: {1}, 1: {0}, 0: set()},  # 534210
        {5: {4}, 4: {3}, 3: {1}, 2: {0}, 1: {2}, 0: set()},  # 543120
        {5: {4}, 4: {1}, 3: {2}, 2: {0}, 1: {3}, 0: set()},  # 541320
        {5: {4}, 4: {1}, 3: {0}, 2: {3}, 1: {2}, 0: set()},  # 541230
        {5: {4}, 4: {1}, 3: set(), 2: {0}, 1: {2}, 0: {3}},  # 541203
        {5: {3}, 4: {1}, 3: {4}, 2: {0}, 1: {2}, 0: set()},  # 534120
        {5: {4}, 4: {3}, 3: {2}, 2: {0}, 1: set(), 0: {1}},  # 543201
        {5: {4}, 4: {2}, 3: {0}, 2: {3}, 1: set(), 0: {1}},  # 542301
        {5: {4}, 4: {2}, 3: {1}, 2: {0}, 1: set(), 0: {3}},  # 542031
        {5: {4}, 4: {2}, 3: set(), 2: {0}, 1: {3}, 0: {1}},  # 542013
        {5: {3}, 4: {2}, 3: {4}, 2: {0}, 1: set(), 0: {1}},  # 534201
    ]

    assert sorted(
        correct_reorderings, key=lambda d: _get_dict_key(d)
    ) == sorted(reorderings, key=lambda d: _get_dict_key(d))


def test_reordering_algorithm_with_fix():
    from cedar.compose.utils import calculate_reorderings

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft, tag="noop1")
            ft = NoopPipe(ft, tag="noop2")
            ft = NoopPipe(ft, tag="noop3").depends_on(["noop1"]).fix()
            ft = NoopPipe(ft, tag="noop4").depends_on(["noop1"])
            ft = NoopPipe(ft, tag="noop5").depends_on(["noop3"])
            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    assert test_feature.get_pipe(2)._fix_order is True
    assert test_feature.get_pipe(2)._depends_on_tags == ["noop1"]

    reorderings = calculate_reorderings(
        test_feature.logical_pipes, test_feature.logical_adj_list
    )

    correct_reorderings = [
        {5: {4}, 4: {3}, 3: {2}, 2: {1}, 1: {0}, 0: set()},  # 543210
        {5: {3}, 4: {2}, 3: {4}, 2: {1}, 1: {0}, 0: set()},  # 534210
        {5: {4}, 4: {3}, 3: {2}, 2: {0}, 1: set(), 0: {1}},  # 543201
        {5: {3}, 4: {2}, 3: {4}, 2: {0}, 1: set(), 0: {1}},  # 534201
    ]

    assert sorted(
        correct_reorderings, key=lambda d: _get_dict_key(d)
    ) == sorted(reorderings, key=lambda d: _get_dict_key(d))

from unittest.mock import Mock

import threadpoolctl

from cedar.utils.threading import limit_native_threadpools


def test_limit_native_threadpools_sets_future_and_loaded_pool_limits(
    monkeypatch,
):
    limiter = object()
    mocked_limits = Mock(return_value=limiter)
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", mocked_limits)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.delenv(variable, raising=False)

    assert limit_native_threadpools(1) is limiter
    mocked_limits.assert_called_once_with(limits=1)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert __import__("os").environ[variable] == "1"

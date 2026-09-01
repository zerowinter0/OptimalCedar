import pytest

from cedar.pipes.ray_variant import (
    RAY_PLACEMENT_RESOURCE_ENV,
    RAY_PLACEMENT_RESOURCE_FRACTION_ENV,
    get_ray_actor_options,
)


def test_ray_actor_options_preserve_default_placement(monkeypatch):
    monkeypatch.delenv(RAY_PLACEMENT_RESOURCE_ENV, raising=False)
    monkeypatch.delenv(RAY_PLACEMENT_RESOURCE_FRACTION_ENV, raising=False)

    assert get_ray_actor_options(0.25) == {
        "num_cpus": 1.0,
        "num_gpus": 0.25,
    }


def test_ray_actor_options_add_remote_resource(monkeypatch):
    monkeypatch.setenv(RAY_PLACEMENT_RESOURCE_ENV, "cedar_remote")
    monkeypatch.delenv(RAY_PLACEMENT_RESOURCE_FRACTION_ENV, raising=False)

    assert get_ray_actor_options() == {
        "num_cpus": 1.0,
        "num_gpus": 0.0,
        "resources": {"cedar_remote": 0.001},
    }


@pytest.mark.parametrize("value", ["0", "-1", "nan", "not-a-number"])
def test_ray_actor_options_reject_invalid_fraction(monkeypatch, value):
    monkeypatch.setenv(RAY_PLACEMENT_RESOURCE_ENV, "cedar_remote")
    monkeypatch.setenv(RAY_PLACEMENT_RESOURCE_FRACTION_ENV, value)

    with pytest.raises(ValueError):
        get_ray_actor_options()

import json
from pathlib import Path

import pytest

from evaluation.pipelines.simclrv2_multimodal.validate_gpu_prefix_plans import (
    validate,
)


def _payload(variant: str = "INPROCESS", include_candidate: bool = False) -> dict:
    run = {
        "physical_plans_by_feature": {
            "feature": {
                "graph": {"0": ""},
                "pipes": {
                    "0": {
                        "name": "FilterPipe_ClipPredicate",
                        "variant": variant,
                        "variant_ctx": {"variant_type": variant},
                        "execution_resource": "cuda",
                    }
                },
            }
        }
    }
    runs = [
            {**run, "optimizer": "optimizer"},
            {**run, "optimizer": "simple_dp_optimizer"},
            {**run, "optimizer": "dp_optimizer"},
    ]
    if include_candidate:
        runs.append({**run, "optimizer": "simple_dp_ray_candidate_optimizer"})
    return {"runs": runs}


def test_validator_accepts_one_independent_local_gpu_stage_per_plan(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    validate(path)


def test_validator_rejects_nonlocal_gpu_stage(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_payload("RAY")), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not fixed to local INPROCESS"):
        validate(path)


def test_validator_accepts_explicit_four_optimizer_set(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(_payload(include_candidate=True)), encoding="utf-8"
    )
    validate(
        path,
        expected_optimizers={
            "optimizer",
            "simple_dp_optimizer",
            "dp_optimizer",
            "simple_dp_ray_candidate_optimizer",
        },
    )

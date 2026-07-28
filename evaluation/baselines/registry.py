"""Declarative support registry for the external-system comparison matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


WORKLOADS = (
    "coco",
    "commonvoice",
    "commonvoice_cache",
    "llava_pretrain",
    "redpajama_c4",
    "simclrv2",
    "simclrv2_cache",
    "stackexchange",
    "wikitext103",
    "wikitext103_cache",
    "pile_europarl",
    "redpajama_code",
    "pile_hackernews",
    "pile_pubmed_abstracts",
    "pile_freelaw",
    "pile_uspto_backgrounds",
)

SYSTEMS = (
    "pytorch",
    "tensorflow",
    "ray",
    "datajuicer",
    "plumber",
    "fastflow",
)

BASE_WORKLOAD = {
    "commonvoice_cache": "commonvoice",
    "simclrv2_cache": "simclrv2",
    "wikitext103_cache": "wikitext103",
}

MODULE_FILENAMES = {
    "pytorch": "torch_dataset.py",
    "tensorflow": "tf_dataset.py",
    "ray": "ray_dataset.py",
}

DATAJUICER_CONFIGS = {
    "llava_pretrain": (
        "evaluation/baselines/datajuicer/llava_pretrain.yaml"
    ),
    "redpajama_c4": (
        "evaluation/baselines/datajuicer/redpajama_c4.yaml"
    ),
    "stackexchange": (
        "evaluation/baselines/datajuicer/stackexchange.yaml"
    ),
}


@dataclass(frozen=True)
class BaselineEntry:
    system: str
    workload: str
    status: str
    implementation: Optional[str]
    execution_env: str
    cache_policy: str = "not_applicable"
    backend: str = "native"
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _native_entry(system: str, workload: str) -> BaselineEntry:
    base = BASE_WORKLOAD.get(workload, workload)
    module = (
        Path("evaluation")
        / "pipelines"
        / base
        / MODULE_FILENAMES[system]
    )
    backend = "native"
    if base in {
        "llava_pretrain",
        "redpajama_c4",
        "stackexchange",
        "wikitext103",
        "pile_europarl",
        "redpajama_code",
        "pile_hackernews",
        "pile_pubmed_abstracts",
        "pile_freelaw",
        "pile_uspto_backgrounds",
    }:
        backend = (
            "tf_py_function"
            if system == "tensorflow"
            else "native_python_operators"
        )

    cache_policy = "not_requested"
    if workload.endswith("_cache"):
        cache_policy = {
            "pytorch": "none_no_native_dataset_cache",
            "tensorflow": "tf_data_file_cache_final_output",
            "ray": "ray_data_materialize_final_output",
        }[system]

    status = "supported"
    reason = None
    if system == "tensorflow" and workload == "redpajama_c4":
        status = "unsupported"
        reason = (
            "The semantically equivalent pipeline requires opaque Python "
            "FastText, SentencePiece, and KenLM callbacks. tf.py_function is "
            "GIL-bound at approximately one effective CPU core, so the "
            "required W=8 comparison is infeasible."
        )

    return BaselineEntry(
        system=system,
        workload=workload,
        status=status,
        implementation=str(module),
        execution_env="optimalcedar-torch201-dev",
        cache_policy=cache_policy,
        backend=backend,
        reason=reason,
    )


def _datajuicer_entry(workload: str) -> BaselineEntry:
    config = DATAJUICER_CONFIGS.get(workload)
    if config:
        return BaselineEntry(
            system="datajuicer",
            workload=workload,
            status="supported",
            implementation=config,
            execution_env="optimalcedar-datajuicer",
            backend="native_recipe_probe_fusion",
        )
    return BaselineEntry(
        system="datajuicer",
        workload=workload,
        status="unsupported",
        implementation=None,
        execution_env="optimalcedar-datajuicer",
        reason=(
            "No semantically equivalent native Data-Juicer reference recipe; "
            "adding custom training-time augmentation operators would no "
            "longer measure Data-Juicer's system."
        ),
    )


def _plumber_entry(workload: str) -> BaselineEntry:
    base = BASE_WORKLOAD.get(workload, workload)
    if base in {
        "llava_pretrain",
        "redpajama_c4",
        "stackexchange",
        "wikitext103",
        "pile_europarl",
        "redpajama_code",
        "pile_hackernews",
        "pile_pubmed_abstracts",
        "pile_freelaw",
        "pile_uspto_backgrounds",
    }:
        return BaselineEntry(
            system="plumber",
            workload=workload,
            status="unsupported",
            implementation=None,
            execution_env="optimalcedar-plumber",
            reason=(
                "The pipeline contains an operator that Plumber cannot model "
                "and safely reconstruct (opaque Python callbacks or a "
                "TextLineDataset FlatMap source without Plumber byte-ratio "
                "metadata)."
            ),
        )
    return BaselineEntry(
        system="plumber",
        workload=workload,
        status="supported",
        implementation=f"evaluation/plumber/{base}/run_plumber.sh",
        execution_env="optimalcedar-plumber",
        cache_policy=(
            "plumber_recommendation"
            if workload.endswith("_cache")
            else "not_requested"
        ),
        backend="tf_data_graph_rewrite",
    )


def _fastflow_entry(workload: str) -> BaselineEntry:
    base = BASE_WORKLOAD.get(workload, workload)
    if base in {
        "llava_pretrain",
        "redpajama_c4",
        "stackexchange",
        "pile_europarl",
        "redpajama_code",
        "pile_hackernews",
        "pile_pubmed_abstracts",
        "pile_freelaw",
        "pile_uspto_backgrounds",
    }:
        return BaselineEntry(
            system="fastflow",
            workload=workload,
            status="unsupported",
            implementation=None,
            execution_env="optimalcedar-fastflow",
            reason=(
                "FastFlow requires a serializable tf.data graph for remote "
                "partitioning; these workloads contain Python/Hugging Face "
                "callbacks that are opaque to tf.data service."
            ),
        )
    return BaselineEntry(
        system="fastflow",
        workload=workload,
        status="supported",
        implementation=f"evaluation/fastflow/workloads/{base}_app.py",
        execution_env="optimalcedar-fastflow",
        cache_policy=(
            "none_no_fastflow_cache_policy"
            if workload.endswith("_cache")
            else "not_requested"
        ),
        backend="fastflow_auto_offload",
    )


def get_entry(system: str, workload: str) -> BaselineEntry:
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system {system!r}; expected one of {SYSTEMS}")
    if workload not in WORKLOADS:
        raise ValueError(
            f"Unknown workload {workload!r}; expected one of {WORKLOADS}"
        )
    if system in MODULE_FILENAMES:
        return _native_entry(system, workload)
    if system == "datajuicer":
        return _datajuicer_entry(workload)
    if system == "plumber":
        return _plumber_entry(workload)
    return _fastflow_entry(workload)


def iter_entries() -> Iterable[BaselineEntry]:
    for workload in WORKLOADS:
        for system in SYSTEMS:
            yield get_entry(system, workload)


def matrix_as_dict() -> Dict[str, Dict[str, dict]]:
    return {
        workload: {
            system: get_entry(system, workload).to_dict()
            for system in SYSTEMS
        }
        for workload in WORKLOADS
    }


__all__ = [
    "BASE_WORKLOAD",
    "BaselineEntry",
    "SYSTEMS",
    "WORKLOADS",
    "get_entry",
    "iter_entries",
    "matrix_as_dict",
]

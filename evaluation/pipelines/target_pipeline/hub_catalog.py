"""Data-Juicer Hub workloads admitted to the target-pipeline study.

The existing Cedar Pile registry is the executable migration of these frozen
Hub recipes. This catalog gives the migration a stable target-pipeline entry
point and records why each workload belongs in the study: its per-record
filters change surviving data volume and are reorderable in Cedar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class HubWorkload:
    name: str
    official_recipe: str
    dataset_path: str
    record_scalers: Tuple[str, ...]
    reorderable_scalers: Tuple[str, ...]


HUB_WORKLOADS = (
    HubWorkload(
        "pile_hackernews",
        "refined_recipes/pretrain/pile-hackernews-refine.yaml",
        "datasets/pile_hackernews/pile-hackernews-raw-100000.jsonl",
        ("text_length", "words_num", "alphanumeric", "perplexity"),
        ("text_length", "words_num", "alphanumeric", "perplexity"),
    ),
    HubWorkload(
        "pile_pubmed_abstracts",
        "refined_recipes/pretrain/pile-pubmed-abstract-refine.yaml",
        "datasets/pile_pubmed_abstracts/pile-pubmed-abstracts-raw-100000.jsonl",
        ("text_length", "words_num", "alphanumeric", "perplexity"),
        ("text_length", "words_num", "alphanumeric", "perplexity"),
    ),
    HubWorkload(
        "pile_freelaw",
        "refined_recipes/pretrain/pile-freelaw-refine.yaml",
        "datasets/pile_freelaw/pile-freelaw-raw-100000.jsonl",
        ("text_length", "words_num", "alphanumeric", "perplexity"),
        ("text_length", "words_num", "alphanumeric", "perplexity"),
    ),
    HubWorkload(
        "pile_europarl",
        "refined_recipes/pretrain/pile-europarl-refine.yaml",
        "datasets/pile_europarl/pile-europarl-raw.jsonl",
        ("text_length", "words_num", "alphanumeric", "perplexity"),
        ("text_length", "words_num", "alphanumeric", "perplexity"),
    ),
    HubWorkload(
        "pile_uspto_backgrounds",
        "refined_recipes/pretrain/pile-uspto-refine.yaml",
        "datasets/pile_uspto_backgrounds/pile-uspto-backgrounds-raw-100000.jsonl",
        ("text_length", "words_num", "alphanumeric", "perplexity"),
        ("text_length", "words_num", "alphanumeric", "perplexity"),
    ),
)

HUB_BY_NAME = {workload.name: workload for workload in HUB_WORKLOADS}


def get_hub_workload(name: str) -> HubWorkload:
    try:
        return HUB_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unknown target Hub workload: {name}") from exc


__all__ = ["HUB_WORKLOADS", "HUB_BY_NAME", "HubWorkload", "get_hub_workload"]

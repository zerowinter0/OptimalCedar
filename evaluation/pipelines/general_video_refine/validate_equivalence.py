#!/usr/bin/env python3
"""Compare Cedar predicates with pinned Data-Juicer decisions on one clip."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from data_juicer_bootstrap import ensure_data_juicer_path


ensure_data_juicer_path()

from data_juicer.ops.filter.language_id_score_filter import (  # noqa: E402
    LanguageIDScoreFilter as DJLanguageIDScoreFilter,
)
from data_juicer.ops.filter.perplexity_filter import (  # noqa: E402
    PerplexityFilter as DJPerplexityFilter,
)
from evaluation.pipelines.general_video_refine import dj_operators as ops  # noqa: E402


def comparable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [comparable(item) for item in value]
    return value


def assert_stats_close(actual, expected, name):
    actual = comparable(actual)
    expected = comparable(expected)
    if actual.keys() != expected.keys():
        raise AssertionError(
            f"{name}: stats keys differ: {actual.keys()} != {expected.keys()}"
        )
    for key in actual:
        left, right = actual[key], expected[key]
        if isinstance(left, float) or isinstance(right, float):
            if not np.allclose(left, right, rtol=1e-6, atol=1e-7):
                raise AssertionError(f"{name}/{key}: {left!r} != {right!r}")
        elif left != right:
            raise AssertionError(f"{name}/{key}: {left!r} != {right!r}")


def direct_single(op, sample):
    computed = op.compute_stats_single(sample, rank=0, context=False)
    return bool(op.process_single(computed)), computed


def direct_language(op, sample):
    computed = op.compute_stats_single(sample)
    return bool(op.process_single(computed)), computed


def direct_perplexity(op, sample):
    batch = {key: [copy.deepcopy(value)] for key, value in sample.items()}
    computed = op.compute_stats_batched(batch)
    decision = list(op.process_batched(computed))[0]
    sample[ops.FIELDS_STATS] = computed[ops.FIELDS_STATS][0]
    return bool(decision), sample


def main() -> None:
    root = Path("datasets/general_video_refine")
    first = next(
        iter((root / "msrvtt-video-text-200000.jsonl").open(encoding="utf-8"))
    )
    base = ops.SetVideoRootMapper(root / "videos")(ops.parse_json_line(first))

    text_pairs = [
        (
            "language_id_score_filter",
            ops.LanguageIDScoreFilter(lang="en", min_score=0.26311219),
            DJLanguageIDScoreFilter(lang="en", min_score=0.26311219),
            direct_language,
        ),
        (
            "perplexity_filter",
            ops.PerplexityFilter(lang="en", max_ppl=7376.81378),
            DJPerplexityFilter(lang="en", max_ppl=7376.81378),
            direct_perplexity,
        ),
    ]
    report = []
    for name, cedar_op, reference_op, direct in text_pairs:
        cedar_sample = copy.deepcopy(base)
        reference_sample = copy.deepcopy(base)
        cedar_decision = bool(cedar_op(cedar_sample))
        reference_decision, reference_sample = direct(
            reference_op, reference_sample
        )
        assert cedar_decision == reference_decision
        assert_stats_close(
            cedar_sample[ops.FIELDS_STATS],
            reference_sample[ops.FIELDS_STATS],
            name,
        )
        report.append(
            {
                "operator": name,
                "decision": cedar_decision,
                "stats": comparable(cedar_sample[ops.FIELDS_STATS]),
            }
        )

    adapters = [
        ops.video_aesthetics_filter(),
        ops.video_text_similarity_filter(),
        ops.video_motion_filter(),
        ops.video_nsfw_filter(),
        ops.video_watermark_filter(),
    ]
    for adapter in adapters:
        cedar_sample = copy.deepcopy(base)
        reference_sample = copy.deepcopy(base)
        reference = type(adapter)(
            adapter.operator_name, **adapter.operator_kwargs
        )._get_operator()
        reference_decision, reference_sample = direct_single(
            reference, reference_sample
        )
        cedar_decision = bool(adapter(cedar_sample))
        assert cedar_decision == reference_decision
        assert_stats_close(
            cedar_sample[ops.FIELDS_STATS],
            reference_sample[ops.FIELDS_STATS],
            adapter.operator_name,
        )
        report.append(
            {
                "operator": adapter.operator_name,
                "decision": cedar_decision,
                "stats": comparable(cedar_sample[ops.FIELDS_STATS]),
            }
        )

    output = Path(
        "outputs/chapter6_experiments/general_video_refine_setup/"
        "operator_equivalence.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

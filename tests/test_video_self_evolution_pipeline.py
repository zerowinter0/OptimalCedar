from pathlib import Path

from cedar.sources import IterSource

from evaluation.pipelines.general_video_refine import dj_operators as ops
from evaluation.pipelines.video_self_evolution.cedar_dataset import (
    VideoSelfEvolutionFeature,
)
from evaluation.pipelines.video_self_evolution.validate_recipe import validate


def test_video_self_evolution_preserves_five_filter_recipe():
    feature = VideoSelfEvolutionFeature("/tmp/videos")
    feature.apply(IterSource(["{}"]))
    tags = {
        pipe.tag for pipe in feature.logical_pipes.values() if not pipe.is_source()
    }
    assert len(tags) == 8
    assert tags == {
        "parse",
        "video_root",
        "video_nsfw",
        "video_text_similarity",
        "video_motion",
        "video_aesthetics",
        "video_duration",
        "project_output",
    }


def test_video_self_evolution_arguments_match_hub_recipe():
    filters = [
        ops.sandbox_video_nsfw_filter(),
        ops.sandbox_video_text_similarity_filter(),
        ops.sandbox_video_motion_filter(),
        ops.sandbox_video_aesthetics_filter(),
        ops.sandbox_video_duration_filter(),
    ]
    assert [item.operator_name for item in filters] == [
        "video_nsfw_filter",
        "video_frames_text_similarity_filter",
        "video_motion_score_filter",
        "video_aesthetics_filter",
        "video_duration_filter",
    ]
    assert filters[0].operator_kwargs["max_score"] == 0.000195383
    assert filters[1].operator_kwargs["min_score"] == 0.306337
    assert filters[2].operator_kwargs["min_score"] == 3.0
    assert filters[2].operator_kwargs["max_score"] == 20.0
    assert filters[3].operator_kwargs["min_score"] == 0.418164
    assert filters[4].operator_kwargs == {
        "min_duration": 2.0,
        "max_duration": 100000.0,
        "any_or_all": "any",
    }


def test_video_self_evolution_matches_pinned_hub_recipe():
    observed = validate(
        Path(
            "data-juicer-hub/refined_recipes/video/"
            "data-juicer-sandbox-self-evolution.yaml"
        )
    )
    assert len(observed) == 5

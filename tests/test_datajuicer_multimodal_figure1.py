from cedar.pipes import PipeExecutionResource, PipeVariantType
from cedar.sources import IterSource

from evaluation.pipelines.datajuicer_multimodal_figure1.catalog import (
    CEDAR_PLAN_TARGET,
    JOINT_DP_PLAN_TARGET,
    MULTIMODAL_PIPELINES,
    SELECTED_PIPELINE,
)
from evaluation.pipelines.datajuicer_multimodal_figure1.cedar_dataset import (
    FIGURE1_TAGS,
    VideoCaptionRefineFeature,
)


def test_catalog_contains_diverse_multimodal_migrations():
    assert len(MULTIMODAL_PIPELINES) >= 7
    assert all(
        len(pipeline.modalities) >= 2
        for pipeline in MULTIMODAL_PIPELINES
    )
    assert all(
        len(pipeline.operators) <= 10
        for pipeline in MULTIMODAL_PIPELINES
    )
    assert all(pipeline.gpu_operators for pipeline in MULTIMODAL_PIPELINES)


def test_selected_pipeline_has_required_modalities_and_gpu_work():
    assert SELECTED_PIPELINE.modalities == ("video", "text")
    assert len(SELECTED_PIPELINE.operators) == 10
    assert {
        "VideoAesthetics",
        "VideoTextSimilarity",
        "VideoNSFW",
        "VideoWatermark",
    } == set(SELECTED_PIPELINE.gpu_operators)


def test_figure1_feature_preserves_backend_boundaries():
    feature = VideoCaptionRefineFeature("/tmp/videos")
    feature.apply(IterSource(["{}"]))
    by_tag = {pipe.tag: pipe for pipe in feature.logical_pipes.values()}

    assert set(FIGURE1_TAGS) <= set(by_tag)
    assert len(FIGURE1_TAGS) == 10
    for tag in (
        "video_aesthetics",
        "video_text_similarity",
        "video_nsfw",
        "video_watermark",
    ):
        assert by_tag[tag].execution_resource == PipeExecutionResource.CUDA

    for tag in ("video_motion", "video_duration"):
        pipe = by_tag[tag]
        assert pipe.pipe_spec.is_fusable is False
        assert pipe.pipe_spec.mutable_variants == [
            PipeVariantType.INPROCESS,
            PipeVariantType.SMP,
        ]


def test_target_plans_cover_each_operator_once():
    expected = set(SELECTED_PIPELINE.operators)
    for plan in (CEDAR_PLAN_TARGET, JOINT_DP_PLAN_TARGET):
        flattened = [
            operator
            for _, stage_operators in plan
            for operator in stage_operators
        ]
        assert len(flattened) == len(expected)
        assert set(flattened) == expected

    assert len(CEDAR_PLAN_TARGET) > len(JOINT_DP_PLAN_TARGET)
    cedar_gpu_stages = [
        stage for stage, _ in CEDAR_PLAN_TARGET if stage == "Ray-GPU"
    ]
    joint_gpu_stages = [
        stage for stage, _ in JOINT_DP_PLAN_TARGET if stage == "Ray-GPU"
    ]
    assert len(cedar_gpu_stages) == 2
    assert len(joint_gpu_stages) == 1

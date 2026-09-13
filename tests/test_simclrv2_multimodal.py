import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from cedar.client import DataSet
from cedar.pipes import PipeExecutionResource
from cedar.pipes.context import PipeVariantContextFactory, PipeVariantType
from cedar.compose.optimizer import PhysicalPlan, PipeDesc
from cedar.sources import LocalLineSource

from evaluation.pipelines.simclrv2_multimodal.cedar_dataset import (
    CaptionThreshold,
    ExtractImagePath,
    MultimodalSimCLRV2Feature,
    PerImageStandardize,
    ToFloatImage,
    displayed_signature,
)
from evaluation.pipelines.simclrv2_multimodal.operator_scaling import (
    DEFAULT_IMAGE_SIDES,
    DEFAULT_TEXT_WORDS,
    OPERATOR_NAMES,
    build_scaling_points,
    summarize_rows,
)
from evaluation.pipelines.simclrv2_multimodal.modality_scaling import (
    IMAGE_SWEEP_SIDES,
    TEXT_SWEEP_WORDS,
    build_modality_scaling_points,
    summarize_modality_rows,
)
from evaluation.pipelines.simclrv2_multimodal.plot_operator_scaling_panel import (
    PANEL_CELL_ID,
    embed_panel,
    image_pixel_positions,
    load_modality_durations,
    load_normalized_rates,
    normalize_modality_rates,
)
from cedar.compose.simple_dp_ray_candidate_optimizer import (
    move_simple_dp_augmentation_fusion_to_ray,
)


def test_local_clip_w1_workflow_uses_fresh_profile_and_four_plans() -> None:
    script = Path(
        "evaluation/pipelines/simclrv2_multimodal/"
        "run_local_clip_w1_workflow.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert "simclrv2_multimodal_local_clip_w1_20260909" in text
    assert "--run_profiling" in text
    assert "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS=1" in text
    assert "--fixed_local_workers_ablation 1" in text
    assert "--num_repeats 3" in text
    assert (
        "--optimizers optimizer simple_dp_optimizer dp_optimizer "
        "simple_dp_ray_candidate_optimizer"
    ) in text


def test_fixed_prefix_workflow_restores_formal_w8_single_round() -> None:
    script = Path(
        "evaluation/pipelines/simclrv2_multimodal/"
        "run_fixed_prefix_w8_workflow.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert "CEDAR_PROFILE_RAY_ACTORS=1" in text
    assert "CEDAR_PROFILE_SMP_PROCS=1" in text
    assert "--fixed_local_workers_ablation 8" in text
    assert "--optimizers optimizer simple_dp_optimizer dp_optimizer" in text
    assert "--num_repeats 1" in text


def test_ray_candidate_changes_only_blur_jitter_flip_fusion_backend() -> None:
    logical_pipes = {
        2: SimpleNamespace(tag="gaussian_blur"),
        3: SimpleNamespace(tag="color_jitter"),
        4: SimpleNamespace(tag="random_flip"),
        5: SimpleNamespace(tag="random_crop"),
    }
    plan = PhysicalPlan(
        graph={5: {12}, 12: {1}, 1: set()},
        pipe_descs={
            1: PipeDesc(
                "finalize",
                PipeVariantType.INPROCESS,
                PipeVariantContextFactory.create_context(
                    PipeVariantType.INPROCESS
                ),
            ),
            5: PipeDesc(
                "crop",
                PipeVariantType.INPROCESS,
                PipeVariantContextFactory.create_context(
                    PipeVariantType.INPROCESS
                ),
            ),
            12: PipeDesc(
                "FusedPipe",
                PipeVariantType.INPROCESS,
                PipeVariantContextFactory.create_context(
                    PipeVariantType.INPROCESS
                ),
                fused_pipes=[2, 3, 4],
            ),
        },
    )
    original_graph = {pid: set(next_pids) for pid, next_pids in plan.graph.items()}
    original_members = list(plan.pipe_descs[12].fused_pipes)

    changed_pid = move_simple_dp_augmentation_fusion_to_ray(
        plan, logical_pipes
    )

    assert changed_pid == 12
    assert plan.graph == original_graph
    assert plan.pipe_descs[12].fused_pipes == original_members
    assert plan.pipe_descs[12].variant_type == PipeVariantType.RAY
    assert plan.pipe_descs[12].execution_resource == PipeExecutionResource.CPU
    assert plan.pipe_descs[12].variant_ctx.serialize() == {
        "variant_type": "RAY",
        "n_actors": 1,
        "max_inflight": 100,
        "max_prefetch": 100,
        "use_threads": True,
        "submit_batch_size": 30,
        "num_gpus": 0.0,
    }
    assert plan.pipe_descs[5].variant_type == PipeVariantType.INPROCESS


def test_modified_workload_has_two_text_and_four_image_operators(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(
        json.dumps({"caption": "an image", "image_path": "one.jpg"}) + "\n",
        encoding="utf-8",
    )
    feature = MultimodalSimCLRV2Feature(CaptionThreshold(0.25), tmp_path, 1)
    feature.apply(LocalLineSource(str(source)))

    assert displayed_signature(feature) == {
        "tags": [
            "caption_normalize",
            "image_text_filter",
            "random_crop",
            "random_flip",
            "color_jitter",
            "gaussian_blur",
        ],
        "text_tags": ["caption_normalize"],
        "multimodal_tags": ["image_text_filter"],
        "image_tags": ["random_crop", "random_flip", "color_jitter", "gaussian_blur"],
        "dependencies": [
            ("caption_normalize", "image_text_filter"),
            ("image_text_filter", "random_crop"),
            ("random_crop", "random_flip"),
        ],
        "fixed_tags": ["caption_normalize", "image_text_filter"],
        "cuda_tags": ["image_text_filter"],
    }

    by_tag = {pipe.tag: pipe for pipe in feature.logical_pipes.values()}
    assert by_tag["caption_normalize"].pipe_spec.mutable_variants == [
        PipeVariantType.INPROCESS
    ]
    assert by_tag["image_text_filter"].execution_resource == (
        PipeExecutionResource.CUDA
    )
    assert by_tag["image_text_filter"].pipe_spec.mutable_variants == [
        PipeVariantType.INPROCESS
    ]
    assert by_tag["image_text_filter"].pipe_spec.is_fusable is False


def test_fixed_boundary_resolves_image_and_simclr_numeric_operators(tmp_path: Path) -> None:
    path = ExtractImagePath(tmp_path)({"image_path": "one.jpg", "caption": "unused"})
    assert path == str(tmp_path / "one.jpg")

    result = ToFloatImage()(torch.full((3, 2, 2), 255, dtype=torch.uint8))
    assert result.shape == (3, 2, 2)
    assert result.dtype == torch.float32
    assert torch.all(result == 1.0)

    standardized = PerImageStandardize()(torch.arange(12.0).reshape(3, 2, 2))
    assert abs(float(standardized.mean())) < 1e-6


def test_bookkeeping_boundaries_are_excluded_from_backend_search(tmp_path: Path) -> None:
    feature = MultimodalSimCLRV2Feature(
        CaptionThreshold(clip_min=0.25),
        tmp_path,
        batch_size=1,
    )
    source = tmp_path / "records.jsonl"
    source.write_text("", encoding="utf-8")
    feature.apply(LocalLineSource(str(source)))
    by_tag = {pipe.tag: pipe for pipe in feature.logical_pipes.values()}

    for tag in ("parse", "caption_normalize", "extract_image"):
        assert by_tag[tag].pipe_spec.mutable_variants == [PipeVariantType.INPROCESS]
        assert not by_tag[tag].is_mutable()

    mutable_tags = {
        "to_float",
        "random_crop",
        "random_flip",
        "color_jitter",
        "grayscale",
        "gaussian_blur",
        "image_normalize",
    }
    assert mutable_tags <= set(by_tag)
    assert all(by_tag[tag].is_mutable() for tag in mutable_tags)

    for tag in mutable_tags:
        assert PipeVariantType.RAY in by_tag[tag].pipe_spec.mutable_variants

    assert by_tag["image_text_filter"].pipe_spec.is_fusable is False

    # Layered profiling replays fixed legal inputs at each operator boundary,
    # so both image operators retain the same SMP choices as original SimCLR.
    for tag in ("random_crop", "random_flip"):
        variants = by_tag[tag].pipe_spec.mutable_variants
        assert PipeVariantType.SMP in variants
        assert PipeVariantType.RAY in variants


def test_profile_ray_excludes_fixed_local_cuda_stage(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(
        json.dumps({"caption": "an image", "image_path": "one.jpg"}) + "\n",
        encoding="utf-8",
    )
    feature = MultimodalSimCLRV2Feature(CaptionThreshold(0.25), tmp_path, 1)
    feature.apply(LocalLineSource(str(source)))
    gpu_pid = next(
        pid
        for pid, pipe in feature.logical_pipes.items()
        if pipe.tag == "image_text_filter"
    )

    observed_contexts = {}
    dataset = object.__new__(DataSet)

    def capture_profile(_name, _feature, _samples, mutation_dict):
        observed_contexts.update(mutation_dict)
        return {"throughput": 1.0}

    monkeypatch.setattr(dataset, "_profile_feature", capture_profile)
    dataset._profile_ray({}, feature, "feature", 1)

    assert gpu_pid not in observed_contexts


def test_clip_backend_constraint_is_encoded_in_pipe_spec(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text("", encoding="utf-8")
    feature = MultimodalSimCLRV2Feature(CaptionThreshold(0.25), tmp_path, 1)
    feature.apply(LocalLineSource(str(source)))
    clip_pipe = next(
        pipe
        for pipe in feature.logical_pipes.values()
        if pipe.tag == "image_text_filter"
    )

    assert clip_pipe.pipe_spec.mutable_variants == [PipeVariantType.INPROCESS]
    assert clip_pipe.pipe_spec.is_fusable is False
    assert clip_pipe.can_mutate_to(PipeVariantType.RAY) is False
    assert clip_pipe.is_fusable(PipeVariantType.RAY) is False

    # The workload does not repair illegal optimizer output after search; the
    # legality constraint belongs to the logical pipe specification above.
    clip_pid = clip_pipe.id
    plan = feature.optimizer.physical_plan
    plan.pipe_descs[clip_pid].variant_type = PipeVariantType.RAY
    feature._constrain_physical_plan(plan)
    assert plan.pipe_descs[clip_pid].variant_type == PipeVariantType.RAY


def test_operator_scaling_points_cover_the_six_displayed_operators() -> None:
    points = build_scaling_points()

    assert {point.operator for point in points} == set(OPERATOR_NAMES)
    assert len(points) == 6 * 4
    assert {
        point.text_words for point in points if point.operator == "normalize"
    } == set(DEFAULT_TEXT_WORDS)
    assert {
        point.image_side for point in points if point.operator == "random_crop"
    } == set(DEFAULT_IMAGE_SIDES)
    assert all(
        point.text_words is not None and point.image_side is not None
        for point in points
        if point.operator == "clip"
    )


def test_operator_scaling_summary_reports_median_rate() -> None:
    rows = [
        {
            "operator": "normalize",
            "work_scale": 1.0,
            "text_words": 64,
            "image_side": None,
            "trial": trial,
            "duration_ns_per_record": duration,
        }
        for trial, duration in enumerate((20.0, 10.0, 30.0))
    ]

    summary = summarize_rows(rows)

    assert summary == [
        {
            "operator": "normalize",
            "work_scale": 1.0,
            "text_words": 64,
            "image_side": None,
            "trials": 3,
            "median_ns_per_record": 20.0,
            "median_records_per_sec": 50_000_000.0,
            "q1_records_per_sec": 41_666_666.666666664,
            "q3_records_per_sec": 75_000_000.0,
        }
    ]


def test_operator_scaling_panel_normalizes_each_operator_at_one(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "operator,work_scale,median_records_per_sec,q1_records_per_sec,q3_records_per_sec\n"
        "normalize,0.25,40,36,44\n"
        "normalize,1.0,20,18,22\n"
        "normalize,4.0,10,9,11\n"
        "normalize,16.0,5,4.5,5.5\n"
        "clip,0.25,30,27,33\n"
        "clip,1.0,30,27,33\n"
        "clip,4.0,24,21,27\n"
        "clip,16.0,15,12,18\n",
        encoding="utf-8",
    )

    curves = load_normalized_rates(summary, required_operators=("normalize", "clip"))

    assert curves["normalize"][1.0] == (1.0, 0.9, 1.1)
    assert curves["normalize"][0.25] == (2.0, 1.8, 2.2)
    assert curves["clip"][1.0] == (1.0, 0.9, 1.1)


def test_operator_scaling_panel_can_normalize_rate_to_smallest_input(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "operator,work_scale,median_records_per_sec,q1_records_per_sec,q3_records_per_sec\n"
        "normalize,0.25,40,36,44\n"
        "normalize,1.0,20,18,22\n"
        "normalize,4.0,10,9,11\n"
        "normalize,16.0,5,4.5,5.5\n",
        encoding="utf-8",
    )

    curves = load_normalized_rates(
        summary, required_operators=("normalize",), baseline_scale=0.25
    )

    assert curves["normalize"][0.25] == (1.0, 0.9, 1.1)
    assert curves["normalize"][1.0] == (0.5, 0.45, 0.55)


def test_operator_scaling_panel_embeds_png_with_drawio_safe_mime(tmp_path: Path) -> None:
    drawio = tmp_path / "figure.drawio"
    drawio.write_text(
        f'<mxfile><diagram><mxGraphModel><root><mxCell id="{PANEL_CELL_ID}" '
        'value="" style="rounded=0;"><mxGeometry /></mxCell></root>'
        "</mxGraphModel></diagram></mxfile>",
        encoding="utf-8",
    )
    panel = tmp_path / "panel.png"
    panel.write_bytes(b"png-test-payload")

    embed_panel(drawio, panel)

    content = drawio.read_text(encoding="utf-8")
    assert "image=data:image/png%3Bbase64," in content


def test_modality_scaling_measures_every_operator_on_linear_grids() -> None:
    points = build_modality_scaling_points()

    assert TEXT_SWEEP_WORDS == (128, 256, 384, 512, 640, 768, 896, 1024)
    assert IMAGE_SWEEP_SIDES == (128, 256, 384, 512, 640, 768, 896, 1024)
    assert len(points) == 2 * len(OPERATOR_NAMES) * 8
    assert {
        point.operator for point in points if point.varied_dimension == "text"
    } == set(OPERATOR_NAMES)
    assert {
        point.operator for point in points if point.varied_dimension == "image"
    } == set(OPERATOR_NAMES)
    assert {
        point.text_words for point in points if point.varied_dimension == "text"
    } == set(TEXT_SWEEP_WORDS)
    assert {
        point.image_side for point in points if point.varied_dimension == "text"
    } == {224}
    assert {
        point.text_words for point in points if point.varied_dimension == "image"
    } == {128}
    assert {
        point.image_side for point in points if point.varied_dimension == "image"
    } == set(IMAGE_SWEEP_SIDES)


def test_modality_scaling_summary_keeps_dimension_and_reports_time_quantiles() -> None:
    rows = [
        {
            "operator": "clip",
            "varied_dimension": "text",
            "input_value": 128,
            "text_words": 128,
            "image_side": 224,
            "trial": trial,
            "duration_ns_per_record": duration,
        }
        for trial, duration in enumerate((30.0, 10.0, 20.0))
    ]

    summary = summarize_modality_rows(rows)

    assert len(summary) == 1
    assert summary[0] == pytest.approx(
        {
            "operator": "clip",
            "varied_dimension": "text",
            "input_value": 128,
            "text_words": 128,
            "image_side": 224,
            "trials": 3,
            "median_ms_per_record": 0.00002,
            "q1_ms_per_record": 0.000015,
            "q3_ms_per_record": 0.000025,
        }
    )


def test_modality_panel_loads_both_linear_sweeps_for_every_operator(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.csv"
    lines = [
        "operator,varied_dimension,input_value,median_ms_per_record,"
        "q1_ms_per_record,q3_ms_per_record"
    ]
    for operator, offset in (("normalize", 0), ("clip", 10)):
        for dimension in ("text", "image"):
            for value in range(128, 1025, 128):
                median = offset + value / 128
                lines.append(
                    f"{operator},{dimension},{value},{median},{median - 0.25},"
                    f"{median + 0.25}"
                )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    curves = load_modality_durations(
        summary, required_operators=("normalize", "clip")
    )

    assert tuple(curves["text"]["normalize"]) == TEXT_SWEEP_WORDS
    assert tuple(curves["image"]["clip"]) == IMAGE_SWEEP_SIDES
    assert curves["text"]["normalize"][128] == (1.0, 0.75, 1.25)
    assert curves["image"]["clip"][1024] == (18.0, 17.75, 18.25)


def test_modality_panel_normalizes_rate_to_each_operators_128_input() -> None:
    durations = {
        "text": {
            "normalize": {
                128: (2.0, 1.0, 4.0),
                256: (4.0, 2.0, 8.0),
            }
        },
        "image": {
            "normalize": {
                128: (5.0, 4.0, 10.0),
                256: (5.0, 2.5, 20.0),
            }
        },
    }

    rates = normalize_modality_rates(durations)

    assert rates["text"]["normalize"][128] == (1.0, 0.5, 2.0)
    assert rates["text"]["normalize"][256] == (0.5, 0.25, 1.0)
    assert rates["image"]["normalize"][256] == (1.0, 0.25, 2.0)


def test_image_axis_uses_square_pixel_counts() -> None:
    assert image_pixel_positions((128, 256, 1024)) == (
        128**2,
        256**2,
        1024**2,
    )

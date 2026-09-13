import csv
import json
from pathlib import Path

import pytest
import yaml

from evaluation.motivation_multimodal.artifacts import sha256_file
from evaluation.motivation_multimodal.plot_figure2 import (
    load_figure2_data,
    render_figure2,
)
from evaluation.motivation_multimodal.plot_figure5 import render_figure5


def _plan(path: Path, reverse: bool = False) -> None:
    ids = [6, 5, 4, 3, 2, 1]
    if reverse:
        ids = [4, 3, 6, 5, 2, 1]
    graph = {str(pipe_id): str(ids[index + 1]) if index + 1 < len(ids) else "" for index, pipe_id in enumerate(ids)}
    pipes = {
        str(pipe_id): {
            "name": "MapperPipe",
            "variant": "SMP" if pipe_id >= 4 else "RAY",
            "variant_ctx": {"n_procs": 2} if pipe_id >= 4 else {"n_actors": 1},
            "execution_resource": "cpu" if pipe_id >= 4 else "cuda",
        }
        for pipe_id in ids
    }
    path.write_text(
        yaml.safe_dump(
            {"physical_plan": {"graph": graph, "pipes": pipes, "n_local_workers": 8}}
        ),
        encoding="utf-8",
    )


def _formal_artifact(root: Path, repetitions: int = 3) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    staged = root / "staged.yml"
    joint = root / "joint.yml"
    _plan(staged)
    _plan(joint, reverse=True)
    runs = {
        name: [
            {"seconds": base + index / 10, "record_ids": ["a", "b"]}
            for index in range(repetitions)
        ]
        for name, base in (("staged", 10.0), ("joint", 8.0))
    }
    payload = {
        "status": "success",
        "formal_protocol": True,
        "records": 3000,
        "equivalent": True,
        "plans": {
            "staged": {
                "path": str(staged),
                "sha256": sha256_file(staged),
                "operator_ids": {
                    "normalize": 6,
                    "perplexity": 5,
                    "safety": 4,
                    "aesthetic": 3,
                    "clip": 2,
                    "blip": 1,
                },
            },
            "joint": {
                "path": str(joint),
                "sha256": sha256_file(joint),
                "operator_ids": {
                    "normalize": 6,
                    "perplexity": 5,
                    "safety": 4,
                    "aesthetic": 3,
                    "clip": 2,
                    "blip": 1,
                },
            },
        },
        "executions": runs,
        "cedar_costs": {"staged": 0.8, "joint": 1.0},
    }
    summary = root / "formal_summary.json"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    return summary


def test_plot_rejects_missing_repetition(tmp_path: Path) -> None:
    _formal_artifact(tmp_path, repetitions=2)

    with pytest.raises(ValueError, match="three formal repetitions"):
        load_figure2_data(tmp_path)


def test_plot_rejects_missing_cedar_plan_costs(tmp_path: Path) -> None:
    summary = _formal_artifact(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload.pop("cedar_costs")
    summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Cedar costs"):
        load_figure2_data(tmp_path)


def test_figure2_manifest_hashes_every_input(tmp_path: Path) -> None:
    _formal_artifact(tmp_path / "formal")

    rendered = render_figure2(tmp_path / "formal", tmp_path / "figures")
    manifest = json.loads(rendered.manifest.read_text(encoding="utf-8"))

    assert rendered.pdf.is_file()
    assert rendered.png.is_file()
    assert all(len(item["sha256"]) == 64 for item in manifest["inputs"])


def test_figure5_manifest_hashes_summary_csv(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    fields = [
        "operator",
        "text_tokens",
        "image_side",
        "trials",
        "median_ns",
        "q1_ns",
        "q3_ns",
    ]
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for operator in ("normalize", "perplexity"):
            for tokens in (16, 32):
                writer.writerow(dict(zip(fields, (operator, tokens, "", 7, 100, 90, 110))))
        for operator in ("safety", "aesthetic"):
            for side in (128, 256):
                writer.writerow(dict(zip(fields, (operator, "", side, 7, 200, 180, 220))))
        for operator in ("clip", "blip"):
            for tokens in (16, 32):
                for side in (128, 256):
                    writer.writerow(dict(zip(fields, (operator, tokens, side, 7, 300, 280, 320))))

    rendered = render_figure5(source, tmp_path / "figures")
    manifest = json.loads(rendered.manifest.read_text(encoding="utf-8"))

    assert rendered.pdf.is_file()
    assert manifest["inputs"] == [
        {"path": str(source), "sha256": sha256_file(source)}
    ]

import json

from evaluation.chapter6_experiments import (
    plot_paper_cross_system_speedup as plotter,
)


def _write_round(root, workload, system, round_number, samples, seconds):
    result = (
        root
        / "results"
        / workload
        / f"round{round_number}__{system}.json"
    )
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(
        json.dumps(
            {
                "num_samples": samples,
                "measured_time_sec": seconds,
            }
        ),
        encoding="utf-8",
    )
    status = (
        root
        / "status"
        / workload
        / f"round{round_number}__{system}.tsv"
    )
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text("success\t\n", encoding="utf-8")


def test_aggregate_system_requires_three_exact_cardinality_rounds(tmp_path):
    for round_number, seconds in enumerate((10.0, 11.0, 12.0), start=1):
        _write_round(
            tmp_path,
            "coco",
            "pytorch",
            round_number,
            50_000,
            seconds,
        )

    result = plotter.aggregate_system(
        tmp_path,
        "coco",
        "pytorch",
        50_000,
    )

    assert result["status"] == "success"
    assert result["repeats"] == 3
    assert result["mean_execution_sec"] == 11.0


def test_aggregate_system_preserves_formal_unsupported_outcome(tmp_path):
    status_dir = tmp_path / "status" / "pile_europarl"
    status_dir.mkdir(parents=True)
    for round_number in range(1, 4):
        (status_dir / f"round{round_number}__plumber.tsv").write_text(
            "unsupported\topaque callback\n",
            encoding="utf-8",
        )

    result = plotter.aggregate_system(
        tmp_path,
        "pile_europarl",
        "plumber",
        2_500,
    )

    assert result["status"] == "unsupported"
    assert result["mean_execution_sec"] is None
    assert result["reasons"] == ["opaque callback"]

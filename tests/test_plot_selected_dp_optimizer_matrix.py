import json

import pytest

from evaluation.chapter6_experiments.plot_selected_dp_optimizer_matrix import (
    OPTIMIZERS,
    WORKLOADS,
    export_matrix,
    load_matrix,
    validate_selection,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_workloads_are_sorted_by_operator_count():
    assert [item[2] for item in WORKLOADS] == sorted(item[2] for item in WORKLOADS)


def test_load_matrix_distinguishes_success_and_timeouts(tmp_path):
    workload = ("sample", "Sample", 3)
    root = tmp_path / "sample"
    root.mkdir()
    (root / "metadata.txt").write_text("samples=10\n", encoding="utf-8")

    _write_json(
        root / "results/round1__dj_optimizer.json",
        {"epoch_run_times": [4.0], "epoch_num_samples": [10]},
    )
    _write_json(
        root / "warmup_results/plan_only__dj_optimizer.json",
        {"runs": [{"setup_time_sec": 2.0}]},
    )
    _write_json(
        root / "results/round1__pecan_optimizer.timeout.json",
        {"reason": "unified_task_timeout_during_execution", "task_timeout_sec": 3600},
    )
    _write_json(
        root / "warmup_results/plan_only__pecan_optimizer.json",
        {"runs": [{"setup_time_sec": 3.0}]},
    )
    _write_json(
        root / "plans/simple_dp_optimizer.unavailable.json",
        {"reason": "plan_generation_timeout", "timeout_sec": 360},
    )
    _write_json(
        root / "results/round1__dp_optimizer.json",
        {"epoch_run_times": [1.0], "epoch_num_samples": [10]},
    )
    _write_json(
        root / "warmup_results/plan_only__dp_optimizer.json",
        {"runs": [{"setup_time_sec": 5.0}]},
    )

    matrix = load_matrix(tmp_path, workloads=[workload])
    items = matrix["sample"]["optimizers"]

    assert items["dj_optimizer"]["status"] == "success"
    assert items["dj_optimizer"]["execution_sec"] == 4.0
    assert items["pecan_optimizer"]["status"] == "execution_timeout"
    assert items["pecan_optimizer"]["setup_sec"] == 3.0
    assert items["simple_dp_optimizer"]["status"] == "plan_timeout"
    assert items["simple_dp_optimizer"]["setup_sec"] == 360.0
    assert items["dp_optimizer"]["execution_sec"] == 1.0
    assert items["dp_optimizer"]["result_source"].endswith(
        "round1__dp_optimizer.json"
    )


def test_export_records_source_paths(tmp_path):
    matrix = {
        "sample": {
            "label": "Sample",
            "operator_count": 3,
            "samples": 10,
            "optimizers": {
                optimizer: {
                    "label": label,
                    "status": "success",
                    "execution_sec": 1.0,
                    "setup_sec": 2.0,
                    "result_source": f"sample/results/{optimizer}.json",
                    "setup_source": f"sample/setup/{optimizer}.json",
                }
                for optimizer, label, _color, _hatch in OPTIMIZERS
            },
        }
    }

    export_matrix(matrix, tmp_path)

    exported = json.loads((tmp_path / "selected_dp_optimizer_data.json").read_text())
    assert exported["sample"]["optimizers"]["dp_optimizer"][
        "result_source"
    ].endswith("dp_optimizer.json")
    tsv = (tmp_path / "selected_dp_optimizer_data.tsv").read_text()
    assert "result_source" in tsv
    assert "setup_source" in tsv


def test_selection_requires_a_dp_method_to_beat_both_heuristics():
    def item(status, execution_sec):
        return {"status": status, "execution_sec": execution_sec}

    valid = {
        "sample": {
            "optimizers": {
                "dj_optimizer": item("success", 4.0),
                "pecan_optimizer": item("success", 3.0),
                "simple_dp_optimizer": item("success", 2.0),
                "dp_optimizer": item("success", 5.0),
            }
        }
    }
    validate_selection(valid)

    invalid = {
        "sample": {
            "optimizers": {
                "dj_optimizer": item("success", 2.0),
                "pecan_optimizer": item("success", 3.0),
                "simple_dp_optimizer": item("success", 2.5),
                "dp_optimizer": item("success", 4.0),
            }
        }
    }
    with pytest.raises(ValueError, match="selection rule"):
        validate_selection(invalid)

from evaluation.pipelines.target_pipeline.benchmark_scaling import (
    calibration_calls, complete_cells,
)


def test_skip_only_pilot_cannot_create_unbounded_random_work():
    assert calibration_calls(0.000001, True) == 32
    assert calibration_calls(0.000001, False) == 8192
    assert calibration_calls(0.3, True) == 4
    for seconds in (0.000001, 0.001, 0.01, 0.1):
        assert calibration_calls(seconds, True) % 4 == 0


def test_resume_discards_entire_partial_cell():
    complete = [
        dict(operator_index="29", point="7", backend=backend,
             round=str(repeat), calls="4")
        for backend in ("local", "smp", "ray") for repeat in (1, 2, 3)
    ]
    partial = [dict(operator_index="29", point="8", backend="local",
                    round="1", calls="8192")]
    kept, keys = complete_cells(complete + partial)
    assert kept == complete
    assert keys == {(29, 7)}


def test_duplicate_rows_do_not_count_as_complete():
    rows = [dict(operator_index="1", point="0", backend="local",
                 round="1", calls="4")] * 9
    assert complete_cells(rows) == ([], set())

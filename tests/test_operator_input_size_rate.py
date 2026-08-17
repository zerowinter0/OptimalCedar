from evaluation.chapter6_experiments.benchmark_operator_input_size_rate import (
    LogSizeReservoir,
    build_stages,
)


def test_log_size_reservoir_selects_ordered_real_snapshots():
    reservoir = LogSizeReservoir(per_bucket=2, seed=17)
    values = ["x" * size for size in (8, 16, 32, 64, 128, 256, 512)]
    for index, value in enumerate(values):
        reservoir.offer(value, index)

    selected = reservoir.select(max_points=4)

    assert 3 <= len(selected) <= 4
    assert [item.input_bytes for item in selected] == sorted(
        item.input_bytes for item in selected
    )
    assert all(item.snapshot for item in selected)


def test_stackexchange_benchmark_uses_real_order_and_both_scalings(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text("{}\n")

    stages = build_stages(dataset)

    assert len(stages) == 19
    assert [stage.position for stage in stages] == list(range(1, 20))
    assert stages[0].tag == "parse"
    assert stages[-1].tag == "extract_text"
    assert {stage.scaling for stage in stages} == {"per_data", "per_record"}
    assert sum(stage.scaling == "per_record" for stage in stages) == 3

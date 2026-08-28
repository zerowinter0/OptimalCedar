import json
import sys

from evaluation.chapter6_experiments.verify_runtime_floor import main


def test_timeout_records_are_not_parsed_as_successes(tmp_path, monkeypatch):
    results = tmp_path / "workload" / "results"
    results.mkdir(parents=True)
    for repeat, runtime in enumerate((1801.0, 1802.0, 1803.0), start=1):
        (results / f"round{repeat}__optimizer.json").write_text(
            json.dumps({"epoch_run_times": [runtime]}), encoding="utf-8"
        )
    (results / "round1__timed_out.timeout.json").write_text(
        json.dumps({"status": "timeout"}), encoding="utf-8"
    )
    output = tmp_path / "validation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_runtime_floor.py",
            "--matrix-root",
            str(tmp_path),
            "--workloads",
            "workload",
            "--required-workloads",
            "workload",
            "--minimum-seconds",
            "1800",
            "--required-rounds",
            "3",
            "--output",
            str(output),
        ],
    )
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["workload"]["passed"] is True
    assert report["workload"]["slowest_available_optimizer"] == "optimizer"

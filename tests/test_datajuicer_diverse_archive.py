import json

from evaluation.chapter6_experiments.archive_datajuicer_diverse_results import (
    _copy_json_without_repo_absolute_paths,
)


def test_source_metadata_archive_removes_repo_absolute_paths(tmp_path):
    source = tmp_path / "metadata.json"
    source.write_text(
        json.dumps(
            {
                "archive": str(tmp_path / "datasets/archive.zip"),
                "upstream": "https://example.com/data",
                "revision": "abc123",
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "archive/source_metadata/data.json"

    _copy_json_without_repo_absolute_paths(source, destination, tmp_path)

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "archive": "datasets/archive.zip",
        "upstream": "https://example.com/data",
        "revision": "abc123",
    }

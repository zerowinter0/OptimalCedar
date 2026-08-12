import json

import pytest

from evaluation.chapter6_experiments.audit_datajuicer_candidate_scale import (
    _prefix_stats,
    _video_prefix_files,
)


def test_prefix_stats_uses_exact_source_order_bytes(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_bytes(b'{"x": 1}\n{"long": 200}\n{"x": 3}\n')

    assert _prefix_stats(source, 2) == (2, 23)
    with pytest.raises(RuntimeError, match="source exhausted"):
        _prefix_stats(source, 4)


def test_video_prefix_requires_distinct_existing_files(tmp_path):
    video_root = tmp_path / "videos"
    (video_root / "video").mkdir(parents=True)
    (video_root / "video/one.mp4").write_bytes(b"123")
    (video_root / "video/two.mp4").write_bytes(b"12345")
    source = tmp_path / "records.jsonl"
    source.write_text(
        json.dumps({"videos": ["video/one.mp4"]})
        + "\n"
        + json.dumps({"videos": ["video/two.mp4"]})
        + "\n",
        encoding="utf-8",
    )

    assert _video_prefix_files(source, 2, video_root) == (2, 8)

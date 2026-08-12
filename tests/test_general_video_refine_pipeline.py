import copy
import json
from pathlib import Path

from evaluation.pipelines.general_video_refine import dj_operators as ops


def test_parse_and_video_root(tmp_path: Path):
    line = json.dumps(
        {
            "video_id": "video0",
            "sentence_id": 7,
            "videos": ["nested/video0.mp4"],
            "text": "<__dj__video> a test caption",
        }
    )
    sample = ops.parse_json_line(line)
    result = ops.SetVideoRootMapper(tmp_path)(sample)
    assert result[ops.TEXT_KEY] == "<__dj__video> a test caption"
    assert result[ops.VIDEO_KEY] == [
        str((tmp_path / "nested/video0.mp4").resolve())
    ]
    assert result[ops.FIELDS_STATS] == {}


def test_video_adapter_is_lazily_serializable(monkeypatch):
    calls = []

    class FakeFilter:
        def __init__(self, threshold):
            self.threshold = threshold

        def compute_stats_single(self, sample, rank=None, context=False):
            calls.append((rank, context))
            sample[ops.FIELDS_STATS]["score"] = 0.75
            return sample

        def process_single(self, sample):
            return sample[ops.FIELDS_STATS]["score"] >= self.threshold

    monkeypatch.setattr(ops, "_operator_class", lambda _: FakeFilter)
    predicate = ops.DataJuicerVideoFilter("fake", threshold=0.5)
    sample = {"text": "x", "videos": [], ops.FIELDS_STATS: {}}
    assert predicate(sample)
    assert calls == [(0, False)]
    cloned = copy.deepcopy(predicate)
    assert cloned._operator is None


def test_video_adapter_supports_operator_without_rank_argument(monkeypatch):
    calls = []

    class FakeFilter:
        def compute_stats_single(self, sample, context=False):
            calls.append(context)
            return sample

        def process_single(self, sample):
            return True

    monkeypatch.setattr(ops, "_operator_class", lambda _: FakeFilter)
    assert ops.DataJuicerVideoFilter("fake")(
        {"text": "x", "videos": [], ops.FIELDS_STATS: {}}
    )
    assert calls == [False]


def test_project_output_removes_internal_stats():
    sample = {
        "video_id": "video1",
        "sentence_id": 3,
        "videos": ["video1.mp4"],
        "text": "caption",
        ops.FIELDS_STATS: {"score": 1.0},
    }
    assert ops.project_output(sample) == {
        "video_id": "video1",
        "sentence_id": 3,
        "videos": ["video1.mp4"],
        "text": "caption",
    }

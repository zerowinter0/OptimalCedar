from pathlib import Path

import yaml

from evaluation.baselines.registry import (
    SYSTEMS,
    WORKLOADS,
    get_entry,
    iter_entries,
)
from evaluation.pipelines.datajuicer_workloads import build_stages


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_registry_covers_the_complete_sixteen_by_six_matrix():
    entries = list(iter_entries())
    assert len(entries) == len(WORKLOADS) * len(SYSTEMS) == 96
    assert {
        (entry.system, entry.workload) for entry in entries
    } == {
        (system, workload)
        for workload in WORKLOADS
        for system in SYSTEMS
    }


def test_every_supported_entry_has_a_real_implementation():
    for entry in iter_entries():
        if entry.status == "supported":
            assert entry.implementation
            assert (REPO_ROOT / entry.implementation).is_file(), entry
        else:
            assert entry.status == "unsupported"
            assert entry.reason


def test_datajuicer_support_is_scoped_to_native_reference_recipes():
    supported = {
        workload
        for workload in WORKLOADS
        if get_entry("datajuicer", workload).status == "supported"
    }
    assert supported == {
        "llava_pretrain",
        "redpajama_c4",
        "stackexchange",
    }


def test_datajuicer_recipe_operator_order_matches_shared_workloads():
    ignored_adapter_stages = {
        "parse",
        "image_root",
        "sync_text",
        "extract_text",
    }
    for workload in (
        "llava_pretrain",
        "redpajama_c4",
        "stackexchange",
    ):
        entry = get_entry("datajuicer", workload)
        config = yaml.safe_load(
            (REPO_ROOT / entry.implementation).read_text(encoding="utf-8")
        )
        recipe_ops = [next(iter(item)) for item in config["process"]]
        adapter_ops = [
            stage.name
            for stage in build_stages(workload)
            if stage.name not in ignored_adapter_stages
        ]
        op_name = {
            "clean_email": "clean_email_mapper",
            "clean_links": "clean_links_mapper",
            "fix_unicode": "fix_unicode_mapper",
            "normalize_punct": "punctuation_normalization_mapper",
            "punctuation_norm": "punctuation_normalization_mapper",
            "normalize_space": "whitespace_normalization_mapper",
            "avg_line_len": "average_line_length_filter",
            "char_repeat": "character_repetition_filter",
            "lang_id": "language_id_score_filter",
            "language_id": "language_id_score_filter",
            "max_line_len": "maximum_line_length_filter",
            "special_chars": "special_characters_filter",
            "word_repeat": "word_repetition_filter",
        }
        assert recipe_ops == [
            op_name.get(name, f"{name}_filter") for name in adapter_ops
        ]


def test_opaque_fm_tensorflow_pipelines_are_not_claimed_by_graph_optimizers():
    for workload in (
        "llava_pretrain",
        "redpajama_c4",
        "stackexchange",
        "pile_europarl",
        "redpajama_code",
        "pile_hackernews",
        "pile_pubmed_abstracts",
        "pile_freelaw",
        "pile_uspto_backgrounds",
    ):
        assert get_entry("tensorflow", workload).backend == "tf_py_function"
        assert get_entry("plumber", workload).status == "unsupported"
        assert get_entry("fastflow", workload).status == "unsupported"


def test_redpajama_tensorflow_w8_python_udf_is_marked_infeasible():
    entry = get_entry("tensorflow", "redpajama_c4")
    assert entry.status == "unsupported"
    assert entry.backend == "tf_py_function"
    assert "W=8" in entry.reason

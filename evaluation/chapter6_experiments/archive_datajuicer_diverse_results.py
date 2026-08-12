#!/usr/bin/env python3
"""Create the immutable public archive for the diverse-workload study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


ALLOWED_SUFFIXES = {".json", ".yaml", ".yml", ".txt", ".md", ".tsv"}
EXCLUDED_PARTS = {"logs", "cache", "diagnostics", "__pycache__"}
EXPECTED_OPTIMIZERS = {
    "optimizer",
    "dj_optimizer",
    "dp_cedar_optimizer",
    "dp_optimizer",
    "dp_two_stage_optimizer",
    "pecan_optimizer",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _relativize_json_paths(value, repo: Path):
    if isinstance(value, dict):
        return {
            key: _relativize_json_paths(item, repo)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_relativize_json_paths(item, repo) for item in value]
    if isinstance(value, str):
        try:
            return str(Path(value).relative_to(repo))
        except ValueError:
            return value
    return value


def _copy_json_without_repo_absolute_paths(
    source: Path, destination: Path, repo: Path
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload = _relativize_json_paths(payload, repo)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _copy_evidence_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if EXCLUDED_PARTS.intersection(relative.parts):
            continue
        if path.name.endswith(".log") or path.suffix not in ALLOWED_SUFFIXES:
            continue
        _copy_file(path, destination / relative)


def _write_manifest(root: Path) -> None:
    rows = ["path\tsize_bytes\tsha256"]
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.tsv":
            rows.append(
                f"{path.relative_to(root)}\t{path.stat().st_size}\t{_sha256(path)}"
            )
    (root / "MANIFEST.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _verify_existing(archive: Path, final_selection: Path) -> None:
    manifest = archive / "MANIFEST.tsv"
    if not manifest.is_file():
        raise RuntimeError(f"archive exists without manifest: {archive}")
    for line in manifest.read_text(encoding="utf-8").splitlines()[1:]:
        relative, raw_size, expected_hash = line.split("\t")
        path = archive / relative
        if not path.is_file() or path.stat().st_size != int(raw_size):
            raise RuntimeError(f"archive file missing or size mismatch: {path}")
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"archive checksum mismatch: {path}")
    if _sha256(archive / "final_selection.json") != _sha256(final_selection):
        raise RuntimeError("archive final_selection differs from current final report")


def _validate_report(report: dict) -> None:
    protocol = report.get("protocol", {})
    expected_protocol = {
        "local_workers": 8,
        "cpu_budget": 64,
        "repetitions": 3,
        "optimizer_plan_limit_sec": 3600,
        "execution_limit_sec": 3600,
        "speedup_threshold": 1.20,
        "maximum_non_win_fraction": 0.40,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise RuntimeError(
                f"unexpected formal protocol {key}: "
                f"{protocol.get(key)!r} != {expected!r}"
            )
    selected = report.get("selected_workloads", [])
    if len(selected) != len(set(selected)) or not 6 <= len(selected) <= 8:
        raise RuntimeError("final selection must contain six to eight unique workloads")
    ledger = report.get("ledger", {})
    wins = 0
    for workload in selected:
        item = ledger.get(workload, {})
        if item.get("valid") is not True:
            raise RuntimeError(f"selected workload is invalid: {workload}")
        runs = item.get("runs", {})
        if set(runs) != EXPECTED_OPTIMIZERS:
            raise RuntimeError(
                f"selected workload lacks exact six-optimizer evidence: {workload}"
            )
        for optimizer, run in runs.items():
            valid = (
                run.get("valid") is True
                and run.get("within_execution_limit", True) is True
                and len(run.get("execution_times_sec", [])) == 3
                and all(
                    0 < float(value) <= 3600
                    for value in run.get("execution_times_sec", [])
                )
            )
            unavailable = run.get("formally_unavailable") is True
            if not (valid or unavailable):
                raise RuntimeError(
                    f"incomplete formal run: {workload}/{optimizer}"
                )
        wins += item.get("dp_at_least_20pct_faster") is True
    non_wins = len(selected) - wins
    failure_fraction = non_wins / len(selected)
    if failure_fraction > 0.40:
        raise RuntimeError("selected workload non-win fraction exceeds 40%")
    summary = report.get("summary", {})
    expected_summary = {
        "selected_count": len(selected),
        "wins": wins,
        "non_wins": non_wins,
        "failure_fraction": failure_fraction,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"inconsistent final summary field: {key}")
    scenarios = summary.get("diversity", {}).get("scenarios", [])
    if len(set(scenarios)) < 6:
        raise RuntimeError("final selection covers fewer than six scenarios")


def create_archive(root: Path, archive: Path, repo: Path) -> None:
    final_selection = root / "final_selection.json"
    report = json.loads(final_selection.read_text(encoding="utf-8"))
    _validate_report(report)
    selected = report["selected_workloads"]
    if archive.exists():
        _verify_existing(archive, final_selection)
        print(f"verified existing archive: {archive}")
        return

    archive.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".datajuicer_diverse.", dir=archive.parent)
    )
    try:
        for name in (
            "final_selection.json",
            "final_selection.md",
            "recipe_classification.json",
            "recipe_classification.tsv",
            "data_scale.json",
        ):
            _copy_file(root / name, staging / name)
        _copy_file(
            repo / "evaluation/chapter6_experiments/DATA_JUICER_DIVERSE_WORKLOADS.md",
            staging / "PROTOCOL.md",
        )
        _copy_file(
            repo / "evaluation/chapter6_experiments/DATA_JUICER_RECIPE_AUDIT.md",
            staging / "RECIPE_AUDIT.md",
        )
        _copy_file(
            repo
            / "evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/data/data_pipeline_matrix.json",
            staging / "evidence/pile/data_pipeline_matrix.json",
        )

        source_metadata = {
            "redpajama_code.json": repo
            / "datasets/redpajama_code/redpajama-github-raw-50000.metadata.json",
            "redpajama_arxiv.json": repo
            / "datasets/redpajama_arxiv/redpajama-arxiv-raw-3gib.metadata.json",
            "alpaca_cot.json": repo
            / "datasets/alpaca_cot/alpaca-cot-en-cot-data.metadata.json",
            "video_self_evolution.json": repo
            / "datasets/general_video_refine/dataset_metadata.json",
        }
        for name, source in source_metadata.items():
            _copy_json_without_repo_absolute_paths(
                source, staging / "source_metadata" / name, repo
            )
        for name in (
            "datajuicer_diverse_runtime.pdf",
            "datajuicer_diverse_runtime.png",
            "datajuicer_diverse_runtime.svg",
            "datajuicer_diverse_runtime.tsv",
        ):
            _copy_file(root / "figures" / name, staging / "figures" / name)

        canonical_profiles = (
            repo
            / "evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/profiles"
        )
        for workload in selected:
            source = root / "profiles" / f"{workload}.yaml"
            if not source.is_file():
                source = canonical_profiles / f"{workload}.yaml"
            if workload == "general_video_refine" and not source.is_file():
                source = (
                    repo
                    / "outputs/chapter6_experiments/general_video_refine_formal/profiles/general_video_refine.yaml"
                )
            _copy_file(source, staging / "profiles" / source.name)

        evidence_roots = {
            "alpaca_cot": root / "alpaca_formal/alpaca_cot",
            "llava_pretrain": repo
            / "evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/data/standard_core/workloads/llava_pretrain",
            "redpajama_code": root
            / "code_formal/datajuicer_candidate_runs/code_additive_formal/redpajama_code",
            "redpajama_arxiv": root / "arxiv_formal/redpajama_arxiv",
            "video_self_evolution": root
            / "video_self_evolution_formal/video_self_evolution",
        }
        for workload in selected:
            if workload in evidence_roots:
                _copy_evidence_tree(
                    evidence_roots[workload], staging / "evidence" / workload
                )
        if "general_video_refine" in selected:
            _copy_evidence_tree(
                repo
                / "outputs/chapter6_experiments/general_video_refine_formal/matrix/general_video_refine",
                staging / "evidence/general_video_refine/competitors",
            )
            _copy_evidence_tree(
                repo
                / "outputs/chapter6_experiments/general_video_refine_cost_model_fix_formal/general_video_refine",
                staging / "evidence/general_video_refine/corrected_dp",
            )

        # Preserve every fresh screen, including candidates omitted from the
        # final subset. This prevents a public archive from hiding non-wins.
        screening_roots = {
            "alpaca_cot": root / "screening_matrix/alpaca_cot",
            "redpajama_code": root
            / "code_screening/datajuicer_candidate_runs/code_additive_screen/redpajama_code",
            "redpajama_arxiv_2500": root
            / "arxiv_scaled_screening/redpajama_arxiv",
            "redpajama_arxiv_20000": root
            / "screening_matrix/redpajama_arxiv",
            "video_self_evolution": root
            / "video_self_evolution_screening/video_self_evolution",
        }
        for name, source in screening_roots.items():
            if source.is_dir():
                _copy_evidence_tree(source, staging / "screening" / name)
        status = root / "video_self_evolution_status"
        if status.is_dir():
            _copy_evidence_tree(status, staging / "screening/video_self_evolution_status")
        _copy_evidence_tree(
            root / "cedar_60m_audit", staging / "cedar_60m_audit"
        )
        _copy_evidence_tree(
            root / "two_stage_modal_audit",
            staging / "two_stage_modal_audit",
        )

        readme = [
            "# Formal Data-Juicer diverse-workload artifacts",
            "",
            "This directory is generated only after every selected workload has complete W=8/CPU-64, three-repeat, 3,600-second evidence for all six optimizers and the reused Pile workloads pass the current 3,600-second original-Cedar and DP two-stage plan audit.",
            "",
            "Selected workloads: " + ", ".join(f"`{name}`" for name in selected) + ".",
            "",
            "`final_selection.json` is authoritative. `screening/` retains non-selected outcomes; `profiles/` and `evidence/` contain the exact inputs to the aggregate; `source_metadata/` records frozen dataset provenance, size, and content hashes; `MANIFEST.tsv` provides size and SHA-256 for every archived file.",
            "",
        ]
        (staging / "README.md").write_text("\n".join(readme), encoding="utf-8")
        _write_manifest(staging)
        os.replace(staging, archive)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"created archive: {archive}")


def main() -> None:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--root",
        type=Path,
        default=repo / "outputs/chapter6_experiments/datajuicer_diverse_workloads",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=repo
        / "evaluation/chapter6_experiments/formal_results/paper_artifacts/datajuicer_diverse",
    )
    args = parser.parse_args()
    create_archive(args.root.resolve(), args.archive.resolve(), repo)


if __name__ == "__main__":
    main()

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


def create_archive(root: Path, archive: Path, repo: Path) -> None:
    final_selection = root / "final_selection.json"
    report = json.loads(final_selection.read_text(encoding="utf-8"))
    selected = report["selected_workloads"]
    if not 6 <= len(selected) <= 8:
        raise RuntimeError("final selection does not contain six to eight workloads")
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

        readme = [
            "# Formal Data-Juicer diverse-workload artifacts",
            "",
            "This directory is generated only after every selected workload passes the W=8/CPU-64, three-repeat, 3,600-second execution audit and the reused Pile workloads pass the current 3,600-second original-Cedar plan audit.",
            "",
            "Selected workloads: " + ", ".join(f"`{name}`" for name in selected) + ".",
            "",
            "`final_selection.json` is authoritative. `screening/` retains non-selected outcomes; `profiles/` and `evidence/` contain the exact inputs to the aggregate; `MANIFEST.tsv` provides size and SHA-256 for every archived file.",
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

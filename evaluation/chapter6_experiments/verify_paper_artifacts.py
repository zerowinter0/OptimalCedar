#!/usr/bin/env python3
"""Validate or rewrite the canonical Chapter 6 artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = (
    Path(__file__).resolve().parent
    / "formal_results"
    / "paper_artifacts"
)
MANIFEST = ROOT / "MANIFEST.tsv"


def rows() -> list[tuple[str, int, str]]:
    result = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        result.append(
            (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_size,
                str(path.relative_to(ROOT)),
            )
        )
    return result


def render(values: list[tuple[str, int, str]]) -> str:
    return "sha256\tbytes\tpath\n" + "".join(
        f"{digest}\t{size}\t{path}\n" for digest, size, path in values
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    current = render(rows())
    if args.write:
        MANIFEST.write_text(current, encoding="utf-8")
        print(f"Wrote {MANIFEST}")
        return
    if not MANIFEST.exists() or MANIFEST.read_text(encoding="utf-8") != current:
        raise SystemExit("Artifact manifest is missing or stale; run with --write")
    print(f"Verified {len(rows())} canonical files")


if __name__ == "__main__":
    main()

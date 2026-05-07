#!/usr/bin/env python3
"""
将 OSCAR 英文子集流式导出为 JSONL，供 DataJuicer ``bloom-oscar.yaml`` 与 Cedar ``BloomOscarFeature`` 使用。

数据来源（二选一，与 ``--source`` 一致）：

1. **legacy**（与 BigScience ``main_filtering.py`` 默认一致）::

       load_dataset("oscar", "unshuffled_deduplicated_en", split="train", streaming=True)

   体量常为 TB 级，视 HF 快照而定；若该入口在本地已下线，请改用 oscar2201。

2. **oscar2201**（HF 数据卡 OSCAR-22.01，英文约 3.2TB 量级）::

       load_dataset("oscar-corpus/OSCAR-2201", "en", split="train", streaming=True, token=...)

   需在 HuggingFace 接受 OSCAR 使用条款，并设置环境变量 ``HF_TOKEN``（或 ``HUGGING_FACE_HUB_TOKEN``）。

官方指引：

- DataJuicer：``/data-juicer/configs/reproduced_bloom/README*.md`` → BLOOM/Oscar 预处理说明
- BigScience：https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/training/01b_oscar_cleaning_and_filtering

示例::

    export HF_TOKEN=hf_xxx   # 仅 oscar2201 必需；legacy 视 HF 策略而定
    python -m evaluation.pipelines.bloom_oscar.download_oscar_en_jsonl \\
        --source auto \\
        --output /data-juicer/datasets/oscar_en/bloom_raw.jsonl \\
        --max-docs 100000

全量下载勿设 ``--max-docs``；请预留 TB 级磁盘与稳定网络。

``--resume`` 按**已写入行数**从流开头跳过相同条数再追加；若上游曾丢弃空文本行，流偏移与行数可能不完全一致，仅作近似续传。大断点下重新跳过流前缀会很慢。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

DATA_JUICER_DEFAULT_OUT = (
    "/data-juicer/datasets/oscar_en/bloom_raw.jsonl"
)


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _row_to_text(row: Dict[str, Any]) -> str:
    for key in ("text", "content", "raw_content"):
        v = row.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


def _iter_legacy() -> Iterator[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(
        "oscar",
        "unshuffled_deduplicated_en",
        split="train",
        streaming=True,
    )
    yield from ds


def _iter_oscar2201(token: Optional[str]) -> Iterator[Dict[str, Any]]:
    from datasets import load_dataset

    extra: Dict[str, Any] = {}
    if token:
        extra["token"] = token
    ds = load_dataset(
        "oscar-corpus/OSCAR-2201",
        "en",
        split="train",
        streaming=True,
        **extra,
    )
    yield from ds


def _count_jsonl_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _open_output(path: str, resume: bool):
    mode = "a" if resume else "w"
    return open(path, mode, encoding="utf-8", buffering=1024 * 1024)


def export_jsonl(
    iterator: Iterator[Dict[str, Any]],
    output_path: str,
    *,
    max_docs: Optional[int],
    skip: int,
    flush_every: int,
) -> int:
    pathlib_parent = os.path.dirname(output_path)
    if pathlib_parent:
        os.makedirs(pathlib_parent, exist_ok=True)

    written = 0
    skipped = 0
    try:
        from tqdm import tqdm

        pbar = tqdm(unit="docs")
    except Exception:  # noqa: BLE001
        pbar = None

    with _open_output(output_path, resume=skip > 0) as out:
        for row in iterator:
            if skipped < skip:
                skipped += 1
                if pbar:
                    pbar.update(1)
                continue

            text = _row_to_text(row)
            if not text:
                if pbar:
                    pbar.update(1)
                continue

            rec = {"text": text}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if flush_every > 0 and written % flush_every == 0:
                out.flush()

            if pbar:
                pbar.update(1)
            if max_docs is not None and written >= max_docs:
                break

        out.flush()
    if pbar:
        pbar.close()

    return written


def _resolve_iterator(source: str, token: Optional[str]) -> tuple[str, Iterator[Dict[str, Any]]]:
    if source == "legacy":
        return "legacy", _iter_legacy()
    if source == "oscar2201":
        if not token:
            logger.warning(
                "未检测到 HF_TOKEN；OSCAR-2201 多为门禁数据集，失败时请 export HF_TOKEN=..."
            )
        return "oscar2201", _iter_oscar2201(token)
    # auto
    try:
        it = _iter_legacy()
        first = next(it)

        def chain() -> Iterator[Dict[str, Any]]:
            yield first
            yield from it

        return "legacy", chain()
    except Exception as e:  # noqa: BLE001
        logger.info("legacy oscar 不可用 (%s)，尝试 oscar-corpus/OSCAR-2201 …", e)
        return "oscar2201", _iter_oscar2201(token)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        "-o",
        default=DATA_JUICER_DEFAULT_OUT,
        help=f"输出 JSONL 路径（默认 {DATA_JUICER_DEFAULT_OUT}）",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "legacy", "oscar2201"),
        default="auto",
        help="数据来源：legacy=BLOOM 默认 HF 名；oscar2201=OSCAR-22.01 英文；auto 先 legacy 再 2201",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="最多写入多少条非空文档；不设则直到流耗尽（全量）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若输出文件已存在，先统计行数并从流中跳过相应条数再追加（慢但简单）",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=5000,
        help="每写入 N 条 flush 一次；0 表示仅结束时 flush",
    )
    parser.add_argument("--log-level", default="INFO", help="logging 级别")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    token = _hf_token()
    skip = 0
    if args.resume and os.path.isfile(args.output):
        skip = _count_jsonl_lines(args.output)
        logger.info("resume：已存在 %s 行，将从流跳过 %s 条", skip, skip)

    if args.source == "auto":
        label, iterator = _resolve_iterator("auto", token)
    elif args.source == "legacy":
        label, iterator = "legacy", _iter_legacy()
    else:
        label, iterator = "oscar2201", _iter_oscar2201(token)

    logger.info("使用数据源：%s → %s", label, args.output)

    try:
        n = export_jsonl(
            iterator,
            args.output,
            max_docs=args.max_docs,
            skip=skip,
            flush_every=args.flush_every,
        )
    except Exception:
        logger.exception("导出失败；若 oscar2201 门禁失败，请设置 HF_TOKEN 或使用 --source legacy")
        return 1

    logger.info("完成：本轮新写入 %s 条（输出 %s）", n, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
在外网服务器（例：xry@172.23.166.102）上运行：从 Hugging Face 流式拉取 OSCAR 英文，
按分片写成 JSONL，每写完一片即通过 ``rsync``（默认）或 ``scp`` 推到内网机目录。

典型拓扑::

    172.23.166.102 可访问外网 ──SSH/rsync──> 172.23.166.103 仅内网
    用户 xry                         用户 xieruiyang，目标 /home/xieruiyang/...

依赖（在 102 上安装）::

    pip install datasets huggingface_hub tqdm

认证::

    export HF_TOKEN=hf_...          # OSCAR-2201 门禁需要
    ssh 密钥已配置：102 -> 103 可无密码 ``ssh xieruiyang@172.23.166.103``

大文件说明：全量英文子集约 TB 级，请保证 102 本机 ``--staging-dir`` 磁盘足够；
每片传完后可用 ``--delete-local-after-push`` 释放 102 空间（仅保留已传至 103 的分片）。

用法示例::

    export HF_TOKEN=hf_xxx
    python fetch_oscar_scp_to_internal.py \\
        --remote-host 172.23.166.103 \\
        --remote-user xieruiyang \\
        --remote-subdir oscar_en_jsonl \\
        --staging-dir /data/xry/oscar_staging \\
        --source oscar2201 \\
        --chunk-lines 200000

单文件模式（不推荐 TB 级）::

    python fetch_oscar_scp_to_internal.py --single-file --output-jsonl /data/xry/all.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from typing import Dict, Iterator, Optional, TextIO, Tuple

logger = logging.getLogger(__name__)


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


def _resolve_iterator(
    source: str, token: Optional[str]
) -> Tuple[str, Iterator[Dict[str, Any]]]:
    if source == "legacy":
        return "legacy", _iter_legacy()
    if source == "oscar2201":
        if not token:
            logger.warning("未设置 HF_TOKEN 时 oscar2201 可能因门禁失败。")
        return "oscar2201", _iter_oscar2201(token)
    try:
        it = _iter_legacy()
        first = next(it)

        def chain() -> Iterator[Dict[str, Any]]:
            yield first
            yield from it

        return "legacy", chain()
    except Exception as e:  # noqa: BLE001
        logger.info("legacy 不可用 (%s)，改用 oscar-corpus/OSCAR-2201", e)
        return "oscar2201", _iter_oscar2201(token)


def _ssh_base(remote_user: str, remote_host: str, ssh_port: int, identity_file: Optional[str]) -> list[str]:
    cmd = ["ssh", "-p", str(ssh_port)]
    if identity_file:
        cmd += ["-i", identity_file]
    cmd.append(f"{remote_user}@{remote_host}")
    return cmd


def _remote_mkdir(
    remote_user: str,
    remote_host: str,
    remote_dir: str,
    ssh_port: int,
    identity_file: Optional[str],
    dry_run: bool,
) -> None:
    inner = f"mkdir -p {shlex.quote(remote_dir)}"
    cmd = _ssh_base(remote_user, remote_host, ssh_port, identity_file) + [inner]
    logger.info("执行: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _push_file(
    local_path: str,
    remote_user: str,
    remote_host: str,
    remote_dir: str,
    *,
    use_rsync: bool,
    ssh_port: int,
    identity_file: Optional[str],
    dry_run: bool,
) -> None:
    if use_rsync:
        ssh_part = f"ssh -p {ssh_port}"
        if identity_file:
            ssh_part += f" -i {shlex.quote(identity_file)}"
        cmd = [
            "rsync",
            "-avz",
            "--progress",
            "-e",
            ssh_part,
            local_path,
            f"{remote_user}@{remote_host}:{remote_dir}/",
        ]
    else:
        cmd = [
            "scp",
            "-P",
            str(ssh_port),
        ]
        if identity_file:
            cmd += ["-i", identity_file]
        cmd += [local_path, f"{remote_user}@{remote_host}:{remote_dir}/"]

    logger.info("执行: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remote-host", default="172.23.166.103", help="内网机 IP")
    parser.add_argument("--remote-user", default="xieruiyang", help="内网机 SSH 用户名")
    parser.add_argument(
        "--remote-subdir",
        default="oscar_en_jsonl",
        help="相对内网用户主目录的子目录，即 ~/remote-subdir/",
    )
    parser.add_argument(
        "--staging-dir",
        default="./oscar_staging",
        help="外网机本地暂存分片目录（需大磁盘）",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "legacy", "oscar2201"),
        default="oscar2201",
        help="HF 数据源；内网实验若对齐 BLOOM legacy 可改为 legacy/auto",
    )
    parser.add_argument(
        "--chunk-lines",
        type=int,
        default=200_000,
        help="每个 JSONL 分片最多写入多少条非空文档（整型）",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="全局最多处理多少条非空文档（调试用）；不设则直到流结束",
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="不分片，整库写入 --output-jsonl 后只推送一次（TB 级慎用）",
    )
    parser.add_argument(
        "--output-jsonl",
        default="",
        help="与 --single-file 配合：单文件本地路径",
    )
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--identity-file", default="", help="SSH 私钥路径，可选")
    parser.add_argument("--use-scp", action="store_true", help="改用 scp（默认 rsync）")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令不执行下载与传输")
    parser.add_argument(
        "--delete-local-after-push",
        action="store_true",
        help="每片 rsync/scp 成功后删除外网机本地该分片，节省磁盘",
    )
    parser.add_argument("--flush-every", type=int, default=5000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    if not args.use_scp and shutil.which("rsync") is None:
        logger.error("未找到 rsync，请安装 rsync 或加 --use-scp")
        return 1
    if args.use_scp and shutil.which("scp") is None:
        logger.error("未找到 scp")
        return 1

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    token = _hf_token()
    use_rsync = not args.use_scp
    ident = args.identity_file or None

    remote_dir = f"/home/{args.remote_user}/{args.remote_subdir.strip('/')}".rstrip("/")
    staging = os.path.abspath(args.staging_dir)
    os.makedirs(staging, exist_ok=True)

    _remote_mkdir(
        args.remote_user,
        args.remote_host,
        remote_dir,
        args.ssh_port,
        ident,
        args.dry_run,
    )

    if args.source == "auto":
        label, iterator = _resolve_iterator("auto", token)
    elif args.source == "legacy":
        label, iterator = "legacy", _iter_legacy()
    else:
        label, iterator = "oscar2201", _iter_oscar2201(token)

    logger.info("数据源: %s | 远端目录: %s:%s", label, args.remote_host, remote_dir)

    try:
        from tqdm import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None  # type: ignore[misc, assignment]

    pbar = tqdm(desc="docs", unit="") if tqdm else None

    if args.single_file:
        if not args.output_jsonl:
            logger.error("--single-file 需要指定 --output-jsonl")
            return 1
        out_path = os.path.abspath(args.output_jsonl)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        n = 0
        with open(out_path, "w", encoding="utf-8", buffering=1024 * 1024) as out:
            for row in iterator:
                text = _row_to_text(row)
                if not text:
                    if pbar:
                        pbar.update(1)
                    continue
                out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n += 1
                if args.flush_every and n % args.flush_every == 0:
                    out.flush()
                if pbar:
                    pbar.update(1)
                if args.max_docs is not None and n >= args.max_docs:
                    break
            out.flush()
        logger.info("单文件写入完成: %s (%s 行)", out_path, n)
        _push_file(
            out_path,
            args.remote_user,
            args.remote_host,
            remote_dir,
            use_rsync=use_rsync,
            ssh_port=args.ssh_port,
            identity_file=ident,
            dry_run=args.dry_run,
        )
        if pbar:
            pbar.close()
        logger.info("完成。已推送单文件到远端 ~/{}".format(args.remote_subdir))
        return 0

    global_count = 0
    chunk_idx = 0
    current_path: Optional[str] = None
    current_out: Optional[TextIO] = None
    lines_in_chunk = 0
    pushed_files = 0

    def open_chunk() -> None:
        nonlocal chunk_idx, current_path, current_out, lines_in_chunk
        name = f"oscar_en_{chunk_idx:05d}.jsonl"
        current_path = os.path.join(staging, name)
        current_out = open(current_path, "w", encoding="utf-8", buffering=1024 * 1024)
        lines_in_chunk = 0
        logger.info("新开分片: %s", current_path)

    def close_and_push() -> None:
        nonlocal current_out, current_path, chunk_idx, pushed_files
        if current_out is None or current_path is None:
            return
        current_out.flush()
        current_out.close()
        current_out = None
        if not args.dry_run and os.path.getsize(current_path) == 0:
            os.remove(current_path)
            logger.info("删除空分片: %s", current_path)
            return
        _push_file(
            current_path,
            args.remote_user,
            args.remote_host,
            remote_dir,
            use_rsync=use_rsync,
            ssh_port=args.ssh_port,
            identity_file=ident,
            dry_run=args.dry_run,
        )
        pushed_files += 1
        if args.delete_local_after_push and not args.dry_run:
            os.remove(current_path)
            logger.info("已删除本地分片: %s", current_path)
        chunk_idx += 1

    open_chunk()

    for row in iterator:
        if args.max_docs is not None and global_count >= args.max_docs:
            break
        text = _row_to_text(row)
        if not text:
            if pbar:
                pbar.update(1)
            continue
        assert current_out is not None
        current_out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        global_count += 1
        lines_in_chunk += 1
        if pbar:
            pbar.update(1)
        if args.flush_every and lines_in_chunk % args.flush_every == 0:
            current_out.flush()

        if lines_in_chunk >= args.chunk_lines:
            close_and_push()
            if args.max_docs is not None and global_count >= args.max_docs:
                break
            open_chunk()

    if current_out is not None:
        if lines_in_chunk > 0:
            close_and_push()
        else:
            current_out.close()
            if current_path and not args.dry_run and os.path.isfile(current_path):
                if os.path.getsize(current_path) == 0:
                    os.remove(current_path)

    if pbar:
        pbar.close()

    logger.info(
        "全部完成。已写入非空文档 %s 条；成功推送分片 %s 个；远端目录 %s:%s/",
        global_count,
        pushed_files,
        args.remote_host,
        remote_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fetch only the ILSVRC2012 training classes a campaign actually needs.

The official train archive is a single 147.9 GB tar holding one 1000-class
archive per member. Downloading all of it is unnecessary when a workload only
asks for a subset: this script random-accesses the remote tar with HTTP range
requests, walks the member headers (a few hundred bytes each), and streams
only the class archives that are needed until the requested image count is
reached. Everything lands in the same ``train/<wnid>/`` layout the extraction
script produces.
"""
from __future__ import annotations

import argparse
import io
import math
import os
import pathlib
import sys
import tarfile
import time
import urllib.request

TRAIN_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"


class RangeReader(io.RawIOBase):
    """Seekable read-only view of a remote file over HTTP range requests."""

    def __init__(self, url: str, readahead: int = 8 << 20) -> None:
        super().__init__()
        self.url = url
        self.name = url
        self.readahead = readahead
        self.pos = 0
        self._buf = b""
        self._buf_start = 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            self.pos = offset
        elif whence == os.SEEK_CUR:
            self.pos += offset
        elif whence == os.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def _fetch(self, start: int, length: int) -> bytes:
        end = min(start + length, self.size) - 1
        if end < start:
            return b""
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}"}
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    return response.read()
            except Exception as exc:  # noqa: BLE001
                print(f"range read {start}-{end} failed ({exc}); retrying", flush=True)
                time.sleep(5)
        raise RuntimeError(f"could not read range {start}-{end} of {self.url}")

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.pos
        size = min(size, self.size - self.pos)
        if size <= 0:
            return b""
        if (
            self._buf_start <= self.pos
            and self.pos + size <= self._buf_start + len(self._buf)
        ):
            start = self.pos - self._buf_start
            self.pos += size
            return self._buf[start : start + size]
        fetch_len = max(self.readahead, size)
        self._buf = self._fetch(self.pos, fetch_len)
        self._buf_start = self.pos
        take = min(size, len(self._buf))
        chunk = self._buf[:take]
        self.pos += take
        return chunk

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def extract_class_archive(blob: bytes, target: pathlib.Path) -> int:
    """Extract one class archive and return how many images it held."""
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = pathlib.PurePosixPath(member.name).name
            if not name:
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            with (target / name).open("wb") as out:
                while True:
                    chunk = source.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest",
        type=pathlib.Path,
        default=pathlib.Path("../datasets/imagenet/ILSVRC2012"),
    )
    parser.add_argument("--images", type=int, default=189380)
    parser.add_argument("--url", default=TRAIN_URL)
    args = parser.parse_args()

    dest = args.dest.resolve()
    train = dest / "train"
    train.mkdir(parents=True, exist_ok=True)
    done_marker = dest / "subset_done.json"
    existing = sum(len(files) for _, _, files in os.walk(train))
    if done_marker.exists() and existing >= args.images:
        print(f"subset already complete: {existing} images", flush=True)
        return 0

    reader = RangeReader(args.url)
    print(
        f"remote archive {reader.size / 1e9:.2f} GB; extracting classes until "
        f"{args.images} images (already present: {existing})",
        flush=True,
    )
    total = existing
    classes = 0
    with tarfile.open(fileobj=reader, mode="r:") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".tar"):
                continue
            wnid = pathlib.PurePosixPath(member.name).stem
            target = train / wnid
            if target.exists() and any(target.iterdir()):
                have = sum(1 for _ in target.iterdir())
                total += have
                classes += 1
                print(f"skip {wnid} (already extracted, {have} files)", flush=True)
                if total >= args.images:
                    break
                continue
            print(
                f"class {wnid}: member={member.size / 1e6:.1f} MB "
                f"offset={member.offset_data}",
                flush=True,
            )
            reader.seek(member.offset_data)
            blob = bytearray()
            remaining = member.size
            while remaining > 0:
                chunk = reader.read(min(1 << 22, remaining))
                if not chunk:
                    break
                blob.extend(chunk)
                remaining -= len(chunk)
            written = extract_class_archive(bytes(blob), target)
            total += written
            classes += 1
            print(f"  extracted {written} images ({total} total)", flush=True)
            done_marker.write_text(
                f'{{"classes": {classes}, "images": {total}}}\n'
            )
            if total >= args.images:
                break
    print(f"subset complete: {total} images in {classes} classes", flush=True)
    return 0 if total >= args.images else 1


if __name__ == "__main__":
    sys.exit(main())

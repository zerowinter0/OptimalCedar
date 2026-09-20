"""Offline verified CommonVoice expansion followed by a fresh one-round matrix."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

REPO = Path("/workspace/OptimalCedar")
ROOT = REPO / "outputs/commonvoice_300000_layered_20260918"
ARCHIVES = REPO / "datasets/commonvoice/cv15_en_train_5shards_archives"
ADDITIONAL = REPO / "datasets/commonvoice/cv15_en_train_additional"
SELECTED = REPO / "datasets/commonvoice/cv15_en_train_300000"
ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(REPO)

def status(phase, **kwargs):
    (ROOT / "queue_status.json").write_text(json.dumps(dict(phase=phase, **kwargs), indent=2))
    print(phase, kwargs, flush=True)

try:
    status("downloading")
    req = urllib.request.Request("https://huggingface.co/api/datasets/fsicoli/common_voice_15_0/tree/main/audio/en/train?limit=1000", headers={"User-Agent": "OptimalCedar-experiment"})
    with urllib.request.urlopen(req, timeout=90) as response:
        entries = json.load(response)
    entries = {Path(e["path"]).name: e for e in entries}
    ADDITIONAL.mkdir(parents=True, exist_ok=True)
    download_metadata = []
    for index in (5, 6, 7):
        name = f"en_train_{index}.tar"
        entry = entries[name]
        lfs = entry.get("lfs", {})
        expected_hash = lfs.get("oid")
        expected_size = entry.get("size")
        if not expected_hash or not expected_size:
            raise RuntimeError(f"No authoritative size/SHA256 for {name}")
        archive = ARCHIVES / name
        marker = ARCHIVES / f"en_train_{index}.300k_verified"
        if not marker.exists():
            url = f"https://huggingface.co/datasets/fsicoli/common_voice_15_0/resolve/main/audio/en/train/{name}?download=true"
            subprocess.run(["curl", "--http1.1", "-fL", "--silent", "--show-error", "--retry", "20", "--retry-all-errors", "--retry-delay", "5", "--connect-timeout", "60", "--speed-limit", "1024", "--speed-time", "120", "-C", "-", "-o", str(archive), url], check=True)
            if archive.stat().st_size != expected_size:
                raise RuntimeError(f"Size mismatch: {name}")
            digest = hashlib.sha256()
            with archive.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise RuntimeError(f"SHA256 mismatch: {name}")
            with tarfile.open(archive) as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".mp3"):
                        continue
                    target = ADDITIONAL / Path(member.name).name
                    if target.exists():
                        raise RuntimeError(f"Duplicate clip name: {target.name}")
                    source = tar.extractfile(member)
                    with target.open("wb") as output:
                        import shutil
                        shutil.copyfileobj(source, output)
            marker.write_text(expected_hash + "\n")
        download_metadata.append(dict(name=name, size=expected_size, sha256=expected_hash))
    status("selecting_dataset")
    files = {}
    for base in (REPO / "datasets/commonvoice/cv15_en_train_5shards", ADDITIONAL):
        for file in sorted(base.glob("*.mp3")):
            if file.name in files:
                raise RuntimeError(f"Duplicate clip name across shards: {file.name}")
            files[file.name] = file
    if len(files) < 300000:
        raise RuntimeError(f"Only {len(files)} unique clips available")
    SELECTED.mkdir(parents=True, exist_ok=True)
    selected = [files[name] for name in sorted(files)[:300000]]
    for file in selected:
        target = SELECTED / file.name
        if not target.exists():
            target.symlink_to(file)
    if len(list(SELECTED.glob("*.mp3"))) != 300000:
        raise RuntimeError("Selected dataset must contain exactly 300000 clips")
    manifest = "".join(str(file) + "\n" for file in selected)
    (ROOT / "dataset_manifest.txt").write_text(manifest)
    (ROOT / "dataset_metadata.json").write_text(json.dumps(dict(dataset="CommonVoice 15.0 English train, shards 0-7", input_records=300000, available_unique_clips=len(files), manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(), additional_archives=download_metadata), indent=2))
    status("experiment_running")
    cmd = [sys.executable, "-u", "evaluation/chapter6_experiments/run_simple_dp_ablation_matrix.py", "--output", str(ROOT / "matrix"), "--workloads", "commonvoice", "--commonvoice-max-samples", "300000", "--commonvoice-dataset-path", str(SELECTED), "--repeats", "1", "--layered-profile", "--cell-timeout-sec", "7200"]
    subprocess.run(cmd, check=True)
    status("finished")
except BaseException as exc:
    status("failed", error=repr(exc))
    raise

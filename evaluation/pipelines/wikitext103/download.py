"""
Download the WikiText103 dataset to local filesystem.

The original MetaMind S3 archive is no longer reliable, so this script first
tries the legacy zip and then falls back to rebuilding the expected token files
from the Hugging Face parquet shards.
"""

import logging
import pathlib
import urllib.request
import zipfile

DATASET_NAME = "wikitext103"
DATASET_LOC = "datasets/wikitext103"
DATASET_FILE = "wikitext-103-v1.zip"
LEGACY_DATASET_SOURCES = [
    "https://wikitext.smerity.com/wikitext-103-v1.zip",
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip",
]
HF_PARQUET_SOURCES = {
    "wiki.train.tokens": [
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        "wikitext-103-v1/train-00000-of-00002.parquet?download=true",
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        "wikitext-103-v1/train-00001-of-00002.parquet?download=true",
    ],
    "wiki.valid.tokens": [
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        "wikitext-103-v1/validation-00000-of-00001.parquet?download=true",
    ],
    "wiki.test.tokens": [
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        "wikitext-103-v1/test-00000-of-00001.parquet?download=true",
    ],
}
DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0",
}


logger = logging.getLogger(__name__)


def _download_to_path(url: str, path: pathlib.Path) -> None:
    request = urllib.request.Request(url, headers=DOWNLOAD_HEADERS)
    with urllib.request.urlopen(request) as response, open(path, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _dataset_ready(dataset_dir: pathlib.Path) -> bool:
    required_files = (
        dataset_dir / "wiki.train.tokens",
        dataset_dir / "wiki.valid.tokens",
        dataset_dir / "wiki.test.tokens",
    )
    return all(path.is_file() for path in required_files)


def _try_legacy_zip(data_dir: pathlib.Path) -> bool:
    dataset_file = data_dir / DATASET_FILE
    for url in LEGACY_DATASET_SOURCES:
        logger.info("Trying WikiText103 archive: %s", url)
        try:
            _download_to_path(url, dataset_file)
            logger.info("Extracting Wikitext103 data from zip file.")
            with zipfile.ZipFile(dataset_file, "r") as zip_ref:
                zip_ref.extractall(path=data_dir)
            logger.info("Done extracting Wikitext103 data from zip file.")
            return _dataset_ready(data_dir / "wikitext-103")
        except Exception as exc:
            logger.warning("Archive download failed from %s: %s", url, exc)
        finally:
            if dataset_file.exists():
                dataset_file.unlink()

    return False


def _rebuild_tokens_from_hf(dataset_dir: pathlib.Path) -> None:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Fallback download requires pyarrow to read Hugging Face parquet "
            "files. Please install dependencies from requirements.txt first."
        ) from exc

    tmp_dir = dataset_dir / "_hf_parquet_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for output_name, urls in HF_PARQUET_SOURCES.items():
        output_path = dataset_dir / output_name
        logger.info("Rebuilding %s from Hugging Face parquet shards.", output_name)
        with open(output_path, "w", encoding="utf-8") as outfile:
            for shard_idx, url in enumerate(urls):
                shard_path = tmp_dir / f"{output_name}.{shard_idx}.parquet"
                logger.info("Downloading shard %s", url)
                _download_to_path(url, shard_path)

                parquet_file = pq.ParquetFile(shard_path)
                for batch in parquet_file.iter_batches(columns=["text"]):
                    for text in batch.column("text").to_pylist():
                        outfile.write(text)
                        outfile.write("\n")

                shard_path.unlink()

    tmp_dir.rmdir()


def download_dataset() -> None:
    logger.info("Downloading Wikitext103 Dataset")
    data_dir = pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = data_dir / "wikitext-103"
    if _dataset_ready(dataset_dir):
        print(f"Path already exists: {dataset_dir}")
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)

    if _try_legacy_zip(data_dir):
        print(f"Downloaded dataset to {dataset_dir}")
        return

    logger.info("Falling back to Hugging Face parquet shards.")
    _rebuild_tokens_from_hf(dataset_dir)
    logger.info("Done rebuilding Wikitext103 token files.")
    print(f"Downloaded dataset to {dataset_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_dataset()

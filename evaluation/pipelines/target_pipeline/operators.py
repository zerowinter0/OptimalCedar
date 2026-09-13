"""Native operations; no model inference is substituted for preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
import random
import re

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision.io import read_image, ImageReadMode


@dataclass(frozen=True)
class ReadImage:
    kind: str = "pil"

    def __call__(self, path):
        if self.kind == "tensor":
            return read_image(path, mode=ImageReadMode.RGB)
        with Image.open(path) as image:
            return image.convert("RGB")


def to_float(image):
    # Same as the existing Cedar SimCLR; intentionally no division by 255.
    return image.to(torch.float32)


@dataclass(frozen=True)
class GaussianBlur:
    probability: float
    numpy_probability: bool = False

    def __call__(self, image):
        draw = np.random.rand() if self.numpy_probability else random.random()
        if draw > self.probability:
            return image
        return image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 2.0)))


@dataclass(frozen=True)
class Solarize:
    probability: float = 0.2

    def __call__(self, image):
        return ImageOps.solarize(image) if random.random() < self.probability else image


def clean_caption(caption):
    caption = re.sub(r'([.!"()*#:;~])', " ", caption.lower())
    return re.sub(r"\s{2,}", " ", caption).rstrip("\n").strip(" ")


def local_only(pipe):
    """Pin a pipe to INPROCESS.

    ``ReadRecord`` opens local file paths, so it cannot execute on a remote
    backend without shared storage. The SimCLR workload expresses the same
    constraint with an ``ImageReaderPipe`` source; the four manifest-based
    workloads read through a mapper, which Cedar's profiler and optimizers would
    otherwise place on the remote Ray node and fail at runtime.
    """
    from cedar.pipes.common import CedarPipeSpec
    from cedar.pipes.context import PipeVariantType

    spec = pipe.pipe_spec
    pipe.pipe_spec = CedarPipeSpec(
        is_mutable=True,
        mutable_variants=[PipeVariantType.INPROCESS],
        is_fusable=spec.is_fusable,
        is_shardable=spec.is_shardable,
        is_fusable_source=spec.is_fusable_source,
        fusable_source_variants=spec.fusable_source_variants,
    )
    return pipe


@dataclass(frozen=True)
class TruncateCaption:
    max_words: int = 30

    def __call__(self, caption):
        words = caption.split(" ")
        return " ".join(words[:self.max_words]) if len(words) > self.max_words else caption


class TokenizeCaption:
    def __init__(self, tokenizer_path, max_length=77):
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self._tokenizer = None

    def _resolve(self):
        """Resolve the tokenizer on any execution host.

        The local host passes the dataset's tokenizer directory directly, but a
        remote backend (Ray) only receives this package's working directory, so
        an absolute path from the driver may not exist there. Fall back to a
        copy shipped next to this module.
        """
        import pathlib

        candidates = [pathlib.Path(self.tokenizer_path)]
        candidates.append(
            pathlib.Path(__file__).resolve().parent
            / "datasets/target_pipeline/clip_tokenizer"
        )
        for candidate in candidates:
            if candidate.is_dir():
                return str(candidate)
        return self.tokenizer_path

    def __call__(self, caption):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._resolve(), local_files_only=True, use_fast=True
            )
        return dict(self._tokenizer(
            caption, max_length=self.max_length, padding="max_length",
            truncation=True, return_token_type_ids=False,
        ))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        return state


def tensorize_tokens(tokens):
    return {key: torch.tensor(value, dtype=torch.long) for key, value in tokens.items()}

import pathlib
import logging

import torch.nn as nn
import multiprocessing as mp

from typing import List

import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

from cedar.client import DataSet
from cedar.config import CedarContext
from cedar.compose import Feature, OptimizerOptions
from cedar.pipes import (
    Pipe,
    MapperPipe,
    BatcherPipe,
)
from cedar.sources import LocalLineSource

from evaluation.cedar_utils import CedarEvalSpec


DATASET_LOC = "datasets/wikitext103"
DATASET_NAME = "wikitext103"
DATASET_FILE = "wikitext-103-v1.zip"
DATASET_SOURCE = "https://s3.amazonaws.com/research.metamind.io/wikitext/\
wikitext-103-v1.zip"


def _get_text(data):
    return data


class Wikitext103Feature(Feature):
    def __init__(self, batch_size: int):
        super().__init__()
        encoder_json_path = (
            "https://download.pytorch.org/models/text/gpt2_bpe_encoder.json"
        )
        vocab_bpe_path = (
            "https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe"
        )
        self.tokenizer = T.GPT2BPETokenizer(encoder_json_path, vocab_bpe_path)
        vocab_path = (
            "https://download.pytorch.org/models/text/roberta.vocab.pt"
        )
        self.vocab = T.VocabTransform(load_state_dict_from_url(vocab_path))
        self.add_bos = T.AddToken(token=0, begin=True)
        self.add_bos_1 = T.AddToken(token=1, begin=True)
        self.add_eos = T.AddToken(token=2, begin=False)
        self.batch_size = batch_size

        self.embedding = nn.Embedding(50257, 764, _freeze=True)

    def _compose(self, source_pipes: List[Pipe]):
        fp = source_pipes[0]
        fp = MapperPipe(fp, self.tokenizer).fix()
        fp = MapperPipe(fp, T.Truncate(max_seq_len=254)).fix()
        fp = MapperPipe(fp, self.vocab).fix()
        fp = MapperPipe(fp, self.add_bos)
        fp = MapperPipe(fp, self.add_bos_1)
        fp = MapperPipe(fp, self.add_eos)
        fp = MapperPipe(fp, T.ToTensor(), tag="tensor").fix()
        fp = MapperPipe(fp, self.embedding).fix()
        fp = BatcherPipe(fp, batch_size=self.batch_size).fix()
        return fp


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    train_filepath = pathlib.Path(data_dir) / pathlib.Path(
        "wikitext-103/wiki.train.tokens"
    )

    ctx = CedarContext(ray_config=spec.to_ray_config())
    max_samples = (spec.kwargs or {}).get("max_samples")
    max_samples = int(max_samples) if max_samples is not None else None
    source = LocalLineSource(str(train_filepath), max_samples=max_samples)
    feature = Wikitext103Feature(batch_size=spec.batch_size)
    feature.apply(source)

    if spec.config:
        dataset = DataSet(
            ctx,
            {"feature": feature},
            spec.config,
            enable_controller=False,
            enable_optimizer=False,
        )
    else:
        dataset = DataSet(
            ctx,
            {"feature": feature},
            enable_controller=not spec.disable_controller,
            enable_optimizer=not spec.disable_optimizer,
            profiled_data=spec.profiled_stats,
            run_profiling=spec.run_profiling,
            optimizer_options=OptimizerOptions(
                enable_prefetch=not spec.disable_prefetch,
                est_throughput=None,
                available_local_cpus=mp.cpu_count(),
                enable_offload=not spec.disable_offload,
                enable_reorder=not spec.disable_reorder,
                enable_caching=not spec.disable_caching,
                num_samples=getattr(spec, "num_total_samples", None),
                enable_local_parallelism=not spec.disable_parallelism,
                enable_fusion=not spec.disable_fusion,
                use_my_optimizer=getattr(spec, "use_my_optimizer", 0),
                reorder_timeout_sec=getattr(spec, "reorder_timeout_sec", None),
            ),
            generate_plan=spec.generate_plan,
        )

    return dataset


def main():
    logging.basicConfig(level=logging.INFO)
    ds = get_dataset(CedarEvalSpec(1, None, 1, 
    run_profiling=False,
    use_ray=True,
    profiled_stats="/tmp/feature_profile.yml",
    disable_offload=False,
    use_my_optimizer=0,
    disable_prefetch=True,
    disable_fusion=False,
    disable_caching=False,
    ))

    i = 0
    for f in ds:
        # print(f)
        print(f)
        print(f.size())
        if i == 10:
            break
        i += 1


if __name__ == "__main__":
    main()

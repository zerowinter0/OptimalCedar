from cedar.sources import IterSource

from evaluation.chapter6_experiments.analyze_datajuicer_diverse_workloads import (
    WORKLOAD_META,
)
from evaluation.pipelines.alpaca_cot.cedar_dataset import AlpacaCotFeature
from evaluation.pipelines.general_video_refine.cedar_dataset import (
    GeneralVideoRefineFeature,
)
from evaluation.pipelines.llava_pretrain.cedar_dataset import LlavaPretrainFeature
from evaluation.pipelines.pile_europarl.cedar_dataset import PileEuroparlFeature
from evaluation.pipelines.pile_recipe_registry import PileRecipeFeature
from evaluation.pipelines.redpajama_arxiv.cedar_dataset import (
    RedPajamaArxivFeature,
)
from evaluation.pipelines.redpajama_code.cedar_dataset import (
    RedPajamaCodeFeature,
)
from evaluation.pipelines.video_self_evolution.cedar_dataset import (
    VideoSelfEvolutionFeature,
)


def _logical_operator_count(feature):
    feature.apply(IterSource(["{}"]),)
    return sum(not pipe.is_source() for pipe in feature.logical_pipes.values())


def test_reported_cedar_operator_counts_match_actual_features():
    features = {
        "pile_europarl": PileEuroparlFeature(),
        "pile_hackernews": PileRecipeFeature("pile_hackernews"),
        "pile_pubmed_abstracts": PileRecipeFeature(
            "pile_pubmed_abstracts"
        ),
        "pile_uspto_backgrounds": PileRecipeFeature(
            "pile_uspto_backgrounds"
        ),
        "redpajama_code": RedPajamaCodeFeature(),
        "redpajama_arxiv": RedPajamaArxivFeature(),
        "alpaca_cot": AlpacaCotFeature(),
        "llava_pretrain": LlavaPretrainFeature(),
        "general_video_refine": GeneralVideoRefineFeature("."),
        "video_self_evolution": VideoSelfEvolutionFeature("."),
    }

    observed = {
        workload: _logical_operator_count(feature)
        for workload, feature in features.items()
    }
    expected = {
        workload: WORKLOAD_META[workload][3] for workload in features
    }
    assert observed == expected

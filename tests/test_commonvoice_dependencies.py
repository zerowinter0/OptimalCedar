from cedar.compose.utils import derive_constraint_graph
from cedar.sources import IterSource
from evaluation.pipelines.commonvoice.cedar_dataset import CommonvoiceFeature


def test_commonvoice_masks_preserve_semantic_prefix_order():
    feature = CommonvoiceFeature(batch_size=1)
    feature.apply(IterSource(["unused.wav"]))
    by_tag = {
        pipe.tag: pipe_id
        for pipe_id, pipe in feature.logical_pipes.items()
        if pipe.tag is not None
    }
    constraints = derive_constraint_graph(feature.logical_pipes)

    stretch = by_tag["commonvoice_stretch"]
    time_mask = by_tag["commonvoice_time_mask"]
    frequency_mask = by_tag["commonvoice_frequency_mask"]
    assert time_mask in constraints[stretch]
    assert frequency_mask in constraints[time_mask]

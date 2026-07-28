from evaluation.pipelines.native_fm_adapters import get_torch_dataset


def get_dataset(spec):
    return get_torch_dataset(spec, "pile_pubmed_abstracts")

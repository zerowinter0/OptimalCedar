from evaluation.pipelines.native_fm_adapters import get_tf_dataset


def get_dataset(spec):
    return get_tf_dataset(spec, "llava_pretrain")

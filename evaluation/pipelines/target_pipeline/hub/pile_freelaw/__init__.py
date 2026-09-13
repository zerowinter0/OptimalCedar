from evaluation.pipelines.target_pipeline.hub_dataset import get_dataset


def get_target_dataset(spec):
    return get_dataset("pile_freelaw", spec)

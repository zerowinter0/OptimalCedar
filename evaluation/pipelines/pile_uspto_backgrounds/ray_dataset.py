from evaluation.pipelines.native_fm_adapters import get_ray_dataset


def get_dataset(spec):
    return get_ray_dataset(spec, "pile_uspto_backgrounds")

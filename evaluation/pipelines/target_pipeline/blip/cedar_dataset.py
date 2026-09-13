"""BLIP pretraining image augmentation and caption preprocessing."""
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from cedar.compose import Feature
from cedar.pipes import MapperPipe, BatcherPipe

from evaluation.pipelines.target_pipeline.fields import OnField, ReadRecord, collect_blip
from evaluation.pipelines.target_pipeline.operators import local_only
from evaluation.pipelines.target_pipeline.operators import clean_caption, TruncateCaption
from evaluation.pipelines.target_pipeline.references.blip.transform.randaugment import RandomAugment
from evaluation.pipelines.target_pipeline.runtime import create_dataset


class PILRandomAugment(RandomAugment):
    """Keep a PIL image boundary so crop/flip can also follow augmentation."""
    def __call__(self, image):
        return Image.fromarray(super().__call__(image))


class BLIPFeature(Feature):
    def __init__(self, batch_size=1, image_root=""):
        super().__init__()
        self.batch_size = batch_size
        self.image_root = image_root

    def _compose(self, source_pipes):
        fp = local_only(MapperPipe(source_pipes[0], ReadRecord(self.image_root), tag="read").fix())
        fp = MapperPipe(fp, OnField("pixels", transforms.RandomResizedCrop(
            224, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC)),
            tag="crop")
        fp = MapperPipe(fp, OnField("pixels", transforms.RandomHorizontalFlip()),
                        tag="flip").depends_on(["crop"])
        # Treat RandAugment's two randomly chosen operations as one policy.
        # Its position relative to crop/flip is deliberately reorderable.
        fp = MapperPipe(fp, OnField("pixels", PILRandomAugment(2, 5, isPIL=True, augs=[
            "Identity", "AutoContrast", "Brightness", "Sharpness", "Equalize",
            "ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate",
        ])), tag="augment")
        fp = MapperPipe(fp, OnField("pixels", transforms.ToTensor()),
                        tag="tensor").depends_on(["flip", "augment"])
        fp = MapperPipe(fp, OnField("pixels", transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711))),
            tag="normalize").depends_on(["tensor"])
        fp = MapperPipe(fp, OnField("caption", clean_caption), tag="clean")
        # Cleaning defines word boundaries, so truncate must follow it.
        fp = MapperPipe(fp, OnField("caption", TruncateCaption(30)),
                        tag="truncate").depends_on(["clean"])
        fp = MapperPipe(fp, collect_blip, tag="collect").fix()
        if self.batch_size > 1:
            fp = BatcherPipe(fp, batch_size=self.batch_size).fix()
        return fp


def get_dataset(spec):
    kwargs = spec.kwargs or {}
    return create_dataset(BLIPFeature(spec.batch_size, kwargs.get("image_root", "")), spec)

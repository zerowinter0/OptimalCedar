"""CLIP image and text preprocessing, with explicit partial dependencies."""
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from cedar.compose import Feature
from cedar.pipes import MapperPipe, BatcherPipe

from evaluation.pipelines.target_pipeline.fields import OnField, ReadRecord, collect_clip
from evaluation.pipelines.target_pipeline.operators import TokenizeCaption, tensorize_tokens
from evaluation.pipelines.target_pipeline.operators import local_only
from evaluation.pipelines.target_pipeline.runtime import create_dataset


class CLIPFeature(Feature):
    def __init__(self, batch_size=1, image_root="",
                 tokenizer_path="openai/clip-vit-base-patch32"):
        super().__init__()
        self.batch_size = batch_size
        self.image_root = image_root
        self.tokenizer_path = tokenizer_path

    def _compose(self, source_pipes):
        fp = local_only(MapperPipe(
            source_pipes[0], ReadRecord(self.image_root, "tensor"),
            tag="read").fix())
        fp = MapperPipe(fp, OnField("pixels", transforms.Resize(
            [224], interpolation=InterpolationMode.BICUBIC)), tag="resize")
        # CenterCrop must operate on the resized image to retain CLIP geometry.
        fp = MapperPipe(fp, OnField("pixels", transforms.CenterCrop(224)),
                        tag="crop").depends_on(["resize"])
        fp = MapperPipe(fp, OnField("pixels", transforms.ConvertImageDtype(
            torch.float32)), tag="float")
        # Float conversion can move across geometry; normalization stays last.
        fp = MapperPipe(fp, OnField("pixels", transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711))),
            tag="normalize").depends_on(["float", "crop"])
        fp = MapperPipe(fp, OnField("caption", TokenizeCaption(self.tokenizer_path)),
                        tag="tokenize")
        fp = MapperPipe(fp, OnField("caption", tensorize_tokens),
                        tag="text_tensor").depends_on(["tokenize"])
        fp = MapperPipe(fp, collect_clip, tag="collect").fix()
        if self.batch_size > 1:
            fp = BatcherPipe(fp, batch_size=self.batch_size).fix()
        return fp


def get_dataset(spec):
    kwargs = spec.kwargs or {}
    return create_dataset(CLIPFeature(
        spec.batch_size, kwargs.get("image_root", ""),
        kwargs.get("tokenizer_path", "openai/clip-vit-base-patch32")), spec)

"""SwAV multi-crop preprocessing with explicit Cedar operators."""
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from cedar.compose import Feature
from cedar.pipes import MapperPipe, BatcherPipe

from evaluation.pipelines.target_pipeline.fields import OnField, ReadRecord, CollectViews
from evaluation.pipelines.target_pipeline.operators import GaussianBlur, Solarize
from evaluation.pipelines.target_pipeline.operators import local_only
from evaluation.pipelines.target_pipeline.runtime import create_dataset


class SwAVFeature(Feature):
    def __init__(self, batch_size=1, image_root="", views=8):
        super().__init__()
        self.batch_size = batch_size
        self.image_root = image_root
        # See DINOFeature: the view count is parameterized so the exact joint DP
        # can be evaluated on a reduced but structurally identical pipeline.
        self.views = max(1, int(views))

    def _compose(self, source_pipes):
        fp = local_only(MapperPipe(
            source_pipes[0], ReadRecord(self.image_root, views=self.views),
            tag="read").fix())
        for view in range(self.views):
            field = f"view{view}"
            global_view = view < 2
            size = 224 if global_view else 96
            scale = (0.14, 1.0) if global_view else (0.05, 0.14)
            fp = MapperPipe(fp, OnField(field, transforms.RandomResizedCrop(
                size, scale=scale)),
                tag=f"{field}_crop")
            # Match Cedar SimCLRv2's crop-before-flip contract.
            fp = MapperPipe(fp, OnField(field, transforms.RandomHorizontalFlip()),
                tag=f"{field}_flip").depends_on([f"{field}_crop"])
            fp = MapperPipe(fp, OnField(field, transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8)),
                tag=f"{field}_jitter")
            fp = MapperPipe(fp, OnField(field, transforms.RandomGrayscale(p=0.2)),
                tag=f"{field}_gray")
            fp = MapperPipe(fp, OnField(field, GaussianBlur(
                0.5, numpy_probability=True)), tag=f"{field}_blur")
            pil_ops = [f"{field}_{op}" for op in
                       ("crop", "flip", "jitter", "gray", "blur")]
            # PIL blur/solarize must finish before ToTensor.
            fp = MapperPipe(fp, OnField(field, transforms.ToTensor()),
                tag=f"{field}_tensor").depends_on(pil_ops)
            fp = MapperPipe(fp, OnField(field, transforms.Normalize(
                (0.485, 0.456, 0.406), (0.228, 0.224, 0.225))),
                tag=f"{field}_normalize").depends_on([f"{field}_tensor"])
        fp = MapperPipe(fp, CollectViews(self.views), tag="collect").fix()
        if self.batch_size > 1:
            fp = BatcherPipe(fp, batch_size=self.batch_size).fix()
        return fp


def get_dataset(spec):
    kwargs = spec.kwargs or {}
    return create_dataset(
        SwAVFeature(
            spec.batch_size,
            kwargs.get("image_root", ""),
            views=int(kwargs.get("views", 8)),
        ),
        spec,
    )

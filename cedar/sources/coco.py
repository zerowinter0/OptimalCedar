from typing import Tuple, Optional
import json
import logging
import torch
from torchvision.datapoints import BoundingBox, BoundingBoxFormat
from PIL import Image


from cedar.pipes import (
    Pipe,
    InProcessPipeVariant,
    InProcessPipeVariantContext,
    PipeVariantType,
    CedarPipeSpec,
    DataSample,
    cedar_pipe,
)
from .source import Source, SourcePipeVariantMixin

logger = logging.getLogger(__name__)


class InProcessCOCOSourcePipeVariant(
    InProcessPipeVariant, SourcePipeVariantMixin
):
    def __init__(
        self,
        source: str,
        rank_spec: Optional[Tuple[int, int]],
        split: str = "val2017",
    ):
        InProcessPipeVariant.__init__(self, None)
        SourcePipeVariantMixin.__init__(self, rank_spec=rank_spec)
        self.source = source
        if split not in ("train2017", "val2017"):
            raise ValueError(f"Unsupported COCO split: {split}")
        self.split = split
        self.num_yielded = 0

        self.img_dir = self.source + f"/{self.split}/"
        self.ann_file = self.source + f"/annotations/instances_{self.split}.json"

        self.imgs = {}
        self.annotations = {}

        with open(self.ann_file, "r") as f:
            self.ann_json = json.load(f)

        for ann in self.ann_json["annotations"]:
            id = ann["image_id"]
            if id in self.annotations:
                self.annotations[id].append(ann)
            else:
                self.annotations[id] = [ann]

        for img in self.ann_json["images"]:
            self.imgs[img["id"]] = img

    def _reset_source_iterator_for_epoch(self):
        it = iter(self.imgs.values())
        self.all_datasamples = self.create_datasamples(it, size=1)

    def _iter_impl(self):
        it = iter(self.imgs.values())
        self.all_datasamples = self.create_datasamples(it, size=1)

        while True:
            try:
                ds = next(self.all_datasamples)

                if isinstance(ds, DataSample):
                    img_id = ds.data["id"]
                    img_path = self.img_dir + ds.data["file_name"]
                else:
                    img_id = ds["id"]
                    img_path = self.img_dir + ds["file_name"]

                img = Image.open(img_path).convert("RGB")

                target = {}

                boxes = []
                labels = []
                for ann in self.annotations.get(img_id, []):
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann["category_id"])

                box_data = boxes if boxes else torch.empty((0, 4))
                bboxes = BoundingBox(
                    box_data,
                    format=BoundingBoxFormat.XYXY,
                    spatial_size=(img.height, img.width),
                )

                target["boxes"] = bboxes
                target["labels"] = labels
                target["image_id"] = torch.tensor([img_id])
                target["image"] = img

                if isinstance(ds, DataSample):
                    ds.data = target
                    yield ds
                else:
                    yield target
            except StopIteration:
                return


@cedar_pipe(
    CedarPipeSpec(
        is_mutable=False,
        mutable_variants=[
            PipeVariantType.INPROCESS,
        ],
        is_fusable=False,
        is_shardable=True,
        is_fusable_source=False,
    )
)
class COCOSourcePipe(Pipe):
    def __init__(
        self,
        source: str,
        rank_spec: Optional[Tuple[int, int]] = None,
        split: str = "val2017",
    ):
        super().__init__("COCOSourcePipe", [])  # empty inputs = source pipe
        self.source = source
        self.rank_spec = rank_spec
        self.split = split

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        assert self.is_source()
        variant = InProcessCOCOSourcePipeVariant(
            self.source, self.rank_spec, self.split
        )
        return variant


class COCOSource(Source):
    """
    COCO dataset source.

    Args:
        root (str): Root directory of the COCO dataset.
    """

    def __init__(
        self,
        root: str,
        rank_spec: Optional[Tuple[int, int]] = None,
        split: str = "val2017",
    ):
        self.source = root
        self.rank_spec = rank_spec
        self.split = split

    def to_pipe(self) -> Pipe:
        pipe = COCOSourcePipe(self.source, self.rank_spec, self.split)
        pipe.fix()
        return pipe


class InProcessCOCOFileSourcePipeVariant(
    InProcessPipeVariant, SourcePipeVariantMixin
):
    def __init__(
        self,
        source: str,
        rank_spec: Optional[Tuple[int, int]],
        tf_source: bool = False,
    ):
        InProcessPipeVariant.__init__(self, None)
        SourcePipeVariantMixin.__init__(self, rank_spec=rank_spec)
        self.source = source
        self.num_yielded = 0

        self.tf_source = tf_source

        self.img_dir = self.source + "/val2017/"
        self.ann_file = self.source + "/annotations/instances_val2017.json"

        self.imgs = {}
        self.annotations = {}

        with open(self.ann_file, "r") as f:
            self.ann_json = json.load(f)

        for ann in self.ann_json["annotations"]:
            id = ann["image_id"]
            if id in self.annotations:
                self.annotations[id].append(ann)
            else:
                self.annotations[id] = [ann]

        for img in self.ann_json["images"]:
            id = img["id"]
            if id not in self.annotations:
                continue
            self.imgs[id] = img

        assert len(self.annotations) == len(self.imgs)

        if self.tf_source:
            self.tf_imgs = []
            for img_id, img in self.imgs.items():
                boxes = []
                labels = []

                for ann in self.annotations[img_id]:
                    x, y, w, h = ann["bbox"]

                    x1 = x / img["width"]
                    y1 = y / img["height"]
                    x2 = (x + w) / img["width"]
                    y2 = (y + h) / img["height"]

                    boxes.append([x1, y1, x2, y2])
                    labels.append(ann["category_id"])

                sample = (self.img_dir + img["file_name"], labels, boxes)
                self.tf_imgs.append(sample)

    def _reset_source_iterator_for_epoch(self):
        if self.tf_source:
            it = iter(self.tf_imgs)
            self.all_datasamples = self.create_datasamples(it, size=1)
        else:
            it = iter(self.imgs.values())
        self.all_datasamples = self.create_datasamples(it, size=1)

    def _iter_impl(self):
        if self.tf_source:
            it = iter(self.tf_imgs)
        else:
            it = iter(self.imgs.values())
        self.all_datasamples = self.create_datasamples(it, size=1)

        while True:
            try:
                ds = next(self.all_datasamples)

                if self.tf_source:
                    yield ds
                else:
                    img_id = ds.data["id"]
                    img_path = self.img_dir + ds.data["file_name"]

                    target = {}

                    boxes = []
                    labels = []
                    for ann in self.annotations[img_id]:
                        x, y, w, h = ann["bbox"]
                        boxes.append([x, y, x + w, y + h])
                        labels.append(ann["category_id"])

                    target["boxes"] = boxes
                    target["labels"] = labels
                    target["image_id"] = torch.tensor([img_id])
                    target["image"] = img_path

                    ds.data = target
                    yield ds
            except StopIteration:
                return


@cedar_pipe(
    CedarPipeSpec(
        is_mutable=False,
        mutable_variants=[
            PipeVariantType.INPROCESS,
        ],
        is_fusable=False,
        is_shardable=True,
        is_fusable_source=False,
    )
)
class COCOFileSourcePipe(Pipe):
    def __init__(
        self,
        source: str,
        rank_spec: Optional[Tuple[int, int]] = None,
        tf_source: bool = False,
    ):
        super().__init__(
            "COCOFileSourcePipe", []
        )  # empty inputs = source pipe
        self.source = source
        self.rank_spec = rank_spec
        self.tf_source = tf_source

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        assert self.is_source()
        variant = InProcessCOCOFileSourcePipeVariant(
            self.source, self.rank_spec, self.tf_source
        )
        return variant


class COCOFileSource(Source):
    """
    COCO dataset source.

    Args:
        root (str): Root directory of the COCO dataset.
    """

    def __init__(
        self,
        root: str,
        rank_spec: Optional[Tuple[int, int]] = None,
        tf_source: bool = False,
    ):
        self.source = root
        self.rank_spec = rank_spec
        self.tf_source = tf_source

    def to_pipe(self) -> Pipe:
        pipe = COCOFileSourcePipe(self.source, self.rank_spec, self.tf_source)
        pipe.fix()
        return pipe

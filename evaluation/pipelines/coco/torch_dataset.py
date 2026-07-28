import torch
import json
import pathlib
import torchvision
from torchvision.transforms import v2
from evaluation.torch_utils import TorchEvalSpec
import torchdata.datapipes as dp
from torchdata.datapipes.iter import IterDataPipe
from PIL import Image

from torch.utils.data import DataLoader

DATASET_LOC = "datasets/coco"


class COCODataPipe(IterDataPipe):
    def __init__(self, root, split="val2017"):
        self.root = root + f"/{split}/"
        self.ann_file = root + f"/annotations/instances_{split}.json"
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
            self.imgs[id] = img

    def __iter__(self):
        for id, img in self.imgs.items():
            img = Image.open(self.root + img["file_name"]).convert("RGB")
            target = {}

            boxes = []
            labels = []
            for ann in self.annotations.get(id, []):
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])

            box_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(
                -1, 4
            )
            bboxes = torchvision.datapoints.BoundingBox(
                box_tensor,
                format=torchvision.datapoints.BoundingBoxFormat.XYXY,
                spatial_size=(img.height, img.width),
            )

            target["boxes"] = bboxes
            target["labels"] = labels
            target["image_id"] = torch.tensor([id])
            target["image"] = img

            yield target


def to_float(x):
    return x.to(torch.float32)


def to_tensor(x):
    x["image"] = v2.ToTensor()(x["image"])
    return x


def distort(x):
    x["image"] = v2.RandomPhotometricDistort(p=1)(x["image"])
    return x


def zoom_out(x):
    x["image"], x["boxes"] = v2.RandomZoomOut(fill=[123.0, 117.0, 104.0], p=1)(
        x["image"], x["boxes"]
    )
    return x


def crop(x):
    x["image"], x["boxes"] = v2.RandomIoUCrop()(x["image"], x["boxes"])
    return x


def build_datapipe(root, spec: TorchEvalSpec):
    datapipe = COCODataPipe(root, spec.kwargs.get("split", "val2017"))
    datapipe = datapipe.sharding_filter()
    datapipe = dp.iter.Mapper(datapipe, zoom_out)
    datapipe = dp.iter.Mapper(datapipe, crop)

    datapipe = dp.iter.Mapper(
        datapipe, v2.SanitizeBoundingBox(labels_getter="boxes")
    )
    datapipe = dp.iter.Mapper(datapipe, v2.RandomHorizontalFlip(p=1))
    datapipe = dp.iter.Mapper(datapipe, distort)
    datapipe = dp.iter.Mapper(datapipe, to_tensor)

    return datapipe


def get_dataset(spec: TorchEvalSpec):
    data_dir = spec.kwargs.get("dataset_path")
    if not data_dir:
        data_dir = (
            pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
        )

    datapipe = build_datapipe(str(data_dir), spec)
    dataloader = DataLoader(
        datapipe, batch_size=spec.batch_size, num_workers=spec.num_workers
    )
    return dataloader


def main():
    dp = get_dataset(TorchEvalSpec(batch_size=1, num_workers=1))

    for i, img in enumerate(dp):
        print(img)
        break


if __name__ == "__main__":
    main()

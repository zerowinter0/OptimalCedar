import pathlib
import torch
import torchvision
from torchvision.transforms import v2
from PIL import Image
import ray
import json
import time

DATASET_LOC = "datasets/coco"


class Timer:
    def __init__(self):
        self._start = None
        self._end = None

    def __enter__(self):
        self._start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.perf_counter()

    def reset(self):
        self._start = time.perf_counter()

    def delta(self):
        if self._start is None or self._end is None:
            raise RuntimeError()
        return self._end - self._start


def read_img(x):
    img = Image.open(x["image"]["file_name"]).convert("RGB")
    x["image"] = img

    return x


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
    box_tensor = torch.as_tensor(x["boxes"], dtype=torch.float32).reshape(-1, 4)
    bboxes = torchvision.datapoints.BoundingBox(
        box_tensor,
        format=torchvision.datapoints.BoundingBoxFormat.XYXY,
        spatial_size=(x["image"].height, x["image"].width),
    )
    x["image"], x["boxes"] = v2.RandomIoUCrop()(x["image"], bboxes)

    # Apply this here to avoid another conversion to tensors
    v2.SanitizeBoundingBox(labels_getter="boxes")(x)
    return x


def build_ds(root, split="val2017"):
    # Build list of imgs/annotations
    ann_file = root + f"/annotations/instances_{split}.json"
    annotations = {}
    ds_items = []

    with open(ann_file, "r") as f:
        ann_json = json.load(f)

    for ann in ann_json["annotations"]:
        img_id = ann["image_id"]
        annotations.setdefault(img_id, []).append(ann)

    for img in ann_json["images"]:
        img_id = img["id"]
        boxes = []
        labels = []
        for ann in annotations.get(img_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        image = dict(img)
        image["file_name"] = root + f"/{split}/" + image["file_name"]
        sample = {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
        }
        ds_items.append(sample)

    ds = ray.data.from_items(ds_items)

    ds = ds.map(read_img)
    ds = ds.map(zoom_out)
    ds = ds.map(crop)
    ds = ds.map(v2.RandomHorizontalFlip(p=1))
    ds = ds.map(distort)
    ds = ds.map(to_tensor)

    return ds


def get_dataset(spec=None):
    data_dir = None if spec is None else spec.kwargs.get("dataset_path")
    if not data_dir:
        data_dir = (
            pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
        )
    split = "val2017" if spec is None else spec.kwargs.get("split", "val2017")
    ds = build_ds(str(data_dir), split)
    ray.data.DataContext.get_current().execution_options.locality_with_output = (
        True
    )

    return ds


if __name__ == "__main__":
    ds = get_dataset()
    epoch_times = []

    for _ in range(3):
        timer = Timer()
        with timer:
            for idx, row in enumerate(ds.iter_rows()):
                pass
        epoch_times.append(timer.delta())
        print(epoch_times[-1])

    print("Epoch times: {}".format(epoch_times))

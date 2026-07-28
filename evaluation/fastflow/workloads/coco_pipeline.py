"""FastFlow-serializable COCO pipeline.

The regular TensorFlow baseline uses ``from_generator`` for variable annotation
counts. tf.data service cannot serialize Python generators, so this adapter
pads annotation tensors and uses ``from_tensor_slices``, as in Cedar's original
FastFlow artifact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import tensorflow as tf

from evaluation.pipelines.coco import tf_dataset as coco


def _read_image(path, labels, boxes, annotation_count):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return (
        image,
        labels[:annotation_count],
        boxes[:annotation_count],
    )


def build_dataset(root: Path, parallelism, batch_size: int):
    split = os.environ.get("COCO_SPLIT", "val2017")
    annotation_file = root / "annotations" / f"instances_{split}.json"
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    annotations = {}
    for annotation in data["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append(annotation)

    samples = []
    max_annotations = 0
    for image in data["images"]:
        image_annotations = annotations.get(image["id"])
        if not image_annotations:
            continue
        labels = []
        boxes = []
        for annotation in image_annotations:
            x, y, width, height = annotation["bbox"]
            boxes.append(
                [
                    x / image["width"],
                    y / image["height"],
                    (x + width) / image["width"],
                    (y + height) / image["height"],
                ]
            )
            labels.append(annotation["category_id"])
        max_annotations = max(max_annotations, len(labels))
        samples.append(
            (
                str(root / split / image["file_name"]),
                labels,
                boxes,
            )
        )

    paths = []
    padded_labels = []
    padded_boxes = []
    counts = []
    for path, labels, boxes in samples:
        count = len(labels)
        paths.append(path)
        padded_labels.append(labels + [0] * (max_annotations - count))
        padded_boxes.append(
            boxes + [[0.0, 0.0, 0.0, 0.0]] * (max_annotations - count)
        )
        counts.append(count)

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(paths),
            tf.constant(padded_labels, dtype=tf.int32),
            tf.constant(padded_boxes, dtype=tf.float32),
            tf.constant(counts, dtype=tf.int32),
        )
    )
    dataset = dataset.map(
        _read_image,
        num_parallel_calls=parallelism,
        name="prep_begin",
    )
    dataset = dataset.map(
        coco.distorted_bounding_box_crop,
        num_parallel_calls=parallelism,
    )
    dataset = dataset.map(coco.resize_image, num_parallel_calls=parallelism)
    dataset = dataset.map(coco.random_flip, num_parallel_calls=parallelism)
    dataset = dataset.map(coco.distort, num_parallel_calls=parallelism)
    dataset = dataset.map(coco.normalize, num_parallel_calls=parallelism)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


__all__ = ["build_dataset"]

"""Small field adapters for Cedar's linear record stream.

Every operator reads the current value of its own image/view/text field.
No predecessor-specific intermediate field names or implicit dependencies.
"""
from dataclasses import dataclass
from pathlib import Path
import json

from .operators import ReadImage


@dataclass(frozen=True)
class OnField:
    field: str
    operation: object

    def __call__(self, record):
        result = dict(record)
        result[self.field] = self.operation(record[self.field])
        return result


@dataclass(frozen=True)
class ReadRecord:
    image_root: str = ""
    image_kind: str = "pil"
    views: int = 0

    def __call__(self, record):
        record = json.loads(record) if isinstance(record, str) else dict(record)
        path = Path(record["image"])
        if not path.is_absolute():
            path = Path(self.image_root) / path
        image = ReadImage(self.image_kind)(str(path))
        if self.views:
            # Each view starts from the original, before any reorderable op.
            return {f"view{i}": image.copy() for i in range(self.views)}
        return {"pixels": image, "caption": record["caption"]}


@dataclass(frozen=True)
class CollectViews:
    count: int

    def __call__(self, record):
        return {"views": [record[f"view{i}"] for i in range(self.count)]}


def collect_clip(record):
    return {"pixel_values": record["pixels"], **record["caption"]}


def collect_blip(record):
    return {"pixel_values": record["pixels"], "caption": record["caption"]}

"""Remove a profile's ``physical_model.worker_contention`` block.

Used when a calibration was measured under a different resource regime than
the recorded cells (e.g. with the per-worker CPU reserve removed), which makes
the inferred worker choice unsound.
"""

import pathlib
import sys

import yaml


def main() -> int:
    for name in sys.argv[1:]:
        path = pathlib.Path(name)
        data = yaml.safe_load(path.read_text())
        model = data.get("physical_model")
        if isinstance(model, dict):
            removed = model.pop("worker_contention", None)
        else:
            removed = None
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        print(f"{path.name}: removed={bool(removed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

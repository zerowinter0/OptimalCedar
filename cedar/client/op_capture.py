"""Diagnostic capture of the payloads a running plan hands to its operators.

Enabled only when ``CEDAR_OP_CAPTURE_DIR`` is set, so it costs nothing in a
normal run.  For every mapper callable of the executing plan it records the
input representation (type, dtype, shape, contiguity, serialized size), the
wall time of the call itself, and a bounded number of the real input payloads
so they can later be replayed in isolation.

The point of the capture is that the layered profiler prices an operator from
payloads it saw in the *declared* pipeline; a reordered plan can hand the same
operator a different representation of the same record.  Only the executing
plan can say which representation actually arrived.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _describe(value: Any) -> Dict[str, Any]:
    """Cheap representation summary; never copies the payload."""
    info: Dict[str, Any] = {"type": type(value).__name__}
    try:
        import PIL.Image

        if isinstance(value, PIL.Image.Image):
            info.update(
                {"dtype": f"PIL.{value.mode}", "shape": list(value.size)}
            )
            return info
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        if isinstance(value, torch.Tensor):
            info.update(
                {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "contiguous": bool(value.is_contiguous()),
                }
            )
            return info
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, (bytes, str, Path)):
        return info
    return info


class OperatorCapture:
    """Wrap every mapper callable of one worker and record what it receives."""

    def __init__(self, directory: Path, worker: int, max_samples: int) -> None:
        self.directory = Path(directory)
        self.worker = worker
        self.max_samples = max(0, int(max_samples))
        self.calls: Dict[int, List[Dict[str, Any]]] = {}
        self.snapshots: Dict[int, List[bytes]] = {}
        self.errors: Dict[int, int] = {}
        self._calls_since_dump = 0
        self._flush_every = int(
            os.environ.get("CEDAR_OP_CAPTURE_FLUSH_EVERY", "20")
        )
        self._enabled = False

    def wrap(self, p_id: int, fn):
        if fn is None:
            return fn
        self.calls.setdefault(p_id, [])
        self.snapshots.setdefault(p_id, [])
        self._enabled = True

        def captured(*args, **kwargs):
            payload = args[0] if len(args) == 1 and not kwargs else None
            meta = _describe(payload) if payload is not None else {}
            if len(self.snapshots[p_id]) < self.max_samples:
                try:
                    self.snapshots[p_id].append(
                        pickle.dumps(
                            payload, protocol=pickle.HIGHEST_PROTOCOL
                        )
                    )
                except Exception:  # noqa: BLE001
                    self.errors[p_id] = self.errors.get(p_id, 0) + 1
            started = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            meta["ms"] = elapsed_ms
            self.calls[p_id].append(meta)
            # The driver stops the iterator as soon as it has enough samples
            # and then tears the worker down, so flush periodically instead of
            # only at the end of the epoch.
            self._calls_since_dump += 1
            if self._flush_every and self._calls_since_dump >= self._flush_every:
                self._calls_since_dump = 0
                try:
                    self.dump()
                except Exception:  # noqa: BLE001
                    pass
            return result

        captured._cedar_operator_capture = True  # type: ignore[attr-defined]
        return captured

    def attach(self, feature) -> None:
        """Wrap the mapper callables of every pipe of an executing plan."""
        for p_id, pipe in getattr(feature, "physical_pipes", {}).items():
            variant = getattr(pipe, "pipe_variant", None)
            if variant is None or not hasattr(variant, "fn"):
                continue
            fn = getattr(variant, "fn", None)
            if fn is None or getattr(fn, "_cedar_operator_capture", False):
                continue
            variant.fn = self.wrap(int(p_id), fn)

    def dump(self) -> None:
        if not self._enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        summary: Dict[str, Any] = {
            "worker": self.worker,
            "pipes": {},
            "snapshot_errors": self.errors,
        }
        for p_id, records in self.calls.items():
            if not records:
                continue
            times = sorted(record["ms"] for record in records)
            reprs: Dict[str, int] = {}
            for record in records:
                key = "%s|%s|%s|%s" % (
                    record.get("type"),
                    record.get("dtype"),
                    record.get("shape"),
                    record.get("contiguous"),
                )
                reprs[key] = reprs.get(key, 0) + 1
            summary["pipes"][str(p_id)] = {
                "calls": len(records),
                "mean_ms": statistics.fmean(times),
                "median_ms": statistics.median(times),
                "p10_ms": times[int(0.1 * (len(times) - 1))],
                "p90_ms": times[int(0.9 * (len(times) - 1))],
                "representations": reprs,
            }
            snapshots = self.snapshots.get(p_id) or []
            if snapshots:
                target = self.directory / f"worker_{self.worker}_pipe_{p_id}.pkl"
                with target.open("wb") as handle:
                    pickle.dump(
                        {
                            "p_id": int(p_id),
                            "sizes": [len(s) for s in snapshots],
                            "snapshots": snapshots,
                        },
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
        target = self.directory / f"worker_{self.worker}_op_capture.json"
        target.write_text(json.dumps(summary, indent=1, sort_keys=True))
        logger.info("Operator capture written to %s", target)


def maybe_attach_operator_capture(feature, worker: int) -> Optional[OperatorCapture]:
    directory = os.environ.get("CEDAR_OP_CAPTURE_DIR")
    if not directory:
        return None
    capture = OperatorCapture(
        Path(directory),
        worker,
        int(os.environ.get("CEDAR_OP_CAPTURE_SAMPLES", "8")),
    )
    try:
        capture.attach(feature)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Operator capture disabled: %s", exc)
        return None
    logger.info("Operator capture enabled at worker %s", worker)
    return capture

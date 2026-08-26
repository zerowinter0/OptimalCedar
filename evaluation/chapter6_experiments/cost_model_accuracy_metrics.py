"""Scale-invariant metrics for comparing plan costs with runtimes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def pairwise_qerrors(
    candidates: list[dict[str, Any]], field: str
) -> tuple[list[float], int, int]:
    """Compare predicted and observed runtime ratios within one workload.

    Absolute service costs can differ from wall-clock time by one workload-wide
    scale. Taking plan ratios cancels that nuisance scale. A value of 1 is
    exact; 2 means the predicted relative slowdown is wrong by a factor of two.
    """
    scored = [item for item in candidates if field in item]
    values: list[float] = []
    ordered_correct = 0
    ordered_total = 0
    for left_idx, left in enumerate(scored):
        for right in scored[left_idx + 1 :]:
            left_cost = float(left[field])
            right_cost = float(right[field])
            left_time = float(left["mean_runtime_sec"])
            right_time = float(right["mean_runtime_sec"])
            if min(left_cost, right_cost, left_time, right_time) <= 0.0:
                continue
            predicted_ratio = left_cost / right_cost
            observed_ratio = left_time / right_time
            ratio_error = predicted_ratio / observed_ratio
            values.append(max(ratio_error, 1.0 / ratio_error))
            if not math.isclose(left_time, right_time, rel_tol=1e-12):
                ordered_total += 1
                if (left_cost < right_cost) == (left_time < right_time):
                    ordered_correct += 1
    return values, ordered_correct, ordered_total


def qerror_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"num_pairs": 0}
    array = np.asarray(values, dtype=float)
    return {
        "num_pairs": len(values),
        "geometric_mean": float(np.exp(np.mean(np.log(array)))),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "maximum": float(np.max(array)),
    }

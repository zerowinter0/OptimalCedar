"""Measure whether LLaVA CUDA compute depends on unused serialized metadata.

Only actual remote Ray actor timings can establish a zero byte slope for
CLIP/BLIP. The experiment holds images and text fixed while adding metadata;
it does not claim invariance to image count, image content, or token length.
"""

import math
import os
import pickle
import statistics

from cedar.pipes import PipeVariantType


def metadata_counterfactual(snapshots, padding_bytes=2048):
    if padding_bytes < 1:
        raise ValueError("padding_bytes must be positive")
    result = []
    for raw in snapshots:
        value = pickle.loads(raw)
        if not isinstance(value, dict) or "_cedar_unused_metadata_probe" in value:
            raise ValueError("CUDA metadata probe needs unmodified dict records")
        value["_cedar_unused_metadata_probe"] = "x" * padding_bytes
        result.append(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    return result


def assess_invariance(
    baseline_runs,
    padded_runs,
    base_bytes,
    padded_bytes,
    max_relative_change=0.1,
):
    """Fit kx+b on the metadata counterfactual and gate the zero-slope claim.

    The two conditions are the small- and large-byte strata of the same legal
    records, so the measured difference is the operator's byte slope. The
    acceptance gate only records whether the counterfactual can claim a
    metadata-invariant operator; the fitted coefficients are reported either
    way, because the optimizer prices every operator with the same equation.
    """
    all_runs = baseline_runs + padded_runs
    base = [float(row["mean_ms_per_sample"]) for row in baseline_runs]
    padded = [float(row["mean_ms_per_sample"]) for row in padded_runs]
    errors = [float(row["stderr_ms_per_sample"]) for row in all_runs]
    converged = all(
        row.get("adaptive_profile", {}).get("converged") is True
        for row in all_runs
    )
    if (
        not converged
        or len(base) < 2
        or len(base) != len(padded)
        or base_bytes <= 0.0
        or padded_bytes <= base_bytes * 1.5
        or any(not math.isfinite(v) or v <= 0 for v in base + padded)
        or any(not math.isfinite(e) or e < 0 for e in errors)
    ):
        return {"accepted": False, "reason": "insufficient_valid_contrast"}
    byte_ratio = padded_bytes / base_bytes
    base_mean = statistics.mean(base)
    padded_mean = statistics.mean(padded)
    uncertainty = max(2 * max(errors), max(base + padded) - min(base + padded))
    upper_relative_change = (
        abs(padded_mean - base_mean) + uncertainty
    ) / base_mean
    accepted = upper_relative_change <= max_relative_change
    slope = max(0.0, (padded_mean - base_mean) / (padded_bytes - base_bytes))
    intercept = max(0.0, base_mean - slope * base_bytes)
    reference_cost = slope * base_bytes + intercept
    return {
        "accepted": accepted,
        "reason": (
            "metadata_invariant" if accepted else "unresolved_or_size_sensitive"
        ),
        "mean_base_ms": base_mean,
        "mean_padded_ms": padded_mean,
        "byte_ratio": byte_ratio,
        "upper_relative_change": upper_relative_change,
        "affine_coefficients": {
            "k_ms_per_byte": slope,
            "b_ms": intercept,
            "x_reference_bytes": base_bytes,
            "fixed_fraction": (
                intercept / reference_cost if reference_cost > 0.0 else 1.0
            ),
            "method": "two_stratum_metadata_counterfactual",
            "slope_uncertainty_ms_per_byte": (
                uncertainty / (padded_bytes - base_bytes)
            ),
        },
    }


def assess_actor_placement(runs, driver_ip):
    """Require each independent trial to use the same remote physical GPU."""
    locations = []
    for row in runs:
        trial = row.get("actor_locations")
        if not isinstance(trial, list) or len(trial) != 1:
            return {"accepted": False, "reason": "missing_single_actor_location"}
        location = trial[0]
        if not isinstance(location, dict):
            return {"accepted": False, "reason": "invalid_actor_location"}
        node_ip = location.get("node_ip")
        gpu_ids = location.get("gpu_ids")
        if not node_ip or not isinstance(gpu_ids, list) or len(gpu_ids) != 1:
            return {"accepted": False, "reason": "missing_remote_gpu"}
        locations.append((node_ip, tuple(str(value) for value in gpu_ids)))
    accepted = bool(locations) and len(set(locations)) == 1 and (
        locations[0][0] != driver_ip
    )
    return {
        "accepted": accepted,
        "reason": "same_remote_gpu" if accepted else "actor_placement_mismatch",
        "node_ip": locations[0][0] if locations else None,
        "gpu_ids": list(locations[0][1]) if locations else [],
    }


def is_llava_image_text_filter(pipe):
    fn = getattr(pipe, "fn", None)
    return (
        fn is not None
        and type(fn).__module__ == "evaluation.pipelines.llava_pretrain.dj_operators"
        and type(fn).__name__
        in ("ImageTextSimilarityFilter", "ImageTextMatchingFilter")
    )


def has_llava_model_work(snapshot):
    """Return whether this captured record reaches a CLIP/BLIP model call."""
    from evaluation.pipelines.llava_pretrain.dj_operators import (
        _iter_image_text_chunks,
    )

    value = pickle.loads(snapshot)
    return isinstance(value, dict) and any(
        paths for _, paths in _iter_image_text_chunks(value)
    )


def profile_cuda_metadata_invariance(dataset, feature, reservoir):
    """ABBA timing inside one warmed remote actor on legal model inputs.

    The four conditions are replayed in sequence inside a single actor, so the
    counterfactual is not exposed to cross-actor startup, clock or cache
    variation. Remote placement is still verified after the fact.
    """
    import ray

    driver_ip = ray.util.get_node_ip_address()
    entries = {}
    for p_id, pipe in feature.logical_pipes.items():
        if not is_llava_image_text_filter(pipe):
            continue
        captured = reservoir.values_for(pipe.input_pipes[0].id)
        original = [raw for raw in captured if has_llava_model_work(raw)][:32]
        if not original:
            entries[str(p_id)] = {
                "accepted": False,
                "reason": "no_model_work_inputs",
            }
            continue
        padded = metadata_counterfactual(original)
        base_bytes = statistics.median(len(a) for a in original)
        padded_bytes = statistics.median(len(b) for b in padded)
        sequence = [
            ("original", original),
            ("padded", padded),
            ("padded", padded),
            ("original", original),
        ]
        trials = dataset._adaptive_operator_benchmark(
            pipe,
            original,
            PipeVariantType.RAY,
            1,
            min_duration=float(os.environ.get("CEDAR_CUDA_WORK_MIN_SEC", "2")),
            max_duration=float(os.environ.get("CEDAR_CUDA_WORK_MAX_SEC", "12")),
            target_rse=0.10,
            min_observations=30,
            record_actor_locations=True,
            snapshot_sequence=sequence,
        )
        if len(trials) != len(sequence):
            raise RuntimeError(
                "CUDA metadata counterfactual expected one timing per ABBA "
                f"condition, got {len(trials)}"
            )
        runs = {"original": [], "padded": []}
        for (label, _), stats in zip(sequence, trials):
            runs[label].append(stats)
        result = assess_invariance(
            runs["original"], runs["padded"], base_bytes, padded_bytes
        )
        placement = assess_actor_placement(
            runs["original"] + runs["padded"], driver_ip
        )
        if not placement["accepted"]:
            result["accepted"] = False
            result["reason"] = placement["reason"]
        result["actor_placement"] = placement
        result.update(
            {
                "method": "same_actor_abba_metadata_counterfactual",
                "operator": type(pipe.fn).__name__,
                "sample_count": len(original),
                "runs": runs,
                "padding_bytes": 2048,
                "work_signature": "same image paths and text; model invoked; metadata only",
            }
        )
        entries[str(p_id)] = result
    return {"schema_version": 1, "operators": entries}

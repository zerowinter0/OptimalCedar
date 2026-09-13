"""Evaluation rules shared by fitted DP and Cedar operator models."""
def affine_value(model, input_bytes):
    if model.get("extrapolation") == "clamp_to_measured_range":
        input_bytes = max(model["input_min_bytes"], min(model["input_max_bytes"], input_bytes))
    return max(0.0, float(model["k"]) * input_bytes + float(model["b"]))

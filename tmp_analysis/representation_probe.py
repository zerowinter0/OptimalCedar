"""Does one byte count buy the same operator cost in every input representation?

The layered profile fits ``cost(record) = k * input_bytes + b`` on the payloads
that reach an operator in the *declared* order.  When the DP reorders the
pipeline, the same operator can be handed the same record in a different
representation (a PIL image instead of a float tensor, or a uint8 tensor
instead of a float32 one).  This probe measures the real SimCLRv2 callables on
every representation they can legally receive and prints the serialized byte
size next to the measured cost, so the two can be compared directly.

Usage (inside the container):
  python -u tmp_analysis/representation_probe.py
"""

import json
import pickle
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import build_feature  # noqa: E402

torch.set_num_threads(1)

PIPE_NAMES = {
    1: "N_normalize",
    2: "B_blur",
    3: "G_grayscale",
    4: "J_jitter",
    5: "H_flip",
    6: "C_crop",
    7: "F_to_float",
}


def make_payloads():
    from PIL import Image

    generator = torch.Generator().manual_seed(20260924)
    float_rgb = torch.rand((3, 244, 244), generator=generator)
    float_gray = float_rgb.mean(dim=0, keepdim=True).contiguous()
    uint8_rgb = (float_rgb * 255.0).round().to(torch.uint8)
    uint8_gray = uint8_rgb[0:1].contiguous()
    pil_rgb = Image.fromarray(
        uint8_rgb.permute(1, 2, 0).numpy(), mode="RGB"
    )
    pil_gray = Image.fromarray(uint8_gray[0].numpy(), mode="L")
    return {
        "pil_rgb_244": pil_rgb,
        "pil_L_244": pil_gray,
        "tensor_f32_rgb_244": float_rgb,
        "tensor_f32_L_244": float_gray,
        "tensor_u8_rgb_244": uint8_rgb,
        "tensor_u8_L_244": uint8_gray,
        "tensor_f32_L_122": torch.nn.functional.interpolate(
            float_gray.unsqueeze(0), size=(122, 122), mode="bilinear",
            align_corners=False,
        ).squeeze(0),
        "tensor_u8_L_122": torch.nn.functional.interpolate(
            uint8_gray.unsqueeze(0).float(), size=(122, 122),
            mode="bilinear", align_corners=False,
        ).round().to(torch.uint8).squeeze(0),
    }


def time_callable(fn, value, repeats=5, calls=8):
    snapshot = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    rates = []
    for _ in range(repeats):
        duration = 0.0
        for _ in range(calls):
            fresh = pickle.loads(snapshot)
            start = time.perf_counter()
            fn(fresh)
            duration += time.perf_counter() - start
            del fresh
        rates.append(calls / max(duration, 1e-9))
    return 1000.0 / statistics.median(rates)


def main() -> int:
    feature = build_feature(batch_size=4)
    pipes = feature.logical_pipes
    payloads = make_payloads()
    sizes = {
        name: len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        for name, value in payloads.items()
    }
    print("payload sizes (pickle bytes):")
    for name, size in sorted(sizes.items(), key=lambda item: item[1]):
        print(f"  {name:22s} {size:>10d}")
    rows = []
    for pipe_id, op_name in PIPE_NAMES.items():
        fn = pipes[pipe_id].get_fused_callable()
        for payload_name, value in payloads.items():
            try:
                cost = time_callable(fn, value)
            except Exception as exc:  # noqa: BLE001
                print(f"{op_name:13s} {payload_name:22s} rejected: {exc}")
                continue
            rows.append(
                {
                    "operator": op_name,
                    "representation": payload_name,
                    "bytes": sizes[payload_name],
                    "ms": cost,
                }
            )
            print(
                f"{op_name:13s} {payload_name:22s} "
                f"{sizes[payload_name]:>10d} B  {cost:8.4f} ms"
            )
    out = ROOT / "tmp_analysis/representation_probe.json"
    out.write_text(json.dumps({"payload_bytes": sizes, "rows": rows}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Motivation figures for Chapter 3, drawn from the saved profile.

(a) operator service vs input size, normalized: one scaling law cannot fit all
(b) per-actor service vs stage width, normalized: width is not a free factor
(c) measured stage-boundary round trip vs payload: boundaries are not negligible
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path("/workspace/OptimalCedar")
PROFILE = ROOT / "outputs/simclrv2_scaling_20260911/profile.yaml"
OUT = ROOT / "outputs/simclrv2_scaling_20260911/figures"

NAMES = {
    0: "Batch",
    1: "Normalize",
    2: "GaussianBlur",
    3: "Grayscale",
    4: "ColorJitter",
    5: "Flip",
    6: "Crop",
    7: "ToFloat",
    8: "Reader",
}


def panel_a(ax, profile):
    base = profile["baseline"]["input_sizes"]
    models = profile["cm_model"]["operators"]
    for pid, model in sorted(models.items(), key=lambda kv: int(kv[0])):
        if not model.get("k"):
            continue
        ref = base.get(int(pid)) or base.get(pid)
        if not ref or not model.get("input_min_bytes"):
            continue
        lo = model["input_min_bytes"] / ref
        hi = (model.get("input_max_bytes") or ref) / ref
        xs = [lo * (hi / lo) ** (i / 40) for i in range(41)]
        ys = [
            (model["k"] * (x * ref) + model["b"])
            / (model["k"] * ref + model["b"])
            for x in xs
        ]
        ax.plot(xs, ys, marker="", linewidth=1.2)
        ax.annotate(
            NAMES.get(int(pid), str(pid)),
            (xs[-1], ys[-1]),
            fontsize=6,
            xytext=(2, 2),
            textcoords="offset points",
        )
    ax.plot([0.2, 6], [0.2, 6], "--", color="0.6", linewidth=0.9)
    ax.annotate("assumed by existing models", (3.0, 3.0), fontsize=6, color="0.4", rotation=25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("input size / profiled mean")
    ax.set_ylabel("operator service / profiled service")
    ax.set_title("(a) one scaling law cannot fit all operators", fontsize=8)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)


def panel_b(ax, profile):
    scaling = profile["physical_model"].get("scaling", {})
    for family in ("RAY", "SMP"):
        for pid, entry in sorted(
            (scaling.get(family) or {}).items(), key=lambda kv: int(kv[0])
        ):
            widths = entry.get("widths", {})
            points = []
            for width, timing in widths.items():
                mean = timing.get("mean_ms_per_sample")
                if mean:
                    points.append((int(width), mean))
            if len(points) < 2:
                continue
            points.sort()
            base = points[0][1]
            style = "-o" if family == "RAY" else "--s"
            ax.plot(
                [p[0] for p in points],
                [p[1] / base for p in points],
                style,
                markersize=3,
                linewidth=1.1,
                label=f"{family} {NAMES.get(int(pid), pid)}",
            )
    widths = [1, 2, 4, 8]
    ax.plot(widths, [1.0] * len(widths), ":", color="0.5", linewidth=1.0)
    ax.annotate("assumed: service/width", (2.2, 1.02), fontsize=6, color="0.4")
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("stage width (actors/processes)")
    ax.set_ylabel("per-actor service / width-1 service")
    ax.set_title("(b) per-actor service degrades with width", fontsize=8)
    ax.legend(fontsize=5.5, ncol=2)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)


def panel_c(ax, profile):
    boundary = profile["physical_model"]["boundary"]
    for variant, marker in (("RAY", "o"), ("SMP", "s")):
        entry = boundary.get(variant)
        if not entry:
            continue
        xs = entry["payload_bytes"]
        ys = [
            measurement["boundary_sec_per_sample"] * 1000
            for measurement in entry["measurements"]
        ]
        ax.plot(xs, ys, marker=marker, linewidth=1.1, label=f"{variant} identity stage")
    base = profile["baseline"]["input_sizes"]
    models = profile["cm_model"]["operators"]
    for pid in ("2", "4", "6"):
        model = models.get(pid) or {}
        ref = base.get(int(pid))
        if not ref or not model.get("k"):
            continue
        value = model["k"] * ref + model["b"]
        ax.axhline(value, color="0.5", linewidth=0.7, linestyle=":")
        ax.annotate(
            f"{NAMES[int(pid)]} compute {value:.1f} ms",
            (520, value),
            fontsize=5.5,
            color="0.35",
            va="bottom",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("payload bytes per record")
    ax.set_ylabel("boundary / service (ms per record)")
    ax.set_title("(c) stage boundaries are not negligible", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)


def main():
    profile = yaml.safe_load(PROFILE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    panel_a(axes[0], profile)
    panel_b(axes[1], profile)
    panel_c(axes[2], profile)
    fig.tight_layout()
    fig.savefig(OUT / "cost_model_gap.pdf")
    fig.savefig(OUT / "cost_model_gap.png", dpi=200)
    print("wrote", OUT / "cost_model_gap.pdf")


if __name__ == "__main__":
    main()

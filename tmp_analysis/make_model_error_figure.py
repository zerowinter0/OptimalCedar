"""Motivation figure: the three cost models on the three SimCLRv2 plans.

Same workload, same profile, W=8, 1000 records (per-record times are
seconds * 8 workers).  Estimates come from tmp_analysis/score_three_plans.py:

  plan (measured ms/rec)   Cedar model   Plumber-style model   PICO model
  Cedar's plan   16.42        9.28            9.97               16.50
  Plumber's plan  8.62       33.38            9.25                8.90
  PICO's plan     4.14        6.38            9.97                3.47

PICO's numbers use the concurrency-aware scoring
(``CEDAR_DP_CONCURRENCY_AWARE=1``): INPROCESS operators add up on the worker
core, an SMP stage pipelines next to it (max), and a Ray stage is submitted and
transported per record so it adds to the worker chain.

Rankings
  measured        : PICO 4.1  < Plumber 8.6  < Cedar 16.4
  Cedar model     : PICO 6.4  < Cedar 9.3    < Plumber 33.4   (inverts both)
  Plumber model   : Plumber 9.2 < PICO 10.0  = Cedar 10.0     (cannot separate)
  PICO model      : PICO 3.5  < Plumber 8.9  < Cedar 16.5     (winner correct)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from pathlib import Path  # noqa: E402

OUT = Path("/workspace/OptimalCedar/outputs/plumber_bench_20260912/figures")
PAPER = Path("/workspace/OptimalCedar/my_paper/69e75a0100d7b4afeb1cfc20/figures")

PLANS = [
    {"key": "pico", "label": "PICO's plan", "measured": 4.14},
    {"key": "plumber", "label": "Plumber-style\nplan", "measured": 8.62},
    {"key": "cedar", "label": "Cedar's plan", "measured": 16.42},
]
MODELS = [
    {
        "key": "cedar_model",
        "label": "Cedar model",
        "color": "#4C78A8",
        "estimates": {"pico": 6.38, "plumber": 33.38, "cedar": 9.28},
    },
    {
        "key": "plumber_model",
        "label": "Plumber-style model",
        "color": "#F58518",
        "estimates": {"pico": 9.97, "plumber": 9.25, "cedar": 9.97},
    },
    {
        "key": "pico_model",
        "label": "PICO model",
        "color": "#54A24B",
        "estimates": {"pico": 3.47, "plumber": 8.90, "cedar": 16.50},
    },
]
CLIP_AT = 22.0


def panel_a(ax):
    width = 0.24
    for index, plan in enumerate(PLANS):
        for offset, model in enumerate(MODELS):
            value = model["estimates"][plan["key"]]
            x = index + (offset - 1) * (width + 0.02)
            clipped = value > CLIP_AT
            height = min(value, CLIP_AT)
            ax.bar(x, height, width=width, color=model["color"],
                   edgecolor="black", linewidth=0.5)
            if clipped:
                # mark the bar as cut off; the label carries the true value
                ax.plot([x - width / 2, x + width / 2], [CLIP_AT - 0.5] * 2,
                        color="white", linewidth=0.9)
                ax.text(x, CLIP_AT + 0.4, f"{value:.1f}",
                        ha="center", fontsize=5.6, color="#B22222")
            else:
                ax.text(x, height + 0.8, f"{value:.1f}", ha="center",
                        fontsize=5.6)
        ax.plot([index - 0.44, index + 0.44], [plan["measured"]] * 2,
                color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(PLANS)))
    ax.set_xticklabels(
        [f"{p['label']}\n(measured {p['measured']:.1f} ms/rec)" for p in PLANS],
        fontsize=5.8,
    )
    ax.set_ylabel("model estimate (ms per record per worker)", fontsize=7)
    ax.set_xlim(-0.62, 3.05)
    ax.set_ylim(0, CLIP_AT + 3.2)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_title("(a) what each cost model says the three plans cost",
                 fontsize=7.5)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=model["color"], edgecolor="black")
        for model in MODELS
    ] + [Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black")]
    ax.legend(
        handles,
        [model["label"] for model in MODELS] + ["measured (dashed)"],
        fontsize=5.0, loc="upper center", bbox_to_anchor=(0.5, -0.28),
        frameon=False, ncol=2,
    )
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)


def panel_b(ax):
    """Ranking strips: does the model pick the plan that is actually fastest?"""
    rows = [
        {
            "label": "measured",
            "order": [("pico", 4.14), ("plumber", 8.62), ("cedar", 16.42)],
            "ok": True,
        },
        {
            "label": "Cedar model",
            "order": [("pico", 6.38), ("cedar", 9.28), ("plumber", 33.38)],
            "ok": False,
        },
        {
            "label": "Plumber-style model",
            "order": [("plumber", 9.25), ("pico", 9.97), ("cedar", 9.97)],
            "ok": False,
        },
        {
            "label": "PICO model",
            "order": [("pico", 3.47), ("plumber", 8.90), ("cedar", 16.50)],
            "ok": True,
        },
    ]
    names = {"pico": "PICO", "plumber": "Plumber", "cedar": "Cedar"}
    colors = {"pico": "#54A24B", "plumber": "#F58518", "cedar": "#4C78A8"}
    for index, row in enumerate(rows):
        y = len(rows) - index
        for position, (plan, value) in enumerate(row["order"]):
            face = colors[plan] if row["label"] != "measured" else "#E8E8E8"
            ax.add_patch(
                Rectangle(
                    (position * 1.0, y - 0.32), 0.92, 0.62,
                    facecolor=face, edgecolor="black", linewidth=0.5,
                    hatch=None if row["label"] != "measured" else "//",
                )
            )
            ax.text(position * 1.0 + 0.46, y, f"{names[plan]}\n{value:.1f}",
                    ha="center", va="center", fontsize=5.0)
        mark = "OK" if row["ok"] else "wrong winner"
        ax.text(3.15, y, mark, fontsize=5.4, va="center",
                color="#2E7D32" if row["ok"] else "#B22222")
        ax.text(-0.12, y, row["label"], fontsize=5.6, va="center", ha="right")
    ax.set_xlim(-2.1, 4.3)
    ax.set_ylim(-1.6, len(rows) + 0.75)
    ax.axis("off")
    ax.set_title("(b) which plan each model thinks is fastest", fontsize=7.5)
    ax.text(
        -2.1, -0.15,
        "Cedar's model ranks its own offload plan above the\n"
        "Plumber-style plan (9.3 vs 33.4) although it is\n"
        "measured 1.9x slower. The Plumber-style model\n"
        "separates the three plans by at most 8% while\n"
        "they differ 4.0x in reality. Ours is within 0.5% on\n"
        "Cedar's own plan (16.5 vs 16.4 measured) and keeps\n"
        "the measured ordering.",
        fontsize=4.7, color="#333333", va="top", ha="left",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.4, 3.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.16)
    panel_a(fig.add_subplot(grid[0, 0]))
    panel_b(fig.add_subplot(grid[0, 1]))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.28)
    fig.savefig(OUT / "simclr_model_estimates.pdf")
    fig.savefig(OUT / "simclr_model_estimates.png", dpi=220)
    if PAPER.is_dir():
        fig.savefig(PAPER / "simclr_model_estimates.pdf")
        fig.savefig(PAPER / "simclr_model_estimates.png", dpi=220)
    print("wrote", OUT / "simclr_model_estimates.pdf")


if __name__ == "__main__":
    main()

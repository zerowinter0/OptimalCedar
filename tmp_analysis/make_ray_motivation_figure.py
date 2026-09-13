"""Motivation figure: where one record's time actually goes, per plan.

SimCLRv2, 1000 records, W=8 local workers, 64 local CPUs + one 64-CPU Ray node,
one shared layered profile, batch 4, three repeats
(outputs/plumber_bench_20260912).

Panel (a): measured end-to-end time of the same 1000 records
  Cedar 2.053 s, Plumber-style 1.078 s, PICO 0.517 s.

Panel (b): measured milliseconds per record per worker (= seconds * workers /
records), split into four parts. A plan assigns operators to stages; a parallel
stage is instantiated as w identical copies of that stage (w processes for SMP,
w actors for a Ray stage - called a "lane" inside the DP optimizer), and every
operator without such a stage runs sequentially inside the worker process.

  parallel stage   worker-side service time the record spends inside the
                   parallel/offloaded stage of that plan (measured by the
                   per-stage service timers): Cedar Ray stage 1.75 ms (model
                   charges 1.15 = compute / 7 actors), Plumber-style widest
                   SMP stage 6.22 ms (model charges 4.50), PICO fused stage
                   3.02 ms (model charges 1.93 = compute / 6 processes).
  stage boundary   marshalling / cross-node round trip of that stage
                   (measured identity round trip 5.23 ms for the Ray stage;
                   per-stage boundary terms 1.93 / 1.11 ms for the SMP stages).
  worker operators operator work that is on the critical path but has no
                   parallel stage: Cedar leaves reader, grayscale, crop and
                   batcher in the worker (7.11 ms at width-1 cost). For the two
                   single-host plans those operators are hidden behind the
                   parallel stages in steady state (each has its own thread),
                   so they contribute ~0 to the per-record time, but they still
                   bound what the plan can reach: as a plain chain they cost
                   24.8 ms (Plumber-style) and 13.0 ms (PICO, after reordering).
  waiting          remainder of the measured per-record time (queued behind a
                   saturated stage). Cedar 2.33 ms, Plumber-style 0.47 ms.

Sanity check: every bar sums to the measured per-record time
  Cedar 1.75 + 5.23 + 7.11 + 2.33 = 16.42 ms  (2.053 s)
  Plumber-style 6.22 + 1.93 + 0.00 + 0.47 = 8.62 ms  (1.078 s)
  PICO 3.02 + 1.11 + 0.00 + 0.00 = 4.13 ms  (0.517 s)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from pathlib import Path  # noqa: E402

OUT = Path("/workspace/OptimalCedar/outputs/plumber_bench_20260912/figures")
PAPER = Path("/workspace/OptimalCedar/my_paper/69e75a0100d7b4afeb1cfc20/figures")

PLANS = [
    {
        "name": "Cedar",
        "sub": "5/9 ops Ray x7",
        "seconds": 2.053,
        "busy": "remote",
        "lane": 1.75,
        "boundary": 5.23,
        "unlaned": 7.11,
        "waiting": 2.33,
        "model": "charges only the offloaded\ncompute (1.2 of 16.4 ms)",
    },
    {
        "name": "Plumber-style",
        "sub": "2/9 ops SMP x3",
        "seconds": 1.078,
        "busy": "local",
        "lane": 6.22,
        "boundary": 1.93,
        "unlaned": 0.00,
        "waiting": 0.47,
        "model": "prices only the 2 operators it\nwidens; 7 get no stage",
    },
    {
        "name": "PICO",
        "sub": "7/9 fused SMP x6",
        "seconds": 0.517,
        "busy": "local",
        "lane": 3.02,
        "boundary": 1.11,
        "unlaned": 0.00,
        "waiting": 0.00,
        "model": "prices the fused stage, its\nboundary and reader/batcher",
    },
]

COLORS = {
    "lane": "#4C78A8",
    "boundary": "#E45756",
    "unlaned": "#F58518",
    "waiting": "#B279A2",
}


def per_record(plan):
    """Measured ms per record per worker for one plan."""
    return plan["seconds"] * 1000.0 / (1000.0 / 8.0)


def panel_a(ax):
    width = 0.32
    for index, plan in enumerate(PLANS):
        for offset, family in ((-width / 1.8, "local"), (width / 1.8, "remote")):
            busy = plan["busy"] == family
            ax.bar(index + offset, plan["seconds"], width=width,
                   color="#4C78A8" if busy else "#E8E8E8",
                   edgecolor="black", linewidth=0.6,
                   hatch="//" if busy else None)
        ax.text(index, plan["seconds"] + 0.055, f"{plan['seconds']:.3f} s",
                ha="center", fontsize=6.0)
    ax.set_xticks(range(len(PLANS)))
    ax.set_xticklabels([f"{p['name']}\n{p['sub']}" for p in PLANS], fontsize=5.4)
    ax.set_ylabel("execution time (s)", fontsize=7)
    ax.set_ylim(0, 2.45)
    ax.tick_params(labelsize=6.5)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#4C78A8", edgecolor="black", hatch="//"),
        Rectangle((0, 0), 1, 1, facecolor="#E8E8E8", edgecolor="black"),
    ]
    ax.legend(handles,
              ["server carrying the bottleneck", "idle server"],
              fontsize=5.2, loc="upper right", frameon=False)
    ax.set_title("(a) the same 1000 records, three plans", fontsize=7.5)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)


def panel_b(ax):
    width = 0.46
    order = ["lane", "boundary", "unlaned", "waiting"]
    for index, plan in enumerate(PLANS):
        bottom = 0.0
        for key in order:
            value = plan[key]
            if value <= 0:
                continue
            ax.bar(index, value, width=width, bottom=bottom,
                   color=COLORS[key], edgecolor="black", linewidth=0.5,
                   hatch="//" if key in ("boundary", "waiting") else None)
            if value >= 0.9:
                ax.text(index, bottom + value / 2, f"{value:.1f}",
                        ha="center", va="center", fontsize=5.6,
                        color="white" if key == "lane" else "black",
                        bbox=(dict(facecolor="white", edgecolor="none",
                                   alpha=0.7, pad=0.8)
                              if key in ("boundary", "waiting") else None))
            bottom += value
        ax.text(index, bottom + 0.35, f"measured {bottom:.1f} ms",
                ha="center", fontsize=5.6)
        ax.text(index, -2.3, plan["model"], ha="center", va="top",
                fontsize=4.8, color="#333333")
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=COLORS["lane"], edgecolor="black"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["boundary"], edgecolor="black",
                  hatch="//"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["unlaned"], edgecolor="black"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["waiting"], edgecolor="black",
                  hatch="//"),
    ]
    ax.legend(
        handles,
        ["inside the parallel / offloaded stage",
         "stage boundary or cross-node round trip",
         "operators left in the worker",
         "waiting behind a saturated stage"],
        fontsize=4.9, loc="upper center", bbox_to_anchor=(0.5, -0.36),
        frameon=False, ncol=2,
    )
    ax.set_xticks(range(len(PLANS)))
    ax.set_xticklabels([p["name"] for p in PLANS], fontsize=6.2)
    ax.set_ylabel("ms per record per worker", fontsize=7)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 19.0)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_title("(b) where one record's time goes (measured)", fontsize=7.5)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for plan in PLANS:
        total = sum(plan[k] for k in ("lane", "boundary", "unlaned", "waiting"))
        print(f"{plan['name']:<14} parts={total:6.2f} ms "
              f"measured={per_record(plan):6.2f} ms")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    panel_a(axes[0])
    panel_b(axes[1])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.34,
                        wspace=0.22)
    fig.savefig(OUT / "simclr_ray_two_server_motivation.pdf")
    fig.savefig(OUT / "simclr_ray_two_server_motivation.png", dpi=220)
    if PAPER.is_dir():
        fig.savefig(PAPER / "simclr_ray_two_server_motivation.pdf")
        fig.savefig(PAPER / "simclr_ray_two_server_motivation.png", dpi=220)
    print("wrote", OUT / "simclr_ray_two_server_motivation.pdf")


if __name__ == "__main__":
    main()

"""Turn the reconciliation runs into the Chapter 3 calibration artifacts.

Outputs (next to the profile):
  calibration_table.csv   per-stage model terms vs measured service/residence
  calibration_parity.pdf  plan-level parity and per-stage service/queue split
"""
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/workspace/OptimalCedar")
RUN = ROOT / "outputs/simclrv2_scaling_20260911"
RECON = ROOT / "tmp_analysis/reconcile"
OUT = RUN / "figures"
SAMPLES = 9469
WORKERS = 8

PLANS = [
    ("dp", "newdp", 7.427),
    ("old_dp", "newold", 10.313),
    ("cm/cedar", "newcm", 19.354),
]

# Plan-level scores measured by the probes (ms per worker per sample).
MODEL_UNCALIBRATED = {"dp": 4.863, "old_dp": 7.874, "cm/cedar": 17.551}
MODEL_CALIBRATED = {"dp": 5.5832, "old_dp": 8.5941, "cm/cedar": 18.2707}


def rows():
    data = []
    for plan, tag, total in PLANS:
        measured = json.loads((RECON / f"{tag}.measured.json").read_text())
        model = json.loads((RECON / f"{tag}.model.json").read_text())
        stats = measured.get("stage_statistics", {})
        services = measured.get("service_stats", {})
        for row, pid in zip(model["rows"], model["stages"]):
            name = row["variant"]
            width = int(row["width"])
            stage = stats.get(str(pid), {})
            service = services.get(str(pid)) or {}
            service_ms = service.get("mean_ms_per_sample")
            per_record_service = (
                service_ms / width if service_ms is not None else None
            )
            data.append(
                {
                    "plan": plan,
                    "tag": tag,
                    "pipe": pid,
                    "ops": "-".join(str(o) for o in row["ops"]),
                    "variant": name,
                    "width": width,
                    "model_compute_per_record": round(
                        row["compute"] / (width if name != "INPROCESS" else 1), 3
                    ),
                    "model_boundary_local": round(row["boundary_local"], 3),
                    "model_boundary_external": round(row["boundary_external"], 3),
                    "model_stage_total": round(row["lane_total"], 3),
                    "measured_residence_median": round(
                        stage.get("median_ms", float("nan")), 3
                    ),
                    "measured_service_per_record": (
                        round(per_record_service, 3)
                        if per_record_service is not None
                        else None
                    ),
                    "service_ratio": (
                        round(per_record_service / (row["compute"] / width), 3)
                        if per_record_service is not None
                        else None
                    ),
                    "queue_residence": (
                        round(stage.get("median_ms", 0.0) - per_record_service, 3)
                        if per_record_service is not None
                        else None
                    ),
                    "plan_total_sec": total,
                }
            )
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = rows()
    with (OUT / "calibration_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)

    measured_plan = {plan: WORKERS * total / SAMPLES * 1000 for plan, _, total in PLANS}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ax = axes[0]
    for label, values, marker in (
        ("uncalibrated", MODEL_UNCALIBRATED, "o"),
        ("calibrated", MODEL_CALIBRATED, "s"),
    ):
        xs = [values[plan] for plan, _, _ in PLANS]
        ys = [measured_plan[plan] for plan, _, _ in PLANS]
        ax.scatter(xs, ys, marker=marker, label=label, s=45)
        for plan, x, y in zip((p[0] for p in PLANS), xs, ys):
            ax.annotate(plan, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7)
    lo, hi = 3.0, 20.0
    ax.plot([lo, hi], [lo, hi], "--", color="0.6", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("model objective $J(P)$ (ms/record/worker)")
    ax.set_ylabel("measured (ms/record/worker)")
    ax.set_title("Plan-level parity")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)

    ax = axes[1]
    labels, services, queues = [], [], []
    for row in data:
        if row["measured_service_per_record"] is None:
            continue
        labels.append(f"{row['plan']}\n{row['ops']} ({row['variant'][:3].lower()}"
                      f"×{row['width']})")
        services.append(row["measured_service_per_record"])
        queues.append(max(0.0, row["queue_residence"] or 0.0))
    index = range(len(labels))
    ax.bar(index, services, color="#4C78A8", label="worker-side service")
    ax.bar(index, queues, bottom=services, color="#F58518", alpha=0.7, label="queue residency")
    for position, row in zip(index, [r for r in data if r["measured_service_per_record"]]):
        ax.plot(
            [position - 0.4, position + 0.4],
            [row["model_stage_total"]] * 2,
            color="black",
            linewidth=1.2,
        )
    ax.set_yscale("log")
    ax.set_xticks(list(index))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("ms per record")
    ax.set_title("Per-stage service vs queue (black line = model)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT / "calibration_parity.pdf")
    fig.savefig(OUT / "calibration_parity.png", dpi=200)
    print("wrote", OUT / "calibration_table.csv")
    print("wrote", OUT / "calibration_parity.pdf")
    for plan, _, total in PLANS:
        per_worker = WORKERS * total / SAMPLES * 1000
        print(
            f"  {plan:<9} model(cal)={MODEL_CALIBRATED[plan]:6.2f} "
            f"measured={per_worker:6.2f} ratio={per_worker / MODEL_CALIBRATED[plan]:.2f}"
        )


if __name__ == "__main__":
    main()

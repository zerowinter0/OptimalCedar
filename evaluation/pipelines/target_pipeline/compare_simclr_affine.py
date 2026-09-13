"""Refit SimCLRv2 local sweep with the optimizer's nonnegative affine fitter."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from cedar.client.linear_cost_profile import fit_affine

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs/target_pipeline_scaling_20260910/results"
OUT = ROOT / "outputs/simclrv2_affine_comparison_20260911"
CM = ROOT / "outputs/cm_vs_cedar_simclr_20260911/profile.yaml"
DP = ROOT / "outputs/dp_vs_old_dp_simclrv2_20260911/profile.yaml"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE / "raw.csv")
    data = raw[(raw.workload == "simclr") & (raw.backend == "local")].copy()
    assert len(data) == 9 * 12 * 3
    assert not data.duplicated(["operator_index", "round", "point"]).any()
    assert np.allclose(data.ms_per_record,
                       1000 * data.seconds / (data.calls * data.records_per_call))
    cm = yaml.safe_load(CM.read_text())["cm_model"]["operators"]
    dp = yaml.safe_load(DP.read_text())["cm_model"]["operators"]
    assert cm == dp, "DP and CM profiles differ; inspect before overlaying"
    ops = json.loads((SOURCE / "operators.json").read_text())
    rows, models = [], {}
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for index, ax in enumerate(axes.flat):
        op = ops[index]
        assert op["workload"] == "simclr"
        observed = data[data.operator_index == index].sort_values(["point", "round"])
        fit = fit_affine(list(zip(observed.input_bytes, observed.ms_per_record * 1e6)))
        # Logical IDs are reversed relative to execution order in this feature.
        pid = 8 - index
        current = cm[pid]
        if not op["batch"]:
            assert current["pipe_name"] == op["name"]
        else:
            assert current["pipe_name"].startswith("BatcherPipe")
        models[pid] = dict(fit, pipe_name=op["name"], tag=op["tag"])
        mean = observed.groupby("point").agg(
            x=("input_bytes", "mean"), y=("ms_per_record", "mean"))
        k_mib = fit["k"] * 1024**2
        current_k_mib = current["k"] * 1024**2
        lo, hi = mean.x.min(), mean.x.max()
        x = np.linspace(lo, hi, 200)
        ax.plot(mean.x / 1024**2, mean.y, "o", ms=4, label="Sweep: 3-round mean")
        ax.plot(x / 1024**2, fit["k"] * x + fit["b"], label="Sweep affine fit")
        comparable = not op["reader"] and not op["batch"]
        if comparable:
            ax.plot(x / 1024**2, current["k"] * x + current["b"],
                    "--", label="Current DP = CM (extrapolated)")
            a, b = current["input_min_bytes"], current["input_max_bytes"]
            # Mark only the overlap of the natural input range with the sweep.
            if a == b:
                ax.axvline(a / 1024**2, color="gray", linestyle=":", lw=1)
            elif max(a, lo) <= min(b, hi):
                ax.axvspan(max(a, lo) / 1024**2, min(b, hi) / 1024**2,
                           color="gray", alpha=.08)
        title = op["name"].replace("MapperPipe_", "")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Input size (MiB; linear)")
        ax.set_ylabel("Time per record (ms)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.text(.02, .98, f"Sweep: k={k_mib:.4g} ms/MiB, b={fit['b']:.4g} ms\n"
                f"R²={fit.get('r_squared', float('nan')):.3f}" +
                ("\nNot directly comparable" if not comparable else ""),
                transform=ax.transAxes, va="top", fontsize=8)
        note = ("File bytes (sweep) vs path-object bytes (profile); PNG vs JPEG inputs"
                if op["reader"] else "Batch size 4 (sweep) vs 1 (profile); timing method differs"
                if op["batch"] else "Same callable; controlled sweep vs untouched natural inputs")
        rows.append(dict(operator=title, pipe_id=pid,
            sweep_k_ms_per_byte=fit["k"], sweep_k_ms_per_mib=k_mib,
            sweep_b_ms=fit["b"], sweep_r2=fit.get("r_squared"),
            sweep_min_mib=lo/1024**2, sweep_max_mib=hi/1024**2,
            current_dp_k_ms_per_byte=current["k"],
            current_cm_k_ms_per_byte=current["k"],
            current_dp_cm_k_ms_per_mib=current_k_mib,
            current_dp_cm_b_ms=current["b"],
            current_r2=current.get("r_squared"),
            current_status=current["status"],
            current_min_mib=current["input_min_bytes"]/1024**2,
            current_max_mib=current["input_max_bytes"]/1024**2,
            directly_comparable=comparable, note=note))
    handles, labels = axes.flat[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("SimCLRv2: wide-range local sweep vs current DP / CM affine models", fontsize=14)
    fig.tight_layout(rect=(0, .055, 1, .96))
    fig.savefig(OUT / "comparison.png", dpi=170)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "coefficients.csv", index=False)
    data.to_csv(OUT / "local_measurements.csv", index=False)
    (OUT / "wide_range_models.yaml").write_text(yaml.safe_dump(dict(
        schema_version=1, equation="ms_per_record = k * input_bytes + b",
        profile_backend="INPROCESS", input_policy="controlled_size_sweep",
        directly_installable=False, operators=models)))
    metadata = dict(source=str(SOURCE), cm_profile=str(CM), dp_profile=str(DP),
        profiles_identical=CM.read_bytes()==DP.read_bytes(),
        cm_sha256=hashlib.sha256(CM.read_bytes()).hexdigest(),
        dp_sha256=hashlib.sha256(DP.read_bytes()).hexdigest(),
        observations=324, rounds=3, sizes=12,
        fit="Same fit_affine: 12 equal-width bins, nonnegative least squares",
        caveats=["Controlled inputs may not be reachable after the original prefix.",
                 "Reader byte definitions and file formats differ.",
                 "Batcher batch sizes and timing methods differ.",
                 "Gray shading / dotted line marks natural-profile input support, not uncertainty.",
                 "Coefficients are analysis artifacts; active optimizer profiles are unchanged."])
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    lines = ["# SimCLRv2 affine coefficient comparison", "",
        "t = k*x+b; x in MiB, t in ms/record. DP and CM currently share identical coefficients.",
        "Fits use the same nonnegative affine fitter on 12 linearly spaced size targets × 3 local rounds.",
        "", "| Operator | Sweep k | Sweep b | Sweep R² | Current DP=CM k | Current b | Current status |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['operator']} | {r['sweep_k_ms_per_mib']:.6g} | {r['sweep_b_ms']:.6g} | "
                     f"{r['sweep_r2']:.4f} | {r['current_dp_cm_k_ms_per_mib']:.6g} | "
                     f"{r['current_dp_cm_b_ms']:.6g} | {r['current_status']} |")
    lines += ["", "Reader and Batcher coefficients are not directly comparable; see coefficients.csv.",
              "The natural profile gives fixed-size fallback for Flip/Jitter/Grayscale/Blur/Normalize/Batcher.",
              "The sweep profiles hypothetical input sizes; this is not evidence every size is reachable.",
              "", "![comparison](comparison.png)"]
    (OUT / "README.md").write_text("\n".join(lines)+"\n")
    print(pd.DataFrame(rows)[["operator", "sweep_k_ms_per_mib", "sweep_b_ms",
                              "sweep_r2", "current_dp_cm_k_ms_per_mib", "current_dp_cm_b_ms"]].to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()

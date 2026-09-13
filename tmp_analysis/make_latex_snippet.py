"""Emit the Chapter 3 calibration figure/table/prose in the paper's style."""
import csv
import json
import shutil
from pathlib import Path

ROOT = Path("/home/xieruiyang/OptimalCedar")
RUN = ROOT / "outputs/simclrv2_scaling_20260911"
PAPER = ROOT / "my_paper/69e75a0100d7b4afeb1cfc20"
SECTION = PAPER / "section/03_calibration_layer.tex"
FIGURE = PAPER / "figures/pico_calibration_reconciliation.pdf"

PLAN_LABEL = {"dp": "PICO", "old_dp": "DP-Cedar", "cm/cedar": "Cedar-CM"}
OP_NAME = {
    "0": "Batch",
    "1": "Norm",
    "2": "Blur",
    "3": "Gray",
    "4": "Jitter",
    "5": "Flip",
    "6": "Crop",
    "7": "ToFloat",
    "8": "Reader",
}
PLAN_ORDER = ["dp", "old_dp", "cm/cedar"]
MODEL_UNCALIBRATED = {"dp": 4.86, "old_dp": 7.87, "cm/cedar": 17.55}
MODEL_CALIBRATED = {"dp": 5.58, "old_dp": 8.59, "cm/cedar": 18.27}
MEASURED = {"dp": 6.27, "old_dp": 8.71, "cm/cedar": 16.35}


def load_rows():
    path = RUN / "figures/calibration_table.csv"
    return list(csv.DictReader(path.open()))


def stage_table(rows):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-stage reconciliation of the cost model against one "
        r"executed plan per optimizer (SimCLR-v2, eight local workers, "
        r"9469 samples). \emph{Model} is the stage service predicted by "
        r"Section~\ref{sec:cost-model} at the plan's own width; "
        r"\emph{service} is the worker-side time measured inside the stage "
        r"during that execution; \emph{residence} is the wall-clock time a "
        r"record spends in the stage, and \emph{queue} is their difference. "
        r"Local operators are measured through residence because they run "
        r"synchronously in the worker.}",
        r"\label{tab:calibration-reconciliation}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Plan & Stage & $w$ & Model & Bnd.\ local & Service & Svc./model "
        r"& Residence & Queue \\",
        r"\midrule",
    ]
    for plan in PLAN_ORDER:
        entries = [r for r in rows if r["plan"] == plan]
        for index, row in enumerate(entries):
            stage = r"$\rightarrow$".join(
                OP_NAME.get(part, part) for part in row["ops"].split("-")
            )
            if len(row["ops"].split("-")) == 1:
                stage = OP_NAME.get(row["ops"], row["ops"])
            variant = row["variant"].replace("INPROCESS", "Local")
            service = row["measured_service_per_record"]
            ratio = row["service_ratio"]
            queue = row["queue_residence"]
            name = PLAN_LABEL[plan] if index == 0 else ""
            lines.append(
                " & ".join(
                    [
                        name,
                        f"{stage} ({variant})",
                        row["width"],
                        row["model_stage_total"],
                        row["model_boundary_local"],
                        service if service else "--",
                        ratio if ratio else "--",
                        row["measured_residence_median"],
                        queue if queue else "--",
                    ]
                )
                + r" \\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    return lines


def parity_table():
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Plan-level accuracy of the cost-model objective on the "
        r"three executed plans of Table~\ref{tab:calibration-reconciliation}. "
        r"Values are milliseconds per source record per worker. Calibration "
        r"uses only the residual introduced by co-running the other stages of "
        r"the same plan.}",
        r"\label{tab:calibration-parity}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Plan & Measured & Uncalibrated & Calibrated & Cal./meas. \\",
        r"\midrule",
    ]
    for plan in PLAN_ORDER:
        lines.append(
            "{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
                PLAN_LABEL[plan],
                MEASURED[plan],
                MODEL_UNCALIBRATED[plan],
                MODEL_CALIBRATED[plan],
                MODEL_CALIBRATED[plan] / MEASURED[plan],
            )
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return lines


PROSE = r"""
\subsection{Calibration and Reconciliation}
\label{sec:cost-model-calibration}

The stage-service terms of Section~\ref{sec:mixed-pipeline-cost} are measured
in isolation, but they execute next to every other stage of the materialized
plan. We therefore close the loop between model and execution: every parallel
stage records its worker-side service time and every pipe records the
wall-clock time a record spends inside it, so each modeled term can be compared
with the run that produced it. Table~\ref{tab:calibration-reconciliation}
reports this reconciliation for one executed plan per optimizer.

Two properties of the measurement matter. First, residence and service are
different quantities. A saturated parallel stage accumulates a queue, and its
residence is dominated by that queue ($4.6$--$1315$\,ms in our runs) while the
service that the plan actually pays for is $0.2$--$6.0$\,ms. By Little's law the
queue wait is amortized across in-flight records and must not be charged as
work; accordingly, our objective prices service and boundary terms only, and
the plan-level model error stays within $12\%$ once the residual is
calibrated (Table~\ref{tab:calibration-parity}, Figure~\ref{fig:calibration}).
Second, the residual is not a global constant. Compute-bound operators match
their isolated profile, I/O-bound operators inflate by $1.3$--$1.7\times$ when
eight workers share the storage path, and the same fused block can be
over-predicted or under-predicted depending on the order inside it: our two
six-operator SMP blocks differ by $26\%$ in the model but by $2.25\times$ in
measured service, because moving the crop ahead of grayscale changes the data
each operator sees.

The optimizer consumes the calibrated residuals from the profile
(\texttt{calibration.co\_run\_factors}, \texttt{calibration.stage\_factors}),
so a re-profile automatically tightens the model without changing the search.
We report the residual as a calibration artifact rather than hiding it inside
the fitted coefficients, which keeps the model auditable and lets the
evaluation distinguish ranking quality from absolute accuracy.
"""


def main():
    rows = load_rows()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUN / "figures/calibration_parity.pdf", FIGURE)
    body = [
        "% Generated by tmp_analysis/make_latex_snippet.py",
        "% Requires \\usepackage{booktabs} and the figures/ directory.",
        PROSE.strip(),
        "",
        r"\begin{figure*}[t]",
        r"    \centering",
        r"    \includegraphics[width=\textwidth]{figures/pico_calibration_reconciliation.pdf}",
        r"    \caption{Left: modeled plan objective against measured execution "
        r"for the three executed plans of Table~\ref{tab:calibration-reconciliation}, "
        r"before and after residual calibration. Right: per-stage worker-side "
        r"service (blue) and the queue residency that the plan does not pay for "
        r"(orange); the black rule marks the modeled stage service.}",
        r"    \Description{The left panel plots modeled against measured cost per "
        r"plan and shows the calibrated points closer to the diagonal. The right "
        r"panel shows stacked bars of measured service and queue residency per "
        r"stage with a line for the model.}",
        r"    \label{fig:calibration}",
        r"\end{figure*}",
        "",
    ]
    body += stage_table(rows)
    body += parity_table()
    SECTION.write_text("\n".join(body) + "\n")
    print("wrote", SECTION)
    print("wrote", FIGURE)


if __name__ == "__main__":
    main()

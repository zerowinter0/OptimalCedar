"""Reconcile measured plan throughput with per-operator profile data.

For each recorded plan of a workload, sum the per-operator measurements the
profile already contains (local service, backend service, cross-host identity
round trip) into a per-record latency, then compare with the measured rate.
"""

import argparse
import json

import yaml

NAME = {
    "MapperPipe__read": "read",
    "MapperPipe__resample": "resample",
    "MapperPipe__spec": "spec",
    "MapperPipe__stretch": "stretch",
    "MapperPipe_time_mask": "time_mask",
    "MapperPipe_frequency_mask": "freq_mask",
    "MapperPipe_mel": "mel",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--block-payload", type=float, default=0.55)
    args = ap.parse_args()

    prof = yaml.safe_load(open(args.profile))
    base = prof["baseline"]
    lat = {int(k): float(v) / 1e6 for k, v in base["latencies"].items()}
    sizes = {int(k): float(v) for k, v in base["input_sizes"].items()}
    iso = (prof.get("layered_profile") or {}).get("isolated_operator_costs") or {}
    boundary = (prof.get("physical_model") or {}).get("object_boundary") or {}
    ray_b = boundary.get("RAY", {}).get("operators", {})
    smp_b = boundary.get("SMP", {}).get("operators", {})

    def entry(table, pid):
        return table.get(pid) or table.get(str(pid)) or {}

    payload_mb = args.block_payload
    print(f"# payload assumption {payload_mb:.2f} MB/record (plan log)")

    results = json.load(open(args.results))
    for run in results.get("runs", []):
        plans = run.get("physical_plans_by_feature") or {}
        if not plans:
            continue
        plan = next(iter(plans.values()))
        pipes = plan["pipes"]
        nw = max(1, int(plan.get("n_local_workers") or 1))
        raw = run.get("raw_workload_results") or {}
        batch = raw.get("batch_size") or 4
        samples = run.get("num_samples") or 0
        perf = run.get("perf_time_sec")
        rate = (samples / batch) / perf if perf and perf else float("nan")
        measured_ms = 1000.0 / rate if rate else float("nan")

        # Sum the operators of every stage, charging cross-host round trips for
        # Ray/SMP stages from the identity-boundary measurements.
        total_ms = 0.0
        detail = []
        stage_latencies = []
        kinds = []
        for pid, p in sorted(pipes.items(), key=lambda kv: int(kv[0])):
            name = NAME.get(p.get("name", ""), p.get("name", "")[:18])
            variant = p.get("variant")
            fused = p.get("fused_pipes") or []
            members = [int(x) for x in fused] or [int(pid)]
            local_ms = sum(lat.get(m, 0.0) for m in members)
            if variant in ("RAY", "SMP"):
                table = ray_b if variant == "RAY" else smp_b
                first, last = members[0], members[-1]
                in_ms = (entry(table, first).get("input_identity_stage") or {}).get(
                    "mean_ms_per_sample", 0.0
                )
                out_ms = (entry(table, last).get("output_identity_stage") or {}).get(
                    "mean_ms_per_sample", 0.0
                )
                rt = 0.5 * float(in_ms or 0.0) + 0.5 * float(out_ms or 0.0)
                stage = local_ms + rt
                detail.append((name, variant, round(local_ms, 2), round(rt, 2)))
            else:
                stage = local_ms
                detail.append((name, "INPROCESS", round(local_ms, 2), 0.0))
            total_ms += stage
            stage_latencies.append(stage)
            kinds.append(variant or "INPROCESS")
        # Two combination rules: the serial one (every stage in one process)
        # and the pipeline one (each parallel stage owns its process, so the
        # lane sustains its slowest stage).
        parallel_latencies = [
            stage for stage, kind in zip(stage_latencies, kinds)
            if kind != "INPROCESS"
        ]
        inprocess_latency = sum(
            stage for stage, kind in zip(stage_latencies, kinds)
            if kind == "INPROCESS"
        )
        serial_latency = inprocess_latency + sum(parallel_latencies)
        pipeline_latency = inprocess_latency + (
            max(parallel_latencies) if parallel_latencies else 0.0
        )
        pred_serial = nw / serial_latency * 1000.0 if serial_latency else 0.0
        pred_pipeline = nw / pipeline_latency * 1000.0 if pipeline_latency else 0.0
        print(
            f"== {run['optimizer']:<20} nw={nw:<4} "
            f"measured={rate:7.1f} rec/s | serial rule {pred_serial:7.1f} | "
            f"pipeline rule {pred_pipeline:7.1f} rec/s"
        )
        for item in detail:
            print("     ", item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

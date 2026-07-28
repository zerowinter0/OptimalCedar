#!/usr/bin/env python3
"""Run a command and sample Cedar/Ray process resource usage via /proc."""

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import psutil


RAY_CORE_NAMES = {
    "raylet",
    "gcs_server",
    "plasma_store",
    "dashboard",
    "dashboard_agent",
    "log_monitor",
}


def process_category(proc, driver_pid, descendants):
    try:
        name = proc.name()
        cmd = " ".join(proc.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    if proc.pid == driver_pid:
        return "driver"
    if name.startswith("ray::") or "RayActor" in name or "ray::" in cmd:
        return "ray_actor"
    if name in RAY_CORE_NAMES or "raylet" in cmd or "gcs_server" in cmd:
        return "ray_core"
    if proc.pid in descendants:
        return "local_worker"
    return None


def counters(proc):
    cpu = proc.cpu_times()
    ctx = proc.num_ctx_switches()
    io = proc.io_counters()
    return {
        "user_cpu_sec": cpu.user,
        "system_cpu_sec": cpu.system,
        "voluntary_ctx_switches": ctx.voluntary,
        "involuntary_ctx_switches": ctx.involuntary,
        "read_bytes": io.read_bytes,
        "write_bytes": io.write_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("A command is required after --")

    started = time.time()
    child = subprocess.Popen(command)
    driver = psutil.Process(child.pid)
    previous = {}
    totals = defaultdict(lambda: defaultdict(float))
    maxima = defaultdict(lambda: {"rss_bytes": 0, "processes": 0, "threads": 0})
    timeline = []

    while child.poll() is None:
        sample_started = time.time()
        try:
            descendants = {p.pid for p in driver.children(recursive=True)}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            descendants = set()
        instantaneous = defaultdict(lambda: {"rss_bytes": 0, "processes": 0, "threads": 0})
        cpu_delta = defaultdict(float)
        for proc in psutil.process_iter():
            category = process_category(proc, child.pid, descendants)
            if category is None:
                continue
            try:
                current = counters(proc)
                memory = proc.memory_info().rss
                threads = proc.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            prior = previous.get(proc.pid)
            if prior is not None:
                prior_category, prior_values = prior
                if prior_category == category:
                    for key, value in current.items():
                        delta = max(0.0, value - prior_values[key])
                        totals[category][key] += delta
                        if key in ("user_cpu_sec", "system_cpu_sec"):
                            cpu_delta[category] += delta
            previous[proc.pid] = (category, current)
            instantaneous[category]["rss_bytes"] += memory
            instantaneous[category]["processes"] += 1
            instantaneous[category]["threads"] += threads

        row = {"time_sec": sample_started - started, "categories": {}}
        for category, values in instantaneous.items():
            for key, value in values.items():
                maxima[category][key] = max(maxima[category][key], value)
            row["categories"][category] = dict(values)
            row["categories"][category]["cpu_sec_since_sample"] = cpu_delta[category]
        timeline.append(row)
        time.sleep(max(0.0, args.interval - (time.time() - sample_started)))

    returncode = child.wait()
    finished = time.time()
    output = {
        "command": command,
        "returncode": returncode,
        "wall_time_sec": finished - started,
        "interval_sec": args.interval,
        "totals_by_category": {
            category: dict(values) for category, values in totals.items()
        },
        "maxima_by_category": dict(maxima),
        "timeline": timeline,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()

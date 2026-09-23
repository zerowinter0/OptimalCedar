"""Record the remote CPU topology used by the harness's actor pins.

Reads, for every CPU the experiment pins actors to: NUMA node, thread siblings
(hyper-thread pairing), core id, and the current frequency policy, plus the
node's NUMA topology.  Used to interpret per-actor compute differences.

Usage:
  python -u scripts/block_cpu_topology_probe.py --cpus 8 9 10 12 --out <run>/cpu_topology.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpus", type=int, nargs="+", default=[8, 9, 10, 12])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ray-ip", default="172.23.166.105:6379")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    def read(path: str) -> str:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return ""

    def describe(cpus):
        info = {}
        for cpu in cpus:
            base = f"/sys/devices/system/cpu/cpu{cpu}"
            info[str(cpu)] = {
                "thread_siblings": read(f"{base}/topology/thread_siblings_list"),
                "core_id": read(f"{base}/topology/core_id"),
                "physical_package_id": read(f"{base}/topology/physical_package_id"),
                "numa_node": read(f"{base}/node" + read(f"{base}/node")
                                  if False else f"{base}/node0").strip() or None,
                "governor": read(
                    f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                ),
                "cur_freq_khz": read(
                    f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
                ),
            }
            try:
                info[str(cpu)]["numa_node"] = int(
                    read(f"{base}/numa_node") or -1
                )
            except ValueError:
                info[str(cpu)]["numa_node"] = None
        return info

    payload = {"driver": describe(args.cpus)}
    if not args.local_only:
        import ray

        os.environ.setdefault("CEDAR_RAY_PLACEMENT_RESOURCE", "cedar_remote")
        os.environ.setdefault("CEDAR_RAY_REQUIRE_REMOTE", "1")
        from cedar.pipes.ray_variant import (
            configure_remote_ray_experiment,
            get_ray_actor_options,
        )
        from cedar.service import RayActor

        configure_remote_ray_experiment()
        ray.init(address=args.ray_ip, ignore_reinit_error=True)

        @ray.remote(num_cpus=0)
        class TopologyActor(RayActor):
            def report(self, cpus):
                import json as _json

                def read_local(path):
                    try:
                        return Path(path).read_text().strip()
                    except OSError:
                        return ""

                info = {}
                for cpu in cpus:
                    base = f"/sys/devices/system/cpu/cpu{cpu}"
                    entry = {
                        "thread_siblings": read_local(
                            f"{base}/topology/thread_siblings_list"
                        ),
                        "core_id": read_local(f"{base}/topology/core_id"),
                        "physical_package_id": read_local(
                            f"{base}/topology/physical_package_id"
                        ),
                        "governor": read_local(
                            f"{base}/cpufreq/scaling_governor"
                        ),
                        "cur_freq_khz": read_local(
                            f"{base}/cpufreq/scaling_cur_freq"
                        ),
                    }
                    info[str(cpu)] = entry
                numa = {}
                for node in sorted(Path("/sys/devices/system/node").glob("node*")):
                    numa[node.name] = read_local(f"{node}/cpulist")
                return {
                    "node_ip": ray.util.get_node_ip_address(),
                    "cpus": info,
                    "numa_nodes": numa,
                    "cpu_count": os.cpu_count(),
                }

        actor = TopologyActor.options(**get_ray_actor_options(0.0)).remote(
            "topology_probe"
        )
        payload["remote_actor"] = ray.get(actor.report.remote(args.cpus))
        ray.shutdown()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

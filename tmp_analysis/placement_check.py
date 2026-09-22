"""Verify the harness's Ray actors land on the remote node with the placement resource."""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CEDAR_RAY_PLACEMENT_RESOURCE", "cedar_remote")
os.environ.setdefault("CEDAR_RAY_REQUIRE_REMOTE", "1")

import ray  # noqa: E402

from cedar.pipes.ray_variant import configure_remote_ray_experiment, get_ray_actor_options  # noqa: E402
from cedar.service import RayActor  # noqa: E402

configure_remote_ray_experiment()
ray.init(address="172.23.166.105:6379", ignore_reinit_error=True)


@ray.remote(num_cpus=0)
class ProbeActor(RayActor):
    def ping(self):
        return True


options = get_ray_actor_options(0.0)
actors = [ProbeActor.options(**options).remote(f"probe_{i}") for i in range(3)]
ray.get([actor.ping.remote() for actor in actors])
locations = ray.get([actor.get_runtime_location.remote() for actor in actors])
driver_ip = ray.util.get_node_ip_address()

payload = {
    "driver_ip": driver_ip,
    "actor_locations": locations,
    "placement_resource": os.environ.get("CEDAR_RAY_PLACEMENT_RESOURCE"),
    "require_remote": os.environ.get("CEDAR_RAY_REQUIRE_REMOTE"),
    "all_remote": all(loc["node_ip"] == driver_ip for loc in locations) is False,
    "actor_options": {k: str(v) for k, v in options.items()},
}
Path(
    "/workspace/OptimalCedar/outputs/simclrv2_fusion_offload_mechanism_20260923/actor_placement_check.json"
).write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
ray.shutdown()

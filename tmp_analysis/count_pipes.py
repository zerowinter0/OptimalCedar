import importlib, sys
sys.path.insert(0, "/workspace/OptimalCedar")
from evaluation.cedar_utils import CedarEvalSpec
import yaml
module = importlib.import_module(sys.argv[1].replace("/", ".").removesuffix(".py"))
kwargs = {}
for token in sys.argv[2].split(","):
    if token.strip():
        k, _, v = token.partition("=")
        kwargs[k.strip()] = v.strip()
spec = CedarEvalSpec(1, None, 1, kwargs=kwargs, disable_optimizer=True, disable_controller=True)
ds = module.get_dataset(spec)
feat = next(iter(ds.features.values()))
pids = sorted(int(p) for p in feat.logical_pipes.keys())
print("logical pipes:", len(pids), "min", pids[0], "max", pids[-1])
prof = yaml.safe_load(open(sys.argv[3]))
lat = prof["baseline"]["latencies"]
pl = sorted(int(k) for k in lat)
print("profile pipes:", len(pl), "min", pl[0], "max", pl[-1])
print("missing in profile:", sorted(set(pids) - set(pl))[:10])
print("extra in profile:", sorted(set(pl) - set(pids))[:10])

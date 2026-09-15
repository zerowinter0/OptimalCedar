import sys, importlib, types
sys.path.insert(0, "/workspace/OptimalCedar")
from evaluation.cedar_utils import CedarEvalSpec
mod = importlib.import_module("evaluation.pipelines.llava_pretrain.cedar_dataset")
spec = CedarEvalSpec(1, None, 1)
spec.disable_optimizer = True
spec.disable_controller = True
spec.profiled_stats = None
spec.kwargs = {"dataset_path": "/tmp/small/llava_2000.jsonl", "image_root": "/workspace/OptimalCedar/evaluation/datasets/llava_pretrain"}
ds = mod.get_dataset(spec)
feat = next(iter(ds.features.values()))
for p in feat.logical_pipes.values():
    print(f"{p.id:>3}  {p.__class__.__name__:<28} {p.get_logical_name()[:34]:<36} res={p.execution_resource} fixed={getattr(p,'is_fixed',None)}")

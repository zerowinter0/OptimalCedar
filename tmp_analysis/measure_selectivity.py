"""Measure per-operator selectivity of a pipeline on real records.

    python tmp_analysis/measure_selectivity.py \
        --dataset_file evaluation/pipelines/pile_hackernews/cedar_dataset.py \
        --dataset_kwargs dataset_path=/tmp/small/pile_hackernews_2k.jsonl \
        [--records 200] [--seconds 60]

``baseline.selectivities`` drives the DP's byte/cardinality products: at 1.0
the joint search cannot tell that running a selective filter early removes
work from every later operator, which is the main reason reordering pays off
on the text recipes.  This script runs the logical pipeline in-process over
bounded real records and reports what each filter actually drops, so the
profiler can record the same numbers.
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

from cedar.client.dataset import _ProfiledFilterCallable  # noqa: E402
from cedar.pipes import FilterPipe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", required=True)
    parser.add_argument("--dataset_kwargs", default="")
    parser.add_argument("--dataset_func", default="get_dataset")
    parser.add_argument("--records", type=int, default=200)
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    kwargs = {}
    for token in filter(None, args.dataset_kwargs.split(",")):
        key, _, value = token.partition("=")
        kwargs[key] = value

    module = importlib.import_module(
        ".".join(Path(args.dataset_file).with_suffix("").parts)
    )
    getter = getattr(module, args.dataset_func)

    from evaluation.cedar_utils import CedarEvalSpec

    spec = CedarEvalSpec(
        batch_size=4,
        num_total_samples=0,
        num_epochs=1,
        config=None,
        kwargs=kwargs,
        use_ray=False,
        ray_ip="172.23.166.105:6379",
        profiled_stats=None,
        disable_optimizer=True,
        disable_controller=True,
        disable_caching=True,
    )
    dataset = getter(spec)
    feature = next(iter(dataset.features.values()))

    counters = {}
    for p_id, pipe in feature.logical_pipes.items():
        if isinstance(pipe, FilterPipe):
            counter = _ProfiledFilterCallable(pipe.fn)
            pipe.fn = counter
            counters[p_id] = (pipe.name, counter)

    started = time.time()
    processed = 0
    try:
        for _ in dataset:
            processed += 1
            if processed >= args.records or (
                time.time() - started
            ) >= args.seconds:
                break
    finally:
        elapsed = max(1e-9, time.time() - started)

    print(
        f"\n{args.dataset_file}: {processed} records in {elapsed:.1f}s "
        f"({processed / elapsed:.1f} rec/s, "
        f"{len(counters)} filters)"
    )
    print(f"{'pipe':>5} {'name':<46}{'seen':>8}{'kept':>8}{'sel':>8}")
    for p_id, (name, counter) in sorted(counters.items()):
        seen = counter.input_count
        kept = counter.output_count
        selectivity = (kept / seen) if seen else 1.0
        print(
            f"{p_id:>5} {name[:46]:<46}{seen:>8}{kept:>8}"
            f"{selectivity:>8.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

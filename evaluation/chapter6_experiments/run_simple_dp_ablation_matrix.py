"""Offline, sequential three-round baseline and one-factor ablation matrix."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
METHODS = {
    'dj-cedar-opt': 'dj_optimizer',
    'pecan-cedar-opt': 'pecan_optimizer',
    'plumber-opt': 'plumber_optimizer',
    'cedar-opt': 'optimizer',
    'ray-opt': 'raydata_optimizer',
    'unopti': 'unopti',
    'simple-dp-opt': 'simple_dp_optimizer',
    'simple-dp+W': 'simple_dp_workers',
    'simple-dp+boundary': 'simple_dp_boundary',
    'simple-dp+variant-max': 'simple_dp_variant',
    'simple-dp+width': 'simple_dp_width',
}
WORKLOADS = ['simclrv2', 'simclrv2_cache', 'commonvoice', 'coco',
             'llava_pretrain', 'stackexchange']


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd, log, env, timeout):
    started = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, cwd=REPO, start_new_session=True)
        status = 'completed'
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status = 'timeout'
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = process.returncode
        # Clear only descendants of this cell, including orphaned local SMP
        # processes. Ray actors are owned by these drivers and die with them.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return dict(status=status if code == 0 or status == 'timeout' else 'failed',
                returncode=code, wall_time_sec=time.monotonic()-started,
                command=cmd, log=str(log))


def prepare(root):
    if root.exists():
        raise RuntimeError(f'Refusing to overwrite an existing run: {root}')
    root.mkdir(parents=True)
    modules = root / 'modules'
    hashes = {}
    for package in ('cedar', 'evaluation'):
        source = REPO / package
        files = source.rglob('*.py') if package == 'cedar' else itertools.chain(
            source.glob('*.py'), (source / 'pipelines').rglob('*.py'))
        for file in files:
            relative = file.relative_to(REPO)
            target = modules / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
            hashes[str(relative)] = sha(file)
    write_json(root / 'code_sha256.json', hashes)
    subsets = root / 'inputs'
    subsets.mkdir()
    for workload, source, count in (
        ('llava_pretrain', REPO / 'evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_20000_dj_fmt_only_caption.jsonl', 1000),
        ('stackexchange', REPO / 'datasets/stackexchange/redpajama-stackexchange-10000.jsonl', 2000),
    ):
        target = subsets / (workload + '.jsonl')
        with source.open() as src, target.open('w') as dst:
            lines = list(itertools.islice(src, count))
            if len(lines) != count:
                raise RuntimeError(f'Insufficient input records: {source}')
            dst.writelines(lines)
    entry = root / 'entry.py'
    entry.write_text('''import os, runpy, sys, random
from pathlib import Path
root = Path(__file__).resolve().parent / "modules"
sys.path.insert(0, str(root))
import ray
from evaluation.cedar_utils import CedarEvalSpec
keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "CEDAR_RAY_PLACEMENT_RESOURCE",
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CEDAR_DATA_JUICER_ROOT")
def runtime_env():
    return {"working_dir": str(root),
            "env_vars": {k: os.environ[k] for k in keys if k in os.environ}}
original_config = CedarEvalSpec.to_ray_config
def to_ray_config(self):
    config = original_config(self)
    if config is not None:
        config.runtime_env = runtime_env()
    return config
CedarEvalSpec.to_ray_config = to_ray_config
original_init = ray.init
def init(*args, **kwargs):
    if kwargs.get("runtime_env") is None:
        kwargs["runtime_env"] = runtime_env()
    return original_init(*args, **kwargs)
ray.init = init
seed = int(os.environ.get("EXPERIMENT_SEED", "0"))
random.seed(seed)
import numpy as np, torch
np.random.seed(seed)
torch.manual_seed(seed)
script = sys.argv.pop(1)
runpy.run_path(script, run_name="__main__")
''')
    return modules, entry


def config(root, workload):
    batch = 4 if workload.startswith('simclrv2') else 1
    filename = 'cedar_cache_dataset.py' if workload == 'simclrv2_cache' else 'cedar_dataset.py'
    folder = 'simclrv2' if workload.startswith('simclrv2') else workload
    kwargs = {
        'simclrv2': f'dataset_path={REPO}/evaluation/datasets/imagenette2/imagenette2/train',
        'simclrv2_cache': f'dataset_path={REPO}/evaluation/datasets/imagenette2/imagenette2/train',
        'commonvoice': f'dataset_path={REPO}/datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips,max_samples=300',
        'coco': f'dataset_path={REPO}/evaluation/datasets/coco,split=val2017',
        'llava_pretrain': f'dataset_path={root}/inputs/llava_pretrain.jsonl,image_root={REPO}/evaluation/datasets/llava_pretrain',
        'stackexchange': f'dataset_path={root}/inputs/stackexchange.jsonl',
    }[workload]
    return ['--dataset_file', str(root / 'modules/evaluation/pipelines' / folder / filename),
            '--dataset_kwargs', kwargs, '--batch_size', str(batch),
            '--num_total_samples', '0']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--prepared', action='store_true')
    parser.add_argument('--workloads', nargs='+', choices=WORKLOADS, default=WORKLOADS)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.prepared:
        if (root/'status.json').exists():
            raise RuntimeError('This prepared run has already started; use a new output directory')
        modules, entry = root / 'modules', root / 'entry.py'
    else:
        modules, entry = prepare(root)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CEDAR_')}
    env.update(PYTHONPATH=str(modules), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        CEDAR_RAY_PLACEMENT_RESOURCE='cedar_remote',
        CEDAR_DATA_JUICER_ROOT=str(REPO / 'data-juicer'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        CEDAR_PROFILE_RAY_ACTORS='1', CEDAR_PROFILE_SMP_PROCS='1',
        CEDAR_PROFILE_TIME_SEC='10', CEDAR_PROFILE_BOUNDARY_MODEL='1',
        CEDAR_REUSE_BOUNDARY_MODEL='0', CEDAR_PROFILE_INFER_COMPUTE_SCALING='1',
        CEDAR_RAY_ACTOR_READY_TIMEOUT_SEC='240', CEDAR_WORKER_READY_TIMEOUT_SEC='600')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    input_records = dict(simclrv2=sum(1 for p in
        (REPO/'evaluation/datasets/imagenette2/imagenette2/train').rglob('*') if p.is_file()),
        commonvoice=300, coco=5000, llava_pretrain=1000, stackexchange=2000)
    input_records['simclrv2_cache'] = input_records['simclrv2']
    metadata = dict(input_records=input_records, workloads=args.workloads, methods=METHODS, repeats=3,
        cpu_budget=64, ray_cpu_budget=64, ray_address='172.23.166.105:6379',
        fixed_W=None, cell_timeout_sec=3600, profile_timeout_sec=10800,
        timeout_includes_import_setup_warmup_measurement_cleanup=True,
        full_input_pass=True, round_robin=True, independent_ablations=True,
        first_round_timeout_excludes_later_rounds=True,
        profile_seconds_per_stage=10, profile_actors_processes_per_stage=1,
        input_sha256={p.name: sha(p) for p in (root/'inputs').glob('*.jsonl')},
        command_by_workload={w: config(root,w) for w in args.workloads},
        environment={k:v for k,v in env.items() if k.startswith(('CEDAR_', 'OMP_', 'MKL_', 'OPENBLAS_', 'NUMEXPR_'))})
    write_json(root / 'metadata.json', metadata)
    if args.prepare_only:
        return
    (root/'runner.pid').write_text(str(os.getpid()))
    shutil.copy2(Path(__file__), root/'runner_source.py')
    state = {}
    for workload in args.workloads:
        work = root/workload
        for name in ('profiles','plans','results','logs','warmup_results','cache'):
            (work/name).mkdir(parents=True, exist_ok=True)
        profile = work/'profiles/shared.yaml'
        common = config(root, workload)+['--use_ray','--ray_ip',metadata['ray_address'],
                                        '--profiled_stats',str(profile)]
        cmd = [sys.executable, '-u', str(entry), str(modules/'evaluation/eval_cedar.py')]
        cmd += common+['--run_profiling','--disable_controller','--disable_optimizer','--disable_prefetch']
        if not workload.endswith('_cache'):
            cmd += ['--disable_caching']
        print(f'PROFILE {workload}', flush=True)
        state[workload] = {'profile': {'status':'running', 'started_unix':time.time()}, 'cells': []}
        write_json(root/'status.json',state)
        state[workload]['profile'] = run(cmd, work/'logs/profile.log', env, 10800)
        write_json(root/'status.json',state)
        if state[workload]['profile']['status'] != 'completed' or not profile.exists():
            state[workload]['blocked'] = 'profile failed or missing'
            write_json(root/'status.json',state)
            continue
        import yaml
        profile_data = yaml.safe_load(profile.read_text())
        signature = profile_data.get('resource_config', {})
        if any(signature.get(k) != 1 for k in
               ('profile_local_workers','ray_actors_per_stage','smp_procs_per_stage')):
            state[workload]['blocked'] = 'profile resource signature is not width one'
            write_json(root/'status.json',state)
            continue
        state[workload]['profile']['sha256'] = sha(profile)
        excluded = set()
        methods = list(METHODS)
        for repeat in range(3):
            order = methods[repeat:]+methods[:repeat]
            for label in order:
                internal = METHODS[label]
                cell = f'round{repeat+1}__{internal}'
                if label in excluded:
                    state[workload]['cells'].append(dict(method=label, round=repeat+1,
                        status='skipped_first_round_timeout'))
                    write_json(root/'status.json', state)
                    continue
                result = work/'results'/f'{cell}.json'
                cmd = [sys.executable, '-u', str(entry), str(modules/'evaluation/compare_optimizer_perf.py')]
                cmd += common+['--full_data_run','--enable_local_parallelism',
                    '--match_profile_resources','--cpu_budget','64','--ray_cpu_budget','64',
                    '--optimizers',internal,'--optimizer_time_limit_sec','3600',
                    '--cedar_reorder_timeout_sec','3600','--disable_cedar_runtime_timeout',
                    '--num_repeats','1','--skip_pico_plan_cost',
                    '--cache_root',str(work/'cache'),'--results_path',str(result)]
                if not workload.endswith('_cache'):
                    cmd += ['--disable_caching']
                cell_env = dict(env, EXPERIMENT_SEED=str(20260917+repeat))
                print(f'RUN {workload} {cell}', flush=True)
                state[workload]['active_cell'] = dict(method=label, round=repeat+1, started_unix=time.time())
                write_json(root/'status.json',state)
                record = run(cmd, work/'logs'/f'{cell}.log', cell_env, 3600)
                state[workload].pop('active_cell', None)
                record.update(method=label, round=repeat+1, profile_sha256=sha(profile))
                if result.exists():
                    try:
                        payload = json.loads(result.read_text())
                    except (ValueError, OSError) as exc:
                        payload = {}
                        record['result_parse_error'] = str(exc)
                        if record['status'] != 'timeout':
                            record['status'] = 'failed'
                    for measured in payload.get('runs', []):
                        if measured.get('timed_out') or measured.get('skip_reason') == 'optimizer_time_limit_exceeded':
                            record['status']='timeout'
                        elif measured.get('workload_skipped'):
                            record['status']='failed'
                        plans = measured.get('physical_plans_by_feature',{})
                        (work/'plans'/f'{cell}.yaml').write_text(yaml.safe_dump(plans))
                        write_json(work/'warmup_results'/f'{cell}.json', {
                            k:v for k,v in measured.items() if k.startswith('cache_warmup')})
                if repeat == 0 and record['status']=='timeout':
                    excluded.add(label)
                state[workload]['cells'].append(record)
                write_json(root/'status.json',state)
                from summarize_simple_dp_ablation import summarize
                summarize(root)
    write_json(root/'status.json',state)
    from summarize_simple_dp_ablation import summarize
    summarize(root)
    (root/'COMPLETE').write_text('Matrix finished; consult status.json for failures and timeouts.\n')


if __name__ == '__main__':
    main()

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
    'simple_dp': 'simple_dp',
    'simple-dp-opt': 'simple_dp',
    'simple_dp_boundary': 'simple_dp_boundary',
    'simple_dp_workers_boundary': 'simple_dp_workers_boundary',
    'simple_dp_workers_width_boundary': 'simple_dp_workers_width_boundary',
    'old_dp_boundary': 'old_dp_boundary',
    'old-dp-opt': 'old_dp_legacy_optimizer',
    'plumber-opt': 'plumber_optimizer',
    'dj-cedar-opt': 'dj_optimizer',
    'pecan-cedar-opt': 'pecan_optimizer',
    'cedar-opt': 'optimizer',
    'ray-opt': 'raydata_optimizer',
    'unopti': 'unopti',
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


def prepare(root, llava_samples=1000, stackexchange_samples=2000,
            llava_source=None, stackexchange_source=None):
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
        ('llava_pretrain', llava_source or (REPO / 'evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_20000_dj_fmt_only_caption.jsonl'), llava_samples),
        ('stackexchange', stackexchange_source or (REPO / 'datasets/stackexchange/redpajama-stackexchange-10000.jsonl'), stackexchange_samples),
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


SIMCLRV2_DATASET = 'evaluation/datasets/imagenette2/imagenette2/train'
# 9,469 local images replayed this many times give the 189,380 records the
# scaled campaign asks for; no external download is involved.
SIMCLRV2_EPOCHS = 20


def config(root, workload, commonvoice_max_samples=300,
           commonvoice_dataset_path=None, coco_split='val2017',
           simclrv2_dataset=None):
    batch = 4 if workload.startswith('simclrv2') else 1
    filename = 'cedar_cache_dataset.py' if workload == 'simclrv2_cache' else 'cedar_dataset.py'
    folder = 'simclrv2' if workload.startswith('simclrv2') else workload
    simclrv2_path = simclrv2_dataset or (REPO / SIMCLRV2_DATASET)
    kwargs = {
        'simclrv2': f'dataset_path={simclrv2_path}',
        'simclrv2_cache': f'dataset_path={simclrv2_path}',
        'commonvoice': (f'dataset_path={commonvoice_dataset_path or (REPO / "datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips")},'
            f'max_samples={commonvoice_max_samples}'),
        'coco': f'dataset_path={REPO}/evaluation/datasets/coco,split={coco_split}',
        'llava_pretrain': f'dataset_path={root}/inputs/llava_pretrain.jsonl,image_root={REPO}/evaluation/datasets/llava_pretrain',
        'stackexchange': f'dataset_path={root}/inputs/stackexchange.jsonl',
    }[workload]
    epochs = 1
    if workload.startswith('simclrv2'):
        epochs = SIMCLRV2_EPOCHS
    return ['--dataset_file', str(root / 'modules/evaluation/pipelines' / folder / filename),
            '--dataset_kwargs', kwargs, '--batch_size', str(batch),
            '--num_epochs', str(epochs),
            '--num_total_samples', str(record_counts().get(workload, 0))]


LLAVA_INPUT = 'evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_558k.jsonl'
STACKEXCHANGE_INPUT = 'datasets/stackexchange/redpajama-stackexchange-400000.jsonl'


def record_counts() -> dict:
    """Records each workload must process, from the experiment configuration."""
    return dict(RECORD_COUNTS)


# Scaled data volumes for the final campaign. simclrv2 and simclrv2_cache run
# on ImageNet-1k (189,380 = 20x the 9,469 imagenette2 training images), coco
# uses train2017, llava_pretrain and stackexchange take larger subsets of the
# official corpora, and commonvoice uses all 300,000 local clips.
RECORD_COUNTS = {
    'simclrv2': 189380,
    'simclrv2_cache': 189380,
    'commonvoice': 300000,
    'coco': 50000,
    'llava_pretrain': 50000,
    'stackexchange': 20000,
}


def main():
    global METHODS
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--prepared', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-cedar-workloads', nargs='*', choices=WORKLOADS, default=[])
    parser.add_argument('--commonvoice-max-samples', type=int, default=300)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--workloads', nargs='+', choices=WORKLOADS, default=WORKLOADS)
    parser.add_argument('--methods', nargs='+', choices=list(METHODS),
                        help='run only the selected methods')
    parser.add_argument('--smp-aggregate-profile', action='store_true', default=True,
                        help='measure a real-object SMP aggregate IPC curve per workload')
    parser.add_argument('--layered-profile', action='store_true', default=True,
        help='profile each operator x backend with the adaptive layered protocol')
    parser.add_argument('--skip-cell', action='append', default=[],
        metavar='LABEL@WORKLOAD',
        help='mark a previously timed-out cell as skipped without running it')
    parser.add_argument('--cell-timeout-sec', type=int, default=3600)
    parser.add_argument('--commonvoice-dataset-path', type=Path)
    parser.add_argument('--simclrv2-dataset', type=Path,
        help='image directory for the simclrv2 workloads (default: ImageNet-1k train)')
    parser.add_argument('--simclrv2-epochs', type=int, default=1,
        help='repetitions of the simclrv2 image list; the scaled campaign '
             'replays the local full dataset instead of downloading ImageNet')
    parser.add_argument('--coco-split', default='train2017',
        choices=('val2017', 'train2017'))
    parser.add_argument('--llava-samples', type=int, default=RECORD_COUNTS['llava_pretrain'])
    parser.add_argument('--stackexchange-samples', type=int,
        default=RECORD_COUNTS['stackexchange'])
    parser.add_argument('--llava-source', type=Path, default=REPO / LLAVA_INPUT)
    parser.add_argument('--stackexchange-source', type=Path,
        default=REPO / STACKEXCHANGE_INPUT)
    parser.add_argument('--profile-from', nargs='+', type=Path, default=[],
        help=('reuse a validated shared profile from an earlier run directory '
              'instead of profiling again; the source directory is recorded '
              'in status.json for provenance'))
    args = parser.parse_args()
    if args.methods:
        METHODS = {name: METHODS[name] for name in args.methods}
    global SIMCLRV2_EPOCHS
    if args.simclrv2_epochs < 1:
        parser.error('--simclrv2-epochs must be positive')
    SIMCLRV2_EPOCHS = args.simclrv2_epochs
    if args.smp_aggregate_profile and not args.layered_profile:
        parser.error('--smp-aggregate-profile requires --layered-profile')
    if args.cell_timeout_sec < 1:
        parser.error('--cell-timeout-sec must be positive')
    if args.commonvoice_dataset_path:
        args.commonvoice_dataset_path = args.commonvoice_dataset_path.resolve()
        files = sorted(args.commonvoice_dataset_path.rglob('*.mp3'))
        if len(files) < args.commonvoice_max_samples:
            parser.error('CommonVoice dataset contains fewer real clips than requested')
    skip_cells = set()
    for raw in args.skip_cell:
        if '@' not in raw:
            parser.error('--skip-cell must be LABEL@WORKLOAD: ' + raw)
        label, workload = raw.split('@', 1)
        if label not in METHODS or workload not in WORKLOADS:
            parser.error('unknown --skip-cell target: ' + raw)
        skip_cells.add((label, workload))
    if args.commonvoice_max_samples < 1:
        parser.error('--commonvoice-max-samples must be positive')
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    root = args.output.resolve()
    if args.resume:
        if not (root/'status.json').exists():
            raise RuntimeError('Cannot resume without status.json')
        modules, entry = root / 'modules', root / 'entry.py'
    elif args.prepared:
        if (root/'status.json').exists():
            raise RuntimeError('This prepared run has already started; use --resume')
        modules, entry = root / 'modules', root / 'entry.py'
    else:
        modules, entry = prepare(
            root,
            llava_samples=args.llava_samples,
            stackexchange_samples=args.stackexchange_samples,
            llava_source=args.llava_source,
            stackexchange_source=args.stackexchange_source,
        )
    env = {k: v for k, v in os.environ.items() if not k.startswith('CEDAR_')}
    env.update(PYTHONPATH=str(modules), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        CEDAR_RAY_PLACEMENT_RESOURCE='cedar_remote', CEDAR_RAY_REQUIRE_REMOTE='1',
        CEDAR_BOUNDARY_DIAGNOSTICS_DIR=str(root / 'boundary_diagnostics'),
        CEDAR_DATA_JUICER_ROOT=str(REPO / 'data-juicer'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        CEDAR_PROFILE_RAY_ACTORS='1', CEDAR_PROFILE_SMP_PROCS='1',
        CEDAR_PROFILE_TIME_SEC='10', CEDAR_PROFILE_BOUNDARY_MODEL='1',
        CEDAR_REUSE_BOUNDARY_MODEL='0', CEDAR_PROFILE_INFER_COMPUTE_SCALING='1',
        CEDAR_RAY_ACTOR_READY_TIMEOUT_SEC='240', CEDAR_WORKER_READY_TIMEOUT_SEC='600')
    if args.smp_aggregate_profile:
        env["CEDAR_PROFILE_SMP_AGGREGATE_TRANSPORT"] = "1"
    if args.layered_profile:
        env.update(
            CEDAR_LAYERED_ADAPTIVE_PROFILE='1',
            CEDAR_ADAPTIVE_PROFILE_MIN_SEC='3',
            CEDAR_ADAPTIVE_PROFILE_MAX_SEC='30',
            CEDAR_ADAPTIVE_PROFILE_TARGET_RSE='0.10',
            CEDAR_ADAPTIVE_PROFILE_MIN_OBS='30',
            CEDAR_PROFILE_POOL_SAMPLES='64',
            CEDAR_PROFILE_POOL_BYTES_PER_PIPE=str(64 * 1024 * 1024),
            CEDAR_PROFILE_POOL_BYTES_TOTAL=str(512 * 1024 * 1024),
            CEDAR_PROFILE_SCALING_WIDTHS='1,2,4,8',
            CEDAR_PROFILE_SCALING_TOP_K='5',
            CEDAR_PROFILE_SCALING_MAX_SEC='10')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    input_records = record_counts()
    input_records['commonvoice'] = args.commonvoice_max_samples
    input_records['llava_pretrain'] = args.llava_samples
    input_records['stackexchange'] = args.stackexchange_samples
    if args.resume:
        metadata = json.loads((root/'metadata.json').read_text())
        args.commonvoice_max_samples = metadata['commonvoice_max_samples']
        args.repeats = metadata['repeats']
        args.cell_timeout_sec = metadata.get('cell_timeout_sec', args.cell_timeout_sec)
        saved_path = metadata.get('commonvoice_dataset_path')
        if saved_path:
            args.commonvoice_dataset_path = Path(saved_path)
        skipped = set(metadata.get('skip_cedar_workloads', []))
        skipped.update(args.skip_cedar_workloads)
        metadata['skip_cedar_workloads'] = sorted(skipped)
        for raw in metadata.get('skip_cells', []):
            label, workload = raw.split('@', 1)
            skip_cells.add((label, workload))
    else:
        metadata = dict(input_records=input_records, workloads=args.workloads, methods=METHODS, repeats=args.repeats,
            cpu_budget=64, ray_cpu_budget=64, ray_address='172.23.166.105:6379',
            local_runtime_cpu_reserve_per_worker=0, ray_runtime_cpu_reserve_per_worker=0,
            fixed_W=None, cell_timeout_sec=args.cell_timeout_sec, profile_timeout_sec=10800,
            timeout_includes_import_setup_warmup_measurement_cleanup=True,
            full_input_pass=True, round_robin=True, independent_ablations=True,
            first_round_timeout_excludes_later_rounds=True,
            profile_seconds_per_stage=10, profile_actors_processes_per_stage=1,
            input_sha256={p.name: sha(p) for p in (root/'inputs').glob('*.jsonl')},
            commonvoice_max_samples=args.commonvoice_max_samples,
            commonvoice_dataset_path=str(args.commonvoice_dataset_path) if args.commonvoice_dataset_path else None,
            skip_cedar_workloads=sorted(args.skip_cedar_workloads),
            layered_profile=args.layered_profile,
            smp_aggregate_profile=args.smp_aggregate_profile,
            skip_cells=sorted(raw for raw in args.skip_cell),
            profile_protocol=('dual_legacy_whole_pipeline_plus_adaptive_layered'
                              if args.layered_profile
                              else 'standard_single_operator_offload'),
            command_by_workload={w: config(root, w, args.commonvoice_max_samples, args.commonvoice_dataset_path, args.coco_split, args.simclrv2_dataset)
                                 for w in args.workloads},
            environment={k:v for k,v in env.items() if k.startswith(('CEDAR_', 'OMP_', 'MKL_', 'OPENBLAS_', 'NUMEXPR_'))})
    write_json(root / 'metadata.json', metadata)
    if args.prepare_only:
        return
    (root/'runner.pid').write_text(str(os.getpid()))
    shutil.copy2(Path(__file__), root/'runner_source.py')
    state = (json.loads((root/'status.json').read_text())
             if args.resume else {})
    for workload in args.workloads:
        work = root/workload
        for name in ('profiles','plans','results','logs','warmup_results','cache'):
            (work/name).mkdir(parents=True, exist_ok=True)
        profile = work/'profiles/shared.yaml'
        common = config(root, workload, args.commonvoice_max_samples, args.commonvoice_dataset_path, args.coco_split, args.simclrv2_dataset)+['--use_ray','--ray_ip',metadata['ray_address'],
                                        '--profiled_stats',str(profile)]
        cmd = [sys.executable, '-u', str(entry), str(modules/'evaluation/eval_cedar.py')]
        cmd += common+['--run_profiling','--disable_controller','--disable_optimizer','--disable_prefetch']
        if not workload.endswith('_cache'):
            cmd += ['--disable_caching']
        reuse_profile = (
            workload in state
            and state[workload].get('profile', {}).get('status') == 'completed'
            and profile.exists()
        )
        if not reuse_profile and args.profile_from:
            for source in args.profile_from:
                candidate = source / workload / 'profiles' / 'shared.yaml'
                if candidate.exists():
                    shutil.copy2(candidate, profile)
                    state[workload] = {
                        'profile': {
                            'status': 'completed',
                            'source': str(candidate),
                            'source_sha256': sha(candidate),
                        },
                        'cells': [],
                    }
                    write_json(root / 'status.json', state)
                    print(f'REUSE PROFILE {workload} <- {candidate}', flush=True)
                    reuse_profile = True
                    break
        if reuse_profile:
            print(f'REUSE PROFILE {workload}', flush=True)
        else:
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
        from cedar.client.boundary_profiler import validate_remote_ray_boundary
        validate_remote_ray_boundary(profile_data)
        physical = profile_data.get('physical_model', {})
        if (not profile_data.get('layered_profile')
                or physical.get('operator_affine', {}).get('schema_version') != 1):
            state[workload]['blocked'] = (
                'Shared optimizer profile must contain legacy and layered entries; '
                'regenerate the profile instead of reusing a legacy-only profile')
            write_json(root/'status.json', state)
            continue
        if any(not isinstance(entry, dict) or 'throughput' not in entry
               or 'backend_compute' not in entry
               for backend in profile_data.get('offloads', {}).values()
               if isinstance(backend, dict)
               for entry in backend.values()):
            state[workload]['blocked'] = (
                'Shared optimizer profile lost legacy throughput or isolated backend_compute')
            write_json(root/'status.json', state)
            continue
        signature = profile_data.get('resource_config', {})
        if any(signature.get(k) != 1 for k in
               ('profile_local_workers','ray_actors_per_stage','smp_procs_per_stage')):
            state[workload]['blocked'] = 'profile resource signature is not width one'
            write_json(root/'status.json',state)
            continue
        state[workload]['profile']['sha256'] = sha(profile)
        excluded = {c['method'] for c in state[workload].get('cells', [])
                    if c.get('round') == 1 and c.get('status') == 'timeout'}
        methods = list(METHODS)
        for repeat in range(args.repeats):
            order = methods[repeat:]+methods[:repeat]
            for label in order:
                internal = METHODS[label]
                cell = f'round{repeat+1}__{internal}'
                previous = [c for c in state[workload].get('cells', [])
                            if c.get('method') == label and c.get('round') == repeat+1]
                if previous:
                    print(f'REUSE {workload} {cell} status={previous[-1].get("status")}', flush=True)
                    continue
                if (label, workload) in skip_cells:
                    state[workload]['cells'].append(dict(
                        method=label, round=repeat+1,
                        status='skipped_previous_timeout',
                        reason='Known timeout from the previous profile campaign'))
                    write_json(root/'status.json', state)
                    print(f'SKIP {workload} {cell} (previous timeout)', flush=True)
                    continue
                if label == 'cedar-opt' and workload in metadata.get('skip_cedar_workloads', []):
                    state[workload]['cells'].append(dict(
                        method=label, round=repeat+1, status='skipped_user_requested',
                        reason='Known Cedar optimization timeout for this workload'))
                    write_json(root/'status.json', state)
                    print(f'SKIP {workload} {cell} (user requested)', flush=True)
                    continue
                if label in excluded:
                    state[workload]['cells'].append(dict(method=label, round=repeat+1,
                        status='skipped_first_round_timeout'))
                    write_json(root/'status.json', state)
                    continue
                result = work/'results'/f'{cell}.json'
                cmd = [sys.executable, '-u', str(entry), str(modules/'evaluation/compare_optimizer_perf.py')]
                cmd += common+['--full_data_run','--enable_local_parallelism',
                    '--match_profile_resources','--cpu_budget','64','--ray_cpu_budget','64',
                    '--optimizers',internal,'--optimizer_time_limit_sec',str(args.cell_timeout_sec),
                    '--cedar_reorder_timeout_sec',str(args.cell_timeout_sec),'--disable_cedar_runtime_timeout',
                    '--num_repeats','1','--skip_pico_plan_cost',
                    '--cache_root',str(work/'cache'),'--results_path',str(result)]
                if not workload.endswith('_cache'):
                    cmd += ['--disable_caching']
                cell_env = dict(env, EXPERIMENT_SEED=str(20260917+repeat))
                print(f'RUN {workload} {cell}', flush=True)
                state[workload]['active_cell'] = dict(method=label, round=repeat+1, started_unix=time.time())
                write_json(root/'status.json',state)
                record = run(cmd, work/'logs'/f'{cell}.log', cell_env, args.cell_timeout_sec)
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

"""Controlled Cedar Ray measurements; separate from formal workload results."""
import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import os
import signal
import shutil
from pathlib import Path
import subprocess
import sys
import time
import traceback


def partition(stages, iterations):
    if stages not in (1, 2, 3) or iterations < 0 or iterations % stages:
        raise ValueError('stage partition must preserve integer work')
    return [6 // stages] * stages, [iterations // stages] * stages


def rotation(values, repeat):
    offset = repeat % len(values)
    return values[offset:] + values[:offset]


class Work:
    def __init__(self, iterations):
        self.iterations = iterations

    def __call__(self, *args):
        record = args[0] if len(args) == 1 else args
        index, payload, done = record
        if self.iterations:
            # Fixed CPU work independent of payload; never a wall-clock sleep.
            hashlib.pbkdf2_hmac('sha256', b'pico', b'controlled', self.iterations)
        return index, payload, done + self.iterations


class Records:
    def __init__(self, count, size):
        self.count, self.size = count, size

    def __iter__(self):
        payload = b'x' * self.size
        for index in range(self.count):
            yield index, payload, 0


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def execute(command, log, timeout):
    """Bound an owned process group, including all local Cedar drivers."""
    with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                          start_new_session=True) as process:
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return 124


def initialize_ray(lock, **kwargs):
    import ray
    with lock:
        ray.init(**kwargs)


def worker(rank, cfg, barrier, init_lock, out):
    # Spawn imports this file as __mp_main__; pickle used by Cedar needs a
    # stable importable callable name on the remote node.
    from controlled import Work as RemoteWork
    import ray
    from cedar.config import CedarContext
    from cedar.pipes import MapperPipe, PipeVariantType, RayPipeVariantContext
    from cedar.sources import IterSource
    pipes = []
    try:
        os.sched_setaffinity(0, {cfg['cpus'][rank]})
        initialize_ray(init_lock, address=cfg['address'], runtime_env={
            'working_dir': str(Path(__file__).parent),
            'env_vars': {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                         'MKL_NUM_THREADS': '1'}}, log_to_driver=False)
        ctx = CedarContext()
        source = IterSource(Records(cfg['count'], cfg['payload'])).to_pipe()
        source.id = 0
        source.mutate(ctx, PipeVariantType.INPROCESS)
        pipes.append(source)
        for stage, (width, work) in enumerate(zip(cfg['widths'], cfg['work'])):
            pipe = MapperPipe(pipes[-1], RemoteWork(work))
            pipe.id = stage + 1
            pipe.mutate(ctx, PipeVariantType.RAY, RayPipeVariantContext(
                n_actors=width, max_inflight=100, max_prefetch=100,
                submit_batch_size=1, use_threads=True,
                profile_backend_compute=True))
            pipes.append(pipe)
        # Separate warm-up epoch: same actors, no timing observations retained.
        source.source.count = 128
        for _ in pipes[-1].get_variant():
            pass
        source.source.count = cfg['count']
        for pipe in pipes[1:]:
            pipe.get_variant().variant_ctx.service.reset_backend_compute_stats()
        barrier.wait(timeout=240)
        started = time.perf_counter()
        ids = set()
        marks = []
        target_work = sum(cfg['work'])
        for sample in pipes[-1].get_variant():
            if sample.dummy:
                continue
            index, payload, done = sample.data
            if index in ids or len(payload) != cfg['payload'] or done != target_work:
                raise RuntimeError('invalid output or duplicate record')
            ids.add(index)
            if len(ids) % 256 == 0:
                marks.append([len(ids), time.perf_counter() - started])
        elapsed = time.perf_counter() - started
        if ids != set(range(cfg['count'])):
            raise RuntimeError('incomplete record coverage')
        stats = [p.get_variant().variant_ctx.service.get_backend_compute_stats()
                 for p in pipes[1:]]
        if any(s is None or s['count'] != cfg['count'] for s in stats):
            raise RuntimeError('incomplete worker-side timing observations')
        dump(out, dict(rank=rank, seconds=elapsed, started_monotonic=started, samples=len(ids),
                       stage_compute=stats, progress=marks, config=cfg))
    except BaseException:
        dump(out, {'rank': rank, 'error': traceback.format_exc()})
        try:
            barrier.abort()
        except Exception:
            pass
        raise
    finally:
        for pipe in reversed(pipes):
            pipe.get_variant().shutdown()
        ray.shutdown()


def run_case(cfg, directory):
    directory.mkdir(parents=True, exist_ok=False)
    context = mp.get_context('spawn')
    barrier = context.Barrier(cfg['workers'])
    init_lock = context.Lock()
    procs = [context.Process(target=worker, args=(r, cfg, barrier, init_lock,
             str(directory / f'worker{r}.json'))) for r in range(cfg['workers'])]
    for proc in procs:
        proc.start()
    deadline = time.monotonic() + 3500
    for proc in procs:
        proc.join(max(0, deadline - time.monotonic()))
    failed = any(p.exitcode != 0 for p in procs)
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
            proc.join(10)
    if failed:
        raise RuntimeError('worker failure or timeout; see per-worker logs')
    rows = [json.loads((directory / f'worker{r}.json').read_text())
            for r in range(cfg['workers'])]
    dump(directory / 'result.json', dict(config=cfg, workers=rows,
        seconds=max(r['seconds'] for r in rows),
        samples=sum(r['samples'] for r in rows)))


def profile(root, address):
    """Frozen width-one calibration; each timed pipeline runs >=10 seconds."""
    from controlled import Work as RemoteWork
    import ray
    from cedar.config import CedarContext
    from cedar.pipes import MapperPipe, PipeVariantType, RayPipeVariantContext
    from cedar.sources import IterSource
    ray.init(address=address, runtime_env={'working_dir': str(Path(__file__).parent)},
             log_to_driver=False)
    rows = []
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    for payload in (512, 65536, 1048576):
        for iterations in (0, 400, 600, 1200, 40000, 60000, 120000):
            fn, item = RemoteWork(iterations), (0, b'x' * payload, 0)
            started, count = time.perf_counter(), 0
            while time.perf_counter() - started < 10:
                fn(item)
                count += 1
            rows.append(dict(variant='callable_local', payload=payload,
                             iterations=iterations, seconds=time.perf_counter()-started,
                             samples=count))
            local_source = IterSource(Records(10**12, payload)).to_pipe()
            local_source.id = 0
            local_source.mutate(CedarContext(), PipeVariantType.INPROCESS)
            local_pipe = MapperPipe(local_source, fn)
            local_pipe.id = 1
            local_pipe.mutate(CedarContext(), PipeVariantType.INPROCESS)
            started, count = time.perf_counter(), 0
            for sample in local_pipe.get_variant():
                if not sample.dummy:
                    count += 1
                if time.perf_counter() - started >= 10:
                    break
            rows.append(dict(variant='INPROCESS', payload=payload,
                             iterations=iterations, seconds=time.perf_counter()-started,
                             samples=count))
            local_pipe.get_variant().shutdown()
            local_source.get_variant().shutdown()
            ctx = CedarContext()
            source = IterSource(Records(128, payload)).to_pipe()
            source.id = 0
            source.mutate(ctx, PipeVariantType.INPROCESS)
            pipe = MapperPipe(source, RemoteWork(iterations))
            pipe.id = 1
            pipe.mutate(ctx, PipeVariantType.RAY, RayPipeVariantContext(
                n_actors=1, max_inflight=100, max_prefetch=100,
                profile_backend_compute=True))
            try:
                for _ in pipe.get_variant():
                    pass
                service = pipe.get_variant().variant_ctx.service
                service.reset_backend_compute_stats()
                source.source.count = 10**12
                started, count = time.perf_counter(), 0
                for sample in pipe.get_variant():
                    if not sample.dummy:
                        count += 1
                    if time.perf_counter() - started >= 10:
                        break
                elapsed = time.perf_counter() - started
                record = dict(variant='RAY', payload=payload, iterations=iterations, width=1,
                              seconds=elapsed, samples=count,
                              compute=service.get_backend_compute_stats())
                rows.append(record)
                dump(root / 'profile.json', rows)
                print(json.dumps({'event': 'profile', **record}), flush=True)
            finally:
                pipe.get_variant().shutdown()
                source.get_variant().shutdown()
    ray.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--address', default='172.23.166.105:6379')
    parser.add_argument('--case', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--reuse-profile-root', type=Path)
    args = parser.parse_args()
    os.environ.setdefault('CEDAR_RAY_PLACEMENT_RESOURCE', 'cedar_remote')
    if args.profile_only:
        profile(args.root, args.address)
        return
    if args.case:
        run_case(json.loads(args.case.read_text()), args.root)
        return
    args.root.mkdir(parents=True, exist_ok=True)
    status = args.root / 'STATUS'
    status.write_text('RUNNING preflight\n')
    try:
        import ray
        ray.init(address=args.address, log_to_driver=False)
        resources, available = ray.cluster_resources(), ray.available_resources()
        nodes = ray.nodes()
        def runtime_fingerprint():
            import cedar
            import socket
            base = Path(cedar.__file__).parent
            names = ['pipes/map.py', 'pipes/ray_variant.py',
                     'pipes/variant.py', 'service/ray_service.py']
            return dict(host=socket.gethostname(), python=sys.version,
                        load=os.getloadavg(),
                        hashes={name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                                for name in names})
        remote_fingerprint = ray.get(ray.remote(num_cpus=1,
            resources={'cedar_remote': .001})(runtime_fingerprint).remote(), timeout=60)
        local_fingerprint = runtime_fingerprint()
        if remote_fingerprint['hashes'] != local_fingerprint['hashes']:
            raise RuntimeError('local and remote Cedar runtime code differ')
        ray.shutdown()
        if available.get('CPU', 0) < 48 or available.get('cedar_remote', 0) < .048:
            raise RuntimeError('insufficient idle remote resources; no other jobs stopped')
        cpus = sorted(os.sched_getaffinity(0))[:8]
        if len(cpus) != 8:
            raise RuntimeError('eight local CPUs required')
        dump(args.root / 'metadata.json', dict(
            W=8, CPU_BUDGET=64, remote_actors=48, repeats=3,
            profile_seconds=10, submit_batch_size=1, cpus=cpus,
            resources=resources, available=available, nodes=nodes,
            local_runtime=local_fingerprint, remote_runtime=remote_fingerprint,
            smoke=args.smoke, python=sys.version, timestamp=time.time(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        if args.reuse_profile_root:
            old = args.reuse_profile_root
            metadata = json.loads((old / 'metadata.json').read_text())
            for key, fingerprint in [('local_runtime', local_fingerprint),
                                     ('remote_runtime', remote_fingerprint)]:
                if metadata[key]['hashes'] != fingerprint['hashes']:
                    raise RuntimeError('profile runtime fingerprint mismatch')
            def definition(path, name):
                return ast.dump(next(node for node in ast.parse(path.read_text()).body
                                     if getattr(node, 'name', None) == name))
            for name in ('Work', 'Records', 'profile'):
                if definition(old / 'source/controlled.py', name) != definition(Path(__file__), name):
                    raise RuntimeError(f'profile implementation changed: {name}')
            rows = json.loads((old / 'profile.json').read_text())
            expected = {(v, b, i) for v in ('callable_local', 'INPROCESS', 'RAY')
                        for b in (512, 65536, 1048576)
                        for i in (0, 400, 600, 1200, 40000, 60000, 120000)}
            if len(rows) != 63 or {(r['variant'], r['payload'], r['iterations']) for r in rows} != expected:
                raise RuntimeError('incomplete profile matrix')
            if any(r['seconds'] < 10 or r['samples'] <= 0 for r in rows):
                raise RuntimeError('invalid calibration duration/count')
            for name in ('profile.json', 'profile.log'):
                shutil.copyfile(old / name, args.root / name)
            dump(args.root / 'profile_provenance.json', dict(source=str(old.resolve()),
                 sha256=hashlib.sha256((old / 'profile.json').read_bytes()).hexdigest(),
                 validation='63 entries; >=10 sec; identical Work/Records/profile AST and runtime hashes'))
            print(json.dumps({'event': 'profile_reused', 'source': str(old)}), flush=True)
        elif not args.smoke:
            status.write_text('RUNNING profiling\n')
            with (args.root / 'profile.log').open('w') as log:
                code = execute([sys.executable, '-u', __file__, '--root', str(args.root),
                                '--address', args.address, '--profile-only'], log, 1800)
            if code:
                raise RuntimeError(f'profiling failed: exit {code}')
        errors = []
        payloads = (512,) if args.smoke else (512, 65536, 1048576)
        works = (1200,) if args.smoke else (0, 1200, 120000)
        for repeat in range(1 if args.smoke else 3):
            for payload in payloads:
                for work in works:
                    for stages in rotation([1, 2, 3], repeat):
                        name = f'r{repeat+1}_bytes{payload}_work{work}_stages{stages}'
                        status.write_text(f'RUNNING {name}\n')
                        widths, chunks = partition(stages, work)
                        cfg = dict(address=args.address, workers=8, cpus=cpus,
                                   count=32 if args.smoke else 4096,
                                   payload=payload, widths=widths, work=chunks)
                        config_path = args.root / f'{name}.json'
                        dump(config_path, cfg)
                        print(json.dumps({'event': 'start', 'case': name}), flush=True)
                        with (args.root / f'{name}.log').open('w') as log:
                            code = execute([sys.executable, '-u', __file__,
                                '--root', str(args.root / name), '--case', str(config_path)],
                                log, 3600)
                            if code:
                                errors.append(name)
                        print(json.dumps({'event': 'case_finished', 'case': name,
                                          'success': name not in errors}), flush=True)
                        if name in errors:
                            # Abort on infrastructure failures instead of accumulating
                            # invalid or resource-overlapping measurements.
                            raise RuntimeError(f'case failed: {name}')
        status.write_text('COMPLETE\n')
        print(json.dumps({'event': 'complete'}), flush=True)
    except BaseException:
        status.write_text('FAILED\n' + traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

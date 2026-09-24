"""Patch run_simple_dp_ablation_matrix.py with opt-in layered profiling.

Adds:
  --layered-profile       enable the adaptive layered per-operator protocol
  --skip-cell LABEL@WORK  mark one previously timed-out cell skipped

The default behaviour (no flags) is unchanged.  Run inside the container or
directly on the host; a ``.bak_layered_20260918`` backup is created once.
"""
from pathlib import Path

PATH = Path(
    "/workspace/OptimalCedar/evaluation/chapter6_experiments/"
    "run_simple_dp_ablation_matrix.py"
)
if not PATH.exists():
    PATH = Path(
        "/home/xieruiyang/OptimalCedar/evaluation/chapter6_experiments/"
        "run_simple_dp_ablation_matrix.py"
    )
BACKUP = PATH.with_suffix(PATH.suffix + ".bak_layered_20260918")

REPLACEMENTS = [
    (
        """    parser.add_argument('--workloads', nargs='+', choices=WORKLOADS, default=WORKLOADS)
    args = parser.parse_args()
    if args.commonvoice_max_samples < 1:
""",
        """    parser.add_argument('--workloads', nargs='+', choices=WORKLOADS, default=WORKLOADS)
    parser.add_argument('--layered-profile', action='store_true',
        help='profile each operator x backend with the adaptive layered protocol')
    parser.add_argument('--skip-cell', action='append', default=[],
        metavar='LABEL@WORKLOAD',
        help='mark a previously timed-out cell as skipped without running it')
    args = parser.parse_args()
    skip_cells = set()
    for raw in args.skip_cell:
        if '@' not in raw:
            parser.error('--skip-cell must be LABEL@WORKLOAD: ' + raw)
        label, workload = raw.split('@', 1)
        if label not in METHODS or workload not in WORKLOADS:
            parser.error('unknown --skip-cell target: ' + raw)
        skip_cells.add((label, workload))
    if args.commonvoice_max_samples < 1:
""",
    ),
    (
        """        CEDAR_RAY_ACTOR_READY_TIMEOUT_SEC='240', CEDAR_WORKER_READY_TIMEOUT_SEC='600')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
""",
        """        CEDAR_RAY_ACTOR_READY_TIMEOUT_SEC='240', CEDAR_WORKER_READY_TIMEOUT_SEC='600')
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
""",
    ),
    (
        """        skipped = set(metadata.get('skip_cedar_workloads', []))
        skipped.update(args.skip_cedar_workloads)
        metadata['skip_cedar_workloads'] = sorted(skipped)
""",
        """        skipped = set(metadata.get('skip_cedar_workloads', []))
        skipped.update(args.skip_cedar_workloads)
        metadata['skip_cedar_workloads'] = sorted(skipped)
        for raw in metadata.get('skip_cells', []):
            label, workload = raw.split('@', 1)
            skip_cells.add((label, workload))
""",
    ),
    (
        """            commonvoice_max_samples=args.commonvoice_max_samples,
            skip_cedar_workloads=sorted(args.skip_cedar_workloads),
""",
        """            commonvoice_max_samples=args.commonvoice_max_samples,
            skip_cedar_workloads=sorted(args.skip_cedar_workloads),
            layered_profile=args.layered_profile,
            skip_cells=sorted(raw for raw in args.skip_cell),
            profile_protocol=('adaptive_layered_fixed_legal_input'
                              if args.layered_profile
                              else 'standard_single_operator_offload'),
""",
    ),
    (
        """                if previous:
                    print(f'REUSE {workload} {cell} status={previous[-1].get("status")}', flush=True)
                    continue
                if label == 'cedar-opt' and workload in metadata.get('skip_cedar_workloads', []):
""",
        """                if previous:
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
""",
    ),
]


def main():
    source = PATH.read_text()
    if BACKUP.exists():
        if source == BACKUP.read_text():
            pass
        elif '--layered-profile' in source:
            print(f"already patched: {PATH}")
            return
    else:
        BACKUP.write_text(source)
    patched = source
    for old, new in REPLACEMENTS:
        if new in patched and old not in patched:
            continue
        if old not in patched:
            raise SystemExit(f"anchor not found:\n{old[:120]}")
        patched = patched.replace(old, new, 1)
    PATH.write_text(patched)
    import py_compile
    py_compile.compile(str(PATH), doraise=True)
    print(f"patched {PATH} (backup {BACKUP})")


if __name__ == "__main__":
    main()

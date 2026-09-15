"""Run eval_cedar.py with a trace of every parallel stage that gets built.

Answers "what widths did the runtime actually instantiate" for a plan that the
harness rewrites (``apply_profile_matched_resources``), which a plan file alone
cannot tell: the run may re-size every stage.

Usage (inside the container):
  python tmp_analysis/eval_plan_trace.py evaluation/eval_cedar.py <args...>
"""

import runpy
import sys
import os


def _log(line: str) -> None:
    """Workers have their stdout captured, so trace to a file as well."""
    with open(f"/tmp/plantrace_{os.getpid()}.log", "a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _install() -> None:
    try:
        from cedar.service.ray_service import RayService
    except Exception as exc:  # pragma: no cover - import guard
        print(f"[plantrace] hook not installed: {exc}", flush=True)
        return

    ray_register = RayService.register

    def register_patched(self, name, actors):
        _log(
            f"[plantrace] RAY {name} actors={len(actors)} "
            f"submit={getattr(self, 'submit_batch_size', None)}"
        )
        return ray_register(self, name, actors)

    RayService.register = register_patched

    try:
        from cedar.pipes.variant import SMPPipeVariant as smp_class
    except Exception:
        try:
            from cedar.pipes.variant import SMPPipeVariantV2 as smp_class
        except Exception as exc:  # pragma: no cover - optional hook
            print(f"[plantrace] SMP hook unavailable: {exc}", flush=True)
            return
    smp_init = smp_class.__init__

    def smp_patched(self, name, input_pipe_variant, variant_ctx):
        _log(
            f"[plantrace] SMP {name} procs={getattr(variant_ctx, 'n_procs', None)} "
            f"inflight={getattr(variant_ctx, 'max_inflight', None)}"
        )
        return smp_init(self, name, input_pipe_variant, variant_ctx)

    smp_class.__init__ = smp_patched


def main() -> int:
    _install()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())

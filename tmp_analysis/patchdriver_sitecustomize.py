"""Startup hook that records the widths the runtime instantiates.

Copy to ``/tmp/patchdriver/sitecustomize.py`` and export
``PYTHONPATH=/tmp/patchdriver`` to trace every dataset worker (which is a
fresh process, so a hook installed by a wrapper script would be lost).
"""

import os


def _log(line: str) -> None:
    try:
        with open(f"/tmp/plantrace_{os.getpid()}.log", "a") as handle:
            handle.write(line + "\n")
    except Exception:
        pass


def _install() -> None:
    try:
        from cedar.service.ray_service import RayService
    except Exception:
        return

    original = RayService.register

    def register_patched(self, name, actors):
        _log(f"[plantrace] RAY {name} actors={len(actors)}")
        return original(self, name, actors)

    RayService.register = register_patched


try:
    _install()
except Exception:
    pass

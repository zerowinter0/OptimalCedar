"""Per-record counters injected into every worker and Ray actor.

Copy to ``/tmp/patchsite/sitecustomize.py`` on the driver host and on the Ray
host, then export ``PYTHONPATH=/tmp/patchsite`` for the workers and
``CEDAR_RAY_ACTOR_PYTHONPATH=/tmp/patchsite`` for the actors.  Every process
then appends ``label duration_sec timestamp role`` lines to
``/tmp/opcalls_<pid>.log``.

Two counter families are installed:

* Data-Juicer operators (``__call__``/``compute_stats_single``/
  ``process_single``), which give the per-operator CPU time of the text
  pipelines, and
* ``PipeVariant.__iter__``, one line per record that leaves a pipe, which
  works for every other pipeline (image workloads) and gives the number of
  records each stage actually processed plus the window it was busy.
"""

import os
import sys
import time

ROLE = "%s|%s|%s" % (
    os.path.basename(sys.argv[0] or "?"),
    os.environ.get("RAY_WORKER_ID", "-")[:8],
    os.environ.get("RAY_ACTOR_ID", "-"),
)
LOG = "/tmp/opcalls_%s.log" % os.getpid()
METHODS = ("__call__", "compute_stats_single", "process_single")


def _record(label, seconds):
    try:
        with open(LOG, "a") as handle:
            handle.write("%s %.6f %.3f %s\n" % (label, seconds, time.time(), ROLE))
    except Exception:
        pass


def _wrap(cls, name, method):
    orig = cls.__dict__.get(method)
    if orig is None:
        return

    def patched(self, *a, **k):
        t0 = time.perf_counter()
        r = orig(self, *a, **k)
        _record("%s.%s" % (name, method), time.perf_counter() - t0)
        return r

    setattr(cls, method, patched)


def _install_operators():
    try:
        from evaluation.pipelines.stackexchange import dj_operators as ops
    except Exception:
        return
    for attr in dir(ops):
        obj = getattr(ops, attr)
        if isinstance(obj, type):
            for method in METHODS:
                _wrap(obj, attr, method)


def _install_pipe_counter():
    """Count records leaving every pipe, whatever backend ran it."""
    try:
        from cedar.pipes.variant import PipeVariant
    except Exception:
        return

    original = PipeVariant.__iter__

    def patched(self):
        label = "pipe:%s:p%s" % (
            type(self).__name__,
            getattr(self, "p_id", None),
        )
        for sample in original(self):
            _record(label, 0.0)
            yield sample

    PipeVariant.__iter__ = patched


try:
    _install_operators()
except Exception:
    pass
try:
    _install_pipe_counter()
except Exception:
    pass

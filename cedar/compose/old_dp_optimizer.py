"""Joint DP before affine profiling: PERDATA/PERRECORD cost semantics.

The search and resource allocation are shared with DpOptimizer. Initialization
deliberately bypasses AffineDpCostMixin so even a shared profile containing
cm_model retains the original size ratios and compute classification.
"""
from .dp_optimizer import DpOptimizer
from .my_optimizer import MyOptimizer


class OldDpOptimizer(DpOptimizer):
    def _init_stats(self):
        self._dp_affine_enabled = False
        MyOptimizer._init_stats(self)

"""Joint DP scored with Cedar's original byte-volume cost model.

This optimizer is retained only as the historical cost-model control: it
deliberately ignores the fitted ``kx+b`` operator layer and prices compute by
serialized byte volume, while the search and resource allocation are shared
with DpOptimizer.
"""
from .dp_optimizer import DpOptimizer
from .my_optimizer import MyOptimizer


class OldDpOptimizer(DpOptimizer):
    uses_affine_operator_cost = False

    def _init_stats(self):
        # Bypass AffineDpCostMixin so the byte-volume recurrence is used even
        # when the shared profile carries fitted affine coefficients.
        MyOptimizer._init_stats(self)

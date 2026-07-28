"""Backward-compatible alias for the renamed two-stage DP optimizer."""

from .dp_two_stage_optimizer import DpTwoStageOptimizer

DpSeperateOptimizer = DpTwoStageOptimizer

__all__ = ["DpSeperateOptimizer", "DpTwoStageOptimizer"]

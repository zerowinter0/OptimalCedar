"""Tests for the experimental optimizer."""

from evaluation.compare_optimizer_perf import OPTIMIZERS

def test_exp_optimizer_cli_selector():
    assert OPTIMIZERS["exp_optimizer"] == 7

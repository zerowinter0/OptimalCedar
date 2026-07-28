from types import SimpleNamespace

import pytest

from cedar.compose.dp_optimizer import (
    BlockCandidate,
    CacheTransitionPolicy,
    DpStateSummary,
)
from cedar.pipes import PipeVariantType


class _DeterministicPipe:
    def is_random(self) -> bool:
        return False


def test_cache_transition_prices_block_output_and_excludes_source_cost():
    optimizer = SimpleNamespace(
        options=SimpleNamespace(enable_caching=True),
        profiled_stats={
            "baseline": {"output_sizes": {99: 10.0}},
            "disk_info": {"read_latency": 0.001},
        },
        logical_pipes={0: _DeterministicPipe(), 1: _DeterministicPipe()},
        _base_cost_map={99: 1234.0},
        # Mask 01 is the prefix before the block; mask 11 includes an
        # expanding block whose output is five times larger.
        _dp_r_prod=[1.0, 2.0, 5.0, 10.0],
        _get_source_p_id=lambda: 99,
    )
    policy = CacheTransitionPolicy(optimizer, [0, 1])
    block = BlockCandidate(
        mask=0b10,
        order=(1,),
        variant=PipeVariantType.INPROCESS,
        cost=7.0,
        materializes_fusion=False,
    )

    choices = list(
        policy.transitions(
            prev_mask=0b01,
            next_mask=0b11,
            prev_state=DpStateSummary(cache_active=False),
            regular_cost=7.0,
            block=block,
            next_parallel_stage_cpus=5,
        )
    )

    assert len(choices) == 2
    cache_choice = choices[1]
    assert cache_choice.replaces_prefix_cost
    assert cache_choice.cache_after_idx == 1
    assert cache_choice.state.parallel_stage_cpus == 5
    # 0.001 seconds/byte * 1000 ms/second * 10 bytes/source sample
    # * output ratio 10.  The source cost must not be subtracted.
    assert cache_choice.extra_cost == pytest.approx(100.0)


def test_fully_deterministic_pipeline_defers_cache_until_output():
    optimizer = SimpleNamespace(
        options=SimpleNamespace(enable_caching=True),
        profiled_stats={
            "baseline": {"output_sizes": {99: 10.0}},
            "disk_info": {"read_latency": 0.001},
        },
        logical_pipes={
            0: _DeterministicPipe(),
            1: _DeterministicPipe(),
            2: _DeterministicPipe(),
        },
        _base_cost_map={99: 1.0},
        _dp_r_prod=[1.0] * 8,
        _get_source_p_id=lambda: 99,
    )
    policy = CacheTransitionPolicy(optimizer, [0, 1, 2])
    block = BlockCandidate(
        mask=0b010,
        order=(1,),
        variant=PipeVariantType.INPROCESS,
        cost=1.0,
        materializes_fusion=False,
    )

    choices = list(
        policy.transitions(
            prev_mask=0b001,
            next_mask=0b011,
            prev_state=DpStateSummary(),
            regular_cost=1.0,
            block=block,
        )
    )

    assert len(choices) == 1
    assert choices[0].state.cache_active is False

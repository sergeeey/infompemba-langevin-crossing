"""The two counterexample chains behave as claimed in the note (scope of the spectral criterion R)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prereg_v2"))

from counterexamples import (
    chain_r_greater_than_one,
    chain_symmetric_states,
    sign_changes,
)


def test_r_greater_than_one_can_still_cross_twice():
    t, gap = chain_r_greater_than_one()
    changes = sign_changes(t, gap)
    assert len(changes) == 2
    assert 0.40 < changes[0] < 0.43 and 1.69 < changes[1] < 1.72
    assert gap[0] < 0 < gap[t.searchsorted(1.0)] and gap[-1] < 0  # -, +, -


def test_two_symmetric_states_can_still_cross():
    t, gap = chain_symmetric_states()
    assert gap[0] < 0  # hot starts farther (gap = KL_cold - KL_hot < 0)
    assert gap[t.searchsorted(1.0)] > 0  # and the hot state is ahead later
    assert len(sign_changes(t, gap)) >= 1

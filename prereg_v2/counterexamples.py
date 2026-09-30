"""Two small reversible Markov chains that bound what the spectral criterion R can claim (post-hoc, 2026-09-30).

Origin: a third-party review of note v0.4 (an AI reviewer; the review file is not part of this repository) proposed both
constructions; they were re-run and checked here with the assertions below before being used in the note.

1. r > 1 does not exclude a crossing: a 4-state chain in which the hot state has twice the slow-mode coefficient of
   the cold state (r = 2) yet the KL gap changes sign twice. R is an asymptotic statement.
2. Two SYMMETRIC initial states, both with zero projection on the odd slowest mode, can still cross: the ordering is
   then decided by the leading accessible even modes, so parity alone does not rule out a crossing.

These chains are unrelated to the double-well solver; they only test the logical scope of the criterion.
"""

import numpy as np
from scipy.linalg import expm
from scipy.special import xlogy


def kl(p: np.ndarray, pi: np.ndarray) -> float:
    return float(np.sum(xlogy(p, p / pi)))


def _assert_reversible_chain(q: np.ndarray, pi: np.ndarray) -> None:
    off = q - np.diag(np.diag(q))
    assert off.min() >= -1e-15, "off-diagonal rates must be non-negative"
    assert np.allclose(q.sum(axis=0), 0.0, atol=1e-12), (
        "columns of Q must sum to zero (d rho/dt = Q rho)"
    )
    assert np.allclose(q @ pi, 0.0, atol=1e-12), "pi must be stationary"
    assert np.allclose(q * pi[None, :], (q * pi[None, :]).T, atol=1e-12), "detailed balance"


def chain_r_greater_than_one():
    """4 states; slow-mode ratio r = 0.02 / 0.01 = 2; returns (times, gap = KL_cold - KL_hot)."""
    v1 = np.array([1, 1, -1, -1.0]) / 2
    v2 = np.array([1, -1, 1, -1.0]) / 2
    v3 = np.array([1, -1, -1, 1.0]) / 2
    q = -np.outer(v1, v1) - 2 * np.outer(v2, v2) - 2.9 * np.outer(v3, v3)  # rates 1, 2, 2.9
    pi = np.full(4, 0.25)
    cold = pi + 0.01 * v1 + 0.10 * v2
    hot = pi + 0.02 * v1 + 0.14 * v3
    _assert_reversible_chain(q, pi)
    assert (
        cold.min() > 0 and hot.min() > 0 and np.isclose(cold.sum(), 1) and np.isclose(hot.sum(), 1)
    )
    assert kl(hot, pi) > kl(cold, pi), "hot must start farther (P0)"
    t = np.linspace(0.0, 12.0, 2401)
    gap = np.array([kl(expm(s * q) @ cold, pi) - kl(expm(s * q) @ hot, pi) for s in t])
    return t, gap


def chain_symmetric_states():
    """5 states, reflection symmetry; both initial states symmetric; returns (times, gap = KL_cold - KL_hot)."""
    odd = np.array([1, -1, 0, 1, -1.0]) / 2
    even2 = np.array([1, 1, -4, 1, 1.0]) / np.sqrt(20)
    even3 = np.array([1, -1, 0, -1, 1.0]) / 2
    q = (
        -3 * (np.eye(5) - np.ones((5, 5)) / 5)
        + 1.5 * np.outer(odd, odd)
        - 0.5 * np.outer(even3, even3)
    )
    pi = np.full(5, 0.2)
    cold = pi + 0.08 * even2
    hot = pi + 0.10 * even3
    _assert_reversible_chain(q, pi)
    assert np.allclose(cold, cold[::-1]) and np.allclose(hot, hot[::-1]), (
        "both initial states symmetric"
    )
    assert abs(odd @ cold) < 1e-15 and abs(odd @ hot) < 1e-15, "zero odd-mode projection for both"
    assert cold.min() > 0 and hot.min() > 0
    assert kl(hot, pi) > kl(cold, pi), "hot must start farther (P0)"
    t = np.linspace(0.0, 3.0, 601)
    gap = np.array([kl(expm(s * q) @ cold, pi) - kl(expm(s * q) @ hot, pi) for s in t])
    return t, gap


def sign_changes(t: np.ndarray, gap: np.ndarray) -> list[float]:
    s = np.sign(gap)
    idx = np.flatnonzero(s[1:] != s[:-1])
    return [float(t[i + 1]) for i in idx]


if __name__ == "__main__":
    t, g = chain_r_greater_than_one()
    print("chain 1 (r = 2): gap sign changes near t =", [round(x, 3) for x in sign_changes(t, g)])
    t, g = chain_symmetric_states()
    print(
        "chain 2 (both symmetric): gap sign changes near t =",
        [round(x, 3) for x in sign_changes(t, g)],
    )

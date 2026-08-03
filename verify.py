"""Independent Monte-Carlo cross-check of the solver (Phase 4).

The solver computes game value with vectorized tree math over the M / C matrices.
This is a completely separate code path: sample a compatible hole-combo pair in
proportion to the ranges, play the hand out by SAMPLING actions from the average
strategy, and score the terminal directly from strengths and pot/contributions.
Averaged over many hands it must reproduce the solver's computed game value --
the same "independent method must agree" discipline used to validate Leduc
against OpenSpiel.
"""

import numpy as np


def _sample_index(cum, u):
    i = int(np.searchsorted(cum, u))
    return min(i, len(cum) - 1)


def _p0_payoff(node, si, sj):
    """Net chips to player 0 at a terminal, given the two players' strengths."""
    if node.kind == "showdown":
        pot, stake = node.pot, node.stake
        if si > sj:
            return pot - stake
        if si < sj:
            return -stake
        return 0.5 * pot - stake
    # fold: the non-folder wins the pot
    c0 = node.contrib[0]
    return (node.pot - c0) if node.folder == 1 else -c0


def mc_game_value(solver, n_samples=100_000, seed=0):
    """Realized average value to P0 of the average strategy, by sampling.
    Returns (mean, standard_error). Compare to solver.game_values()[0] / Z."""
    rng = np.random.default_rng(seed)
    N = solver.N
    strengths = solver.strengths
    cards = [frozenset(c) for c in solver.combos]

    # sampling distributions and per-node cumulative average strategies
    p0 = solver.range0 / solver.range0.sum()
    p1 = solver.range1 / solver.range1.sum()
    cum0, cum1 = np.cumsum(p0), np.cumsum(p1)
    node_cum = {n: np.cumsum(solver.average_strategy(n), axis=1) for n in solver.regret}

    total = total_sq = 0.0
    for _ in range(n_samples):
        while True:                                  # joint ~ p0(i) p1(j) [compatible]
            i = _sample_index(cum0, rng.random())
            j = _sample_index(cum1, rng.random())
            if cards[i].isdisjoint(cards[j]):
                break
        node = solver.root
        while not node.is_terminal():
            combo = i if node.player == 0 else j
            a = _sample_index(node_cum[node][combo], rng.random())
            node = node.actions[a][1]
        x = _p0_payoff(node, strengths[i], strengths[j])
        total += x
        total_sq += x * x

    mean = total / n_samples
    var = total_sq / n_samples - mean * mean
    return mean, np.sqrt(var / n_samples)

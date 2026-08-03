"""Ranges over a board's hole combos (Phase 1).

A range is a nonnegative weight vector aligned with the combo list returned by
board.board_strengths(board): weights[i] is how often the player holds combos[i].
Weights need not sum to 1 -- the solver only cares about relative reach.
"""

import numpy as np


def uniform_range(n):
    return np.ones(n)


def random_range(n, seed):
    return np.random.default_rng(seed).random(n)


def range_from_weights(combos, weights_by_combo, default=0.0):
    """Build an aligned weight vector from a {(card_a, card_b): weight} dict."""
    index = {c: i for i, c in enumerate(combos)}
    r = np.full(len(combos), float(default))
    for combo, w in weights_by_combo.items():
        a, b = combo
        key = (a, b) if a < b else (b, a)
        r[index[key]] = w
    return r

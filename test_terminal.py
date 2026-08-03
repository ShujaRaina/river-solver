"""Checks for terminal EV over ranges (Phase 1). Run: python test_terminal.py

The fast matrix path is differentially tested against a pure-Python O(N^2) brute
force -- the same discipline the Kuhn cfr value-walk was checked with.
"""

import numpy as np

from cards import parse_cards
from board import board_strengths
from ranges import random_range
from terminal import (compatibility_mask, showdown_matrix, showdown_ev,
                      fold_ev, brute_showdown_ev, brute_fold_ev)

board = parse_cards("Ah Kd 7s 2c 9h")
combos, strengths = board_strengths(board)
strengths = np.array(strengths)
n = len(combos)
index = {c: i for i, c in enumerate(combos)}


def cix(s):
    a, b = parse_cards(s)
    return index[(a, b) if a < b else (b, a)]


def onehot(i):
    r = np.zeros(n)
    r[i] = 1.0
    return r


M = showdown_matrix(combos, strengths)
C = compatibility_mask(combos)

# --- structural properties of the showdown matrix ---
assert M.shape == (n, n)
assert np.allclose(M, -M.T), "antisymmetric on compatible pairs"
assert np.all(np.diag(M) == 0), "a combo vs itself is blocked"
assert np.all((M != 0) <= (C == 1)), "nonzero only where compatible"

# --- hand-checked single-combo showdowns (stake = 10) ---
STAKE = 10.0
assert np.isclose(showdown_ev(M, onehot(cix("3c 4c")), STAKE)[cix("As Ac")], +STAKE), \
    "trips of aces beat 3-4 high"
assert np.isclose(showdown_ev(M, onehot(cix("As Ac")), STAKE)[cix("3c 4c")], -STAKE), \
    "3-4 high loses to trips"
assert np.isclose(showdown_ev(M, onehot(cix("Qc Jh")), STAKE)[cix("Qs Jd")], 0.0), \
    "identical high-card holdings chop"
assert np.isclose(showdown_ev(M, onehot(cix("Ac 5d")), STAKE)[cix("As Ac")], 0.0), \
    "CARD REMOVAL: opponent can't hold the Ac that hero holds"

# --- differential: fast matrix == brute-force O(N^2) ---
R = random_range(n, seed=1)
assert np.allclose(showdown_ev(M, R, STAKE),
                   brute_showdown_ev(combos, strengths, R, STAKE)), "showdown fast==brute"
assert np.allclose(fold_ev(C, R, 7.0),
                   brute_fold_ev(combos, R, 7.0)), "fold fast==brute"

# --- zero-sum: symmetric ranges give hero no showdown edge ---
assert np.isclose(R @ (M @ R), 0.0, atol=1e-8), "R^T M R = 0 by antisymmetry"

print("all terminal checks passed")

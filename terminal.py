"""Terminal (fold / showdown) EV over ranges, with card removal (Phase 1).

At a river terminal we need, for each of the hero's combos, the expected payoff
against the opponent's reach -- a weight vector over the opponent's combos. The
one true subtlety is CARD REMOVAL: hero holding combo i while the opponent holds
a combo j that shares a card is impossible, so that pairing must contribute
nothing. Get it wrong and every value is silently corrupted -- the river's
version of the clairvoyance trap that best-response had in the Kuhn/Leduc work.

Fast path (single board): precompute an N*N showdown matrix M and compatibility
mask C once, with blocked pairs zeroed. Then every terminal EV is a single
matrix-vector product:

    showdown EV to hero  =  stake  * (M @ opp_reach)
    fold EV to hero      =  amount * (C @ opp_reach)

For N ~ 1081, M is ~9 MB -- trivial. (For turn/flop solves, where you can't
store N^2 for every runout, an O(N log N) sorted-prefix-sum showdown replaces
this; that's the only reason to bother with it.)

Every fast function is differentially tested against a pure-Python O(N^2) brute
force in test_terminal.py.
"""

import numpy as np


def _card_membership(combos):
    """(N, 52) indicator: row i has 1s in the two columns of combos[i]'s cards."""
    H = np.zeros((len(combos), 52))
    for i, (a, b) in enumerate(combos):
        H[i, a] = 1.0
        H[i, b] = 1.0
    return H


def compatibility_mask(combos):
    """C[i, j] = 1 if combos i and j share no card, else 0. Diagonal is 0."""
    H = _card_membership(combos)
    shared = H @ H.T          # shared[i, j] = number of cards in common (0, 1, 2)
    return (shared == 0).astype(float)


def showdown_matrix(combos, strengths):
    """M[i, j] in {+1, 0, -1}: +1 if i beats j, -1 if i loses, 0 if tie OR the
    pairing is impossible (shared card). Antisymmetric on compatible pairs."""
    s = np.asarray(strengths)
    sign = np.sign(s[:, None] - s[None, :])
    return sign * compatibility_mask(combos)


def showdown_ev(M, opp_reach, stake):
    """Value to hero at a showdown; `stake` is each player's at-risk contribution
    (hero nets +stake for a win, -stake for a loss, 0 for a chop)."""
    return stake * (M @ opp_reach)


def fold_ev(C, opp_reach, amount):
    """Value to hero when someone folds; strength-independent but still card-
    removed. `amount` is +opp_contribution (opponent folded) or
    -hero_contribution (hero folded)."""
    return amount * (C @ opp_reach)


# --- pure-Python O(N^2) references, only for tests --------------------------

def _compatible(ci, cj):
    a, b = ci
    c, d = cj
    return a != c and a != d and b != c and b != d


def brute_showdown_ev(combos, strengths, opp_reach, stake):
    n = len(combos)
    out = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for j in range(n):
            if _compatible(combos[i], combos[j]):
                acc += np.sign(strengths[i] - strengths[j]) * opp_reach[j]
        out[i] = stake * acc
    return out


def brute_fold_ev(combos, opp_reach, amount):
    n = len(combos)
    out = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for j in range(n):
            if _compatible(combos[i], combos[j]):
                acc += opp_reach[j]
        out[i] = amount * acc
    return out

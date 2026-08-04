"""O(N) sorted-prefix-sum showdown with card removal (Phase T4).

Replaces the O(N^2) `M @ v` / `C @ v` matmuls (which need an N*N matrix per board,
~250 MB per river on the turn) with prefix sums over strength. For opponent reach
v and hero combo i = (a, b):

    Cv[i] = total - contain[a] - contain[b] + v[i]                 (compatible mass)
    Mv[i] = signed(str_i) - signed_a(str_i) - signed_b(str_i)      (win mass - lose mass)

where signed(s) = 2*(reach strictly below s) + (reach at s) - total, computed
globally and per card. The per-card terms remove opponent combos sharing a card
with i; the "shares both a and b" combo is i itself, a tie (sign 0), so it needs
no correction. Ties (equal strength) contribute 0 by construction. Combos with
strength < 0 (blocked by the river card) are masked out.

This is the error-prone one, so it is differential-tested against M @ v / C @ v
in test_showdown_fast.py -- including paired boards (ties) and blocked combos.
"""

import numpy as np


def sorted_showdown(strengths, card_a, card_b, v):
    """Return (Mv, Cv): the signed showdown and compatible-mass vectors for a
    single board. strengths[i] is combo i's dense rank (-1 = blocked); card_a[i],
    card_b[i] its two cards (0..51); v the opponent reach."""
    strengths = np.asarray(strengths)
    card_a = np.asarray(card_a)
    card_b = np.asarray(card_b)
    v = np.asarray(v, dtype=float)

    valid = strengths >= 0
    K = int(strengths[valid].max()) if valid.any() else 0
    s = np.where(valid, strengths, 0)                     # clamp blocked -> 0 for indexing
    vv = v * valid                                        # blocked combos carry no reach

    # global prefix sums over strength
    bucket = np.bincount(s, weights=vv, minlength=K + 1)  # reach at each strength
    total = bucket.sum()
    below = np.empty(K + 1); below[0] = 0.0
    below[1:] = np.cumsum(bucket)[:-1]                    # reach strictly below each strength
    signed = 2.0 * below + bucket - total                # signed(s) over strengths

    # per-card prefix sums (52 x (K+1))
    bucket_c = np.zeros((52, K + 1))
    np.add.at(bucket_c, (card_a, s), vv)                 # combo i adds to its two cards
    np.add.at(bucket_c, (card_b, s), vv)
    contain = bucket_c.sum(axis=1)                        # reach of combos containing each card
    below_c = np.zeros((52, K + 1))
    below_c[:, 1:] = np.cumsum(bucket_c, axis=1)[:, :-1]
    signed_c = 2.0 * below_c + bucket_c - contain[:, None]

    Mv = signed[s] - signed_c[card_a, s] - signed_c[card_b, s]
    Cv = total - contain[card_a] - contain[card_b] + vv
    return Mv * valid, Cv * valid


def sorted_showdown_batched(strengths_stack, card_a, card_b, V):
    """Same computation batched over rivers. strengths_stack, V are (R, N); the
    two card arrays are (N,). Returns (Mv, Cv) each (R, N). Uses bincount (not a
    Python river loop) so the 48 runouts are one vectorized pass."""
    R, N = V.shape
    valid = strengths_stack >= 0
    K1 = int(strengths_stack.max()) + 1
    s = np.where(valid, strengths_stack, 0)
    VV = (V * valid).ravel()
    rows = np.broadcast_to(np.arange(R)[:, None], (R, N))
    ca = np.broadcast_to(np.asarray(card_a)[None, :], (R, N))
    cb = np.broadcast_to(np.asarray(card_b)[None, :], (R, N))

    bucket = np.bincount((rows * K1 + s).ravel(), weights=VV,
                         minlength=R * K1).reshape(R, K1)
    total = bucket.sum(1)
    below = np.zeros((R, K1)); below[:, 1:] = np.cumsum(bucket, 1)[:, :-1]
    signed = 2.0 * below + bucket - total[:, None]

    m = R * 52 * K1
    bucket_c = (np.bincount(((rows * 52 + ca) * K1 + s).ravel(), weights=VV, minlength=m)
                + np.bincount(((rows * 52 + cb) * K1 + s).ravel(), weights=VV, minlength=m)
                ).reshape(R, 52, K1)
    contain = bucket_c.sum(2)
    below_c = np.zeros((R, 52, K1)); below_c[:, :, 1:] = np.cumsum(bucket_c, 2)[:, :, :-1]
    signed_c = 2.0 * below_c + bucket_c - contain[:, :, None]

    ri = np.arange(R)[:, None]
    Mv = signed[ri, s] - signed_c[ri, ca, s] - signed_c[ri, cb, s]
    Cv = total[:, None] - contain[ri, ca] - contain[ri, cb] + (V * valid)
    return Mv * valid, Cv * valid

"""Differential test: sorted showdown == M @ v / C @ v (Phase T4).

The whole point of this file is to catch the silent bugs (ties, card-removal
inclusion-exclusion, blocked combos). We compare the O(N) sorted method against
the trusted O(N^2) matmul on many random reach vectors across several boards,
deliberately including a PAIRED board (lots of chops) and the turn's per-river
BLOCKED combos.
"""

import numpy as np

from cards import parse_cards, hole_combos
from board import board_strengths
from terminal import showdown_matrix, compatibility_mask
from turn_eval import turn_strengths
from showdown_fast import sorted_showdown, sorted_showdown_batched

rng = np.random.default_rng(0)


def check(combos, strengths, label, trials=10):
    strengths = np.asarray(strengths)
    card_a = np.array([c[0] for c in combos])
    card_b = np.array([c[1] for c in combos])
    M = showdown_matrix(combos, np.where(strengths < 0, -1, strengths))
    C = compatibility_mask(combos)
    # blocked combos: zero their rows/cols in the oracle too
    valid = (strengths >= 0).astype(float)
    M = M * valid[:, None] * valid[None, :]
    Cmask = C * valid[:, None] * valid[None, :]
    for _ in range(trials):
        v = rng.random(len(combos)) * valid          # blocked combos carry no reach
        Mv, Cv = sorted_showdown(strengths, card_a, card_b, v)
        assert np.allclose(Mv, M @ v, atol=1e-9), f"{label}: Mv mismatch"
        assert np.allclose(Cv, Cmask @ v, atol=1e-9), f"{label}: Cv mismatch"
    print(f"  {label}: sorted == matmul over {trials} random reaches")


# 1) an ordinary river board
combos, strengths = board_strengths(parse_cards("Ah Kd 7s 2c 9h"))
check(combos, strengths, "dry board")

# 2) a PAIRED board -> many ties/chops (the tie-boundary trap)
combos, strengths = board_strengths(parse_cards("7h 7d Ks Kd 2c"))
check(combos, strengths, "paired board (ties)")

# 3) a monotone board -> flushes, more ties among non-flush hands
combos, strengths = board_strengths(parse_cards("Ts 8s 5s 2s Jd"))
check(combos, strengths, "flushy board")

# 4) the turn's per-river strengths, WITH blocked combos (strength -1)
turn_combos, per_river = turn_strengths(parse_cards("Ts 9h 2c 5d"))
for rc in list(per_river)[:3]:
    check(turn_combos, per_river[rc], f"turn river blocked ({rc})", trials=5)

# 5) the BATCHED (over-rivers) version must match the single-board version
turn_combos, per_river = turn_strengths(parse_cards("Ts 9h 2c 5d"))
card_a = np.array([c[0] for c in turn_combos])
card_b = np.array([c[1] for c in turn_combos])
rivers = list(per_river)
strengths_stack = np.stack([per_river[r] for r in rivers])           # (R, N)
V = rng.random((len(rivers), len(turn_combos))) * (strengths_stack >= 0)
Mv_b, Cv_b = sorted_showdown_batched(strengths_stack, card_a, card_b, V)
for k, r in enumerate(rivers):
    Mv_s, Cv_s = sorted_showdown(per_river[r], card_a, card_b, V[k])
    assert np.allclose(Mv_b[k], Mv_s, atol=1e-9), f"batched Mv mismatch river {r}"
    assert np.allclose(Cv_b[k], Cv_s, atol=1e-9), f"batched Cv mismatch river {r}"
print(f"  batched == single over all {len(rivers)} rivers")

print("all sorted-showdown differential checks passed")

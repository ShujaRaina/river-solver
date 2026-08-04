"""Hand-checks for per-river-card evaluation (Phase T0). Run: python test_turn_eval.py"""

from cards import parse_cards
from turn_eval import river_cards, turn_strengths

turn = parse_cards("Ts 9h 2c 5d")          # unpaired, no flush possible from a draw
combos, per_river = turn_strengths(turn)
index = {c: i for i, c in enumerate(combos)}


def cix(s):
    a, b = parse_cards(s)
    return index[(a, b) if a < b else (b, a)]


def card(s):
    return parse_cards(s)[0]


# --- counts ---
assert len(river_cards(turn)) == 48, "48 possible river cards"
assert len(combos) == 1128, "C(48,2) hole combos on a 4-card board"
for r, arr in per_river.items():
    assert int((arr >= 0).sum()) == 1081, "C(47,2) combos survive each river"
    assert int((arr < 0).sum()) == 47, "a river blocks 47 combos (it + each other card)"

# --- a combo is blocked exactly on the rivers that use one of its cards ---
qj = cix("Qh Jh")
assert per_river[card("Qh")][qj] == -1, "a Qh river blocks QhJh"
assert per_river[card("Jh")][qj] == -1, "a Jh river blocks QhJh"
assert per_river[card("3s")][qj] >= 0, "a 3s river does not block QhJh"

# --- strength REACTS to the river: QhJh (open-ended: T9 board) completes to a
#     straight on a King and stays weak (queen-high) on a blank ---
straight_river = per_river[card("Kc")][qj]      # K-Q-J-T-9
blank_river = per_river[card("3s")][qj]         # queen-high
assert straight_river > blank_river, "completing the straight must raise strength"

# and the completing river should be a big jump, not a marginal one
assert straight_river - blank_river > 0.3 * per_river[card("Kc")].max(), \
    "a made straight is far above queen-high"

print("all turn-eval checks passed")

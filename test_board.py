"""Hand-checks for fixed-board ranking (Phase 0). Run: python test_board.py"""

from cards import parse_cards
from board import board_strengths


def combo(s):
    a, b = parse_cards(s)
    return (a, b) if a < b else (b, a)


# A dry board: no flush, no straight, no pair on the board.
board = parse_cards("Ah Kd 7s 2c 9h")
combos, strengths = board_strengths(board)
smap = dict(zip(combos, strengths))

assert len(combos) == 1081, "C(47,2) valid combos on a 5-card board"
assert len(smap) == 1081, "combos are unique"
assert min(strengths) == 0, "dense strengths start at 0"

# trips aces (As Ac + Ah) beat trips kings (Ks Kc + Kd) beat trips nines
assert smap[combo("As Ac")] > smap[combo("Ks Kc")] > smap[combo("9c 9s")]

# two pair (aces up) beats top pair beats a worse pair
assert smap[combo("As Ks")] > smap[combo("As Js")]      # AK two pair > pair of aces? see below
assert smap[combo("Ac Qc")] > smap[combo("Qs Jd")]      # pair of aces > queen-high

# CHOP: two different Q-J combos both just play A-K-Q-J-9 (high card) -> equal
assert smap[combo("Qs Jd")] == smap[combo("Qc Jh")], "identical high-card holdings tie"

print("all board checks passed")

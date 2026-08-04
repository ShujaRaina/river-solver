"""Per-river-card evaluation for the turn solver (Phase T0).

A turn spot has a 4-card board; the river is one of the 48 remaining cards. The
hole combos live over the TURN board (C(48,2) = 1128 of them), but a hand's
showdown strength depends on which river lands -- and a combo that *contains* the
river card is impossible once that card is public.

`turn_strengths(turn_board)` returns, for every possible river card r, a strength
vector aligned to the fixed 1128-combo turn universe: the dense showdown rank on
`turn_board + r` for combos that survive, and -1 for combos blocked by r. Those
per-river strength vectors are what the chance node (Phase T2) will branch over.
"""

import numpy as np

from cards import hole_combos
from board import board_strengths


def river_cards(turn_board):
    """The 48 cards that could come on the river (any not already on the turn)."""
    dead = set(turn_board)
    return [c for c in range(52) if c not in dead]


def turn_strengths(turn_board):
    """Return (turn_combos, per_river) where:

    turn_combos : the 1128 hole combos valid on the 4-card turn board
    per_river   : {river_card: int32 strength vector aligned to turn_combos}
                  higher rank beats lower; -1 marks a combo blocked by that river
                  card (impossible once it is on the board).
    """
    if len(turn_board) != 4:
        raise ValueError("a turn board is exactly 4 cards")
    turn_combos = hole_combos(dead=turn_board)
    index = {c: i for i, c in enumerate(turn_combos)}

    per_river = {}
    for r in river_cards(turn_board):
        combos5, strengths5 = board_strengths(turn_board + [r])   # C(47,2) survivors
        arr = np.full(len(turn_combos), -1, dtype=np.int32)
        for combo, s in zip(combos5, strengths5):
            arr[index[combo]] = s
        per_river[r] = arr
    return turn_combos, per_river

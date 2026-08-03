"""Fixed-board combo ranking (Phase 0).

Given a 5-card river board, rank every hole combo a player could hold by its
showdown strength on that board, as a DENSE integer (0 = worst, higher = better,
equal = chop). Collapsing to dense integers is what the Phase-1 showdown wants:
it turns "who wins" into plain integer comparison, and the efficient
sorted-prefix-sum showdown needs a compact ordinal per combo.
"""

from cards import hole_combos
from evaluator import eval7


def board_strengths(board):
    """Return (combos, strengths):

    combos    : list of (a, b) hole combos valid on this board (C(47,2) = 1081)
    strengths : parallel list of dense integer strengths (higher beats lower,
                equal ties). strengths[i] is the strength of combos[i].
    """
    if len(board) != 5:
        raise ValueError("a river board is exactly 5 cards")
    board = list(board)
    combos = hole_combos(dead=board)
    keys = [eval7([a, b] + board) for (a, b) in combos]
    # Map the distinct evaluation keys onto 0..K, preserving order and ties.
    ordinal = {k: i for i, k in enumerate(sorted(set(keys)))}
    strengths = [ordinal[k] for k in keys]
    return combos, strengths

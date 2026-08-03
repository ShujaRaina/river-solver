"""Poker hand evaluation (Phase 0).

`eval5` scores a 5-card hand as a comparable tuple `(category, tiebreak)`; `eval7`
returns the best such score over all 21 five-card subsets of 7 cards. Higher
score = stronger hand; equal score = tie (a chop at showdown). The scores are
only ever compared to each other, so their exact numeric form doesn't matter --
only the ordering does.

Category codes (high to low): 8 straight flush, 7 quads, 6 full house, 5 flush,
4 straight, 3 trips, 2 two pair, 1 pair, 0 high card.

This is the correctness bedrock the whole solver rests on, so it is exhaustively
hand-checked in test_evaluator.py. It's plain and unoptimized on purpose; a fast
board-specific table can replace it later without changing the interface.
"""

from collections import Counter
from itertools import combinations

from cards import rank_value, suit

CATEGORY_NAMES = [
    "high card", "one pair", "two pair", "three of a kind", "straight",
    "flush", "full house", "four of a kind", "straight flush",
]


def _straight_high(unique_desc):
    """Top card of a 5-card straight among these distinct rank values, or None.
    Handles the wheel (A-2-3-4-5), where the ace plays low and the high card is 5.
    """
    if len(unique_desc) >= 5:
        for i in range(len(unique_desc) - 4):
            if unique_desc[i] - unique_desc[i + 4] == 4:
                return unique_desc[i]
        if {14, 5, 4, 3, 2}.issubset(unique_desc):
            return 5
    return None


def _eval5(cards):
    rvs = sorted((rank_value(c) for c in cards), reverse=True)
    flush = len({suit(c) for c in cards}) == 1
    unique_desc = sorted(set(rvs), reverse=True)
    sh = _straight_high(unique_desc)
    counts = Counter(rvs)
    pattern = sorted(counts.values(), reverse=True)
    # Ranks ordered by (multiplicity, value): e.g. KKKQQ -> (K,K,K,Q,Q), so two
    # hands of the same category compare correctly element by element.
    order = tuple(sorted(rvs, key=lambda r: (counts[r], r), reverse=True))

    if sh and flush:
        return (8, (sh,))
    if pattern == [4, 1]:
        return (7, order)
    if pattern == [3, 2]:
        return (6, order)
    if flush:
        return (5, tuple(rvs))
    if sh:
        return (4, (sh,))
    if pattern == [3, 1, 1]:
        return (3, order)
    if pattern == [2, 2, 1]:
        return (2, order)
    if pattern == [2, 1, 1, 1]:
        return (1, order)
    return (0, tuple(rvs))


def eval5(cards):
    if len(cards) != 5:
        raise ValueError("eval5 needs exactly 5 cards")
    return _eval5(cards)


def eval7(cards):
    """Best 5-card score among 7 cards (2 hole + 5 board)."""
    if len(cards) != 7:
        raise ValueError("eval7 needs exactly 7 cards")
    return max(_eval5(list(c)) for c in combinations(cards, 5))

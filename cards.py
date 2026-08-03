"""Card representation for the river solver (Phase 0).

A card is an integer 0..51 so it packs tightly for the vectorized CFR to come:

    rank_index = card // 4    (0..12 for deuce..ace)
    suit       = card % 4     (0..3)
    rank_value = rank_index + 2   (2..14, ace high)

Strings use the usual "As", "Td", "9c" form: ranks "23456789TJQKA", suits "cdhs".
So "2c" == 0 (lowest) and "As" == 51 (highest). This file is pure representation
-- no hand strength (that's evaluator.py), no ranges or betting (later phases).
"""

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
NUM_CARDS = 52


def make_card(rank_index, suit):
    return rank_index * 4 + suit


def rank_index(card):
    return card // 4


def suit(card):
    return card % 4


def rank_value(card):
    """2..14, ace high -- what hand evaluation compares on."""
    return card // 4 + 2


def parse_card(s):
    s = s.strip()
    if len(s) != 2:
        raise ValueError(f"bad card {s!r}")
    r = RANK_CHARS.find(s[0].upper())
    su = SUIT_CHARS.find(s[1].lower())
    if r < 0 or su < 0:
        raise ValueError(f"bad card {s!r}")
    return make_card(r, su)


def card_str(card):
    return RANK_CHARS[rank_index(card)] + SUIT_CHARS[suit(card)]


def parse_cards(s):
    """'As Kd 7s' -> [51, 45, 31]"""
    return [parse_card(tok) for tok in s.split()]


def cards_str(cards):
    return " ".join(card_str(c) for c in cards)


def deck():
    return list(range(NUM_CARDS))


def hole_combos(dead=()):
    """Every 2-card combo (a < b) that uses none of the `dead` cards.

    With no dead cards this is all C(52,2) = 1326 combos; on a 5-card board it
    is C(47,2) = 1081 -- the combos a player could actually hold.
    """
    dead = set(dead)
    combos = []
    for a in range(NUM_CARDS):
        if a in dead:
            continue
        for b in range(a + 1, NUM_CARDS):
            if b in dead:
                continue
            combos.append((a, b))
    return combos

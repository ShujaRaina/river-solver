"""Hand-checks for card representation (Phase 0). Run: python test_cards.py"""

from cards import (parse_card, card_str, parse_cards, hole_combos,
                   rank_value, NUM_CARDS)

# --- string <-> int round trips ---
assert parse_card("2c") == 0, "2c is the lowest card"
assert parse_card("As") == 51, "As is the highest card"
assert card_str(51) == "As"
assert card_str(0) == "2c"
for c in range(NUM_CARDS):
    assert parse_card(card_str(c)) == c, f"round trip failed for {c}"

# --- rank values ---
assert rank_value(parse_card("As")) == 14
assert rank_value(parse_card("Kd")) == 13
assert rank_value(parse_card("Td")) == 10
assert rank_value(parse_card("2c")) == 2

# --- combos and card removal ---
assert len(hole_combos()) == 1326, "C(52,2)"
assert len(hole_combos(dead=parse_cards("Ah Kd 7s 2c 9h"))) == 1081, "C(47,2)"
# no combo may contain a dead card
dead = set(parse_cards("Ah Kd 7s 2c 9h"))
assert all(a not in dead and b not in dead for a, b in hole_combos(dead=dead))

print("all card checks passed")

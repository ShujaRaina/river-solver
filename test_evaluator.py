"""Hand-checks for hand evaluation (Phase 0). Run: python test_evaluator.py

Worked out by hand -- this is the bedrock the solver's showdown rests on.
"""

from cards import parse_cards
from evaluator import eval5, eval7


def h(s):
    return parse_cards(s)


# One representative of every category, strongest to weakest.
royal    = eval5(h("As Ks Qs Js Ts"))   # straight flush (8)
sf9      = eval5(h("9s 8s 7s 6s 5s"))   # straight flush (8), lower
quads    = eval5(h("As Ac Ad Ah Ks"))   # four of a kind (7)
boat     = eval5(h("As Ac Ad Ks Kc"))   # full house (6)
flush    = eval5(h("As Ks 9s 5s 2s"))   # flush (5)
straight = eval5(h("As Kd Qc Jh Ts"))   # straight (4), ace high
wheel    = eval5(h("As 2c 3d 4h 5s"))   # straight (4), five high
trips    = eval5(h("As Ac Ad Ks Qc"))   # three of a kind (3)
twopair  = eval5(h("As Ac Ks Kc Qd"))   # two pair (2)
pair     = eval5(h("As Ac Ks Qd Jc"))   # one pair (1)
high     = eval5(h("As Kd Qc Jh 9s"))   # high card (0)

# categories
assert (royal[0], sf9[0], quads[0], boat[0]) == (8, 8, 7, 6)
assert (flush[0], straight[0], wheel[0]) == (5, 4, 4)
assert (trips[0], twopair[0], pair[0], high[0]) == (3, 2, 1, 0)

# strict ranking top to bottom, all distinct
ladder = [royal, sf9, quads, boat, flush, straight, wheel, trips, twopair, pair, high]
assert ladder == sorted(ladder, reverse=True), "categories must rank in order"
assert len(set(ladder)) == len(ladder), "no two of these should tie"

# within-category tie-breaks
assert wheel < straight, "the wheel is the weakest straight"
assert straight > eval5(h("Ks Qd Jc Th 9s")), "ace-high straight beats king-high"
assert eval5(h("As Ac Ks Qd Jc")) > eval5(h("As Ac Ks Qd Tc")), "pair, kicker J>T"
assert eval5(h("As Ac 5d 3c 2h")) > eval5(h("Ks Kc 5d 3c 2h")), "pair of aces > kings"
assert eval5(h("As Ac Kd Kc Qs")) > eval5(h("As Ac Qd Qc Ks")), "aces-up > kings-up"

# eval7 picks the best 5 of 7
assert eval7(h("As Ks 9s 5s 2s 7d 3c"))[0] == 5, "flush is present in the 7"
assert eval7(h("As Ac Ad Kh Kd 2c 7s"))[0] == 6, "full house AAA KK"
assert eval7(h("As Ac Ad Ah Kd 2c 7s"))[0] == 7, "quad aces"
assert eval7(h("5h 6h 7h 8h 9h 2c As"))[0] == 8, "9-high straight flush"

print("all evaluator checks passed")

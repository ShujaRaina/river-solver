"""Hand-checks for range notation (Phase 5). Run: python test_ranges.py"""

import numpy as np

from cards import parse_cards, hole_combos
from ranges import parse_range

# Board: Ah Kd 7s 2c 9h -> blocks one each of A, K, 7, 9.
board = parse_cards("Ah Kd 7s 2c 9h")
combos = hole_combos(dead=board)


def n(spec):
    return int((parse_range(spec, combos) > 0).sum())


# pairs, with card removal (C(4-b, 2) where b = board cards of that rank)
assert n("QQ") == 6, "no queens on board -> C(4,2)"
assert n("AA") == 3, "Ah on board -> C(3,2)"
assert n("KK") == 3 and n("77") == 3 and n("99") == 3

# suited / offsuit / both
assert n("AKs") == 2, "AsKs and AcKc valid (AhKh, AdKd blocked)"
assert n("QJo") == 12, "no Q or J on board -> 12 offsuit"
assert n("QJ") == 16, "suited + offsuit"

# plus and dash ranges
assert n("77+") == 3 + 6 + 3 + 6 + 6 + 6 + 3 + 3, "77,88,99,TT,JJ,QQ,KK,AA with removal"
assert set(np.flatnonzero(parse_range("QQ-TT", combos))) == \
       set(np.flatnonzero(parse_range("QQ, JJ, TT", combos))), "dash == explicit list"
assert n("ATs+") == n("AKs, AQs, AJs, ATs"), "ATs+ expands upward"

# per-combo weights
w = parse_range("A5s:0.5", combos)
assert abs(w.sum() - 1.5) < 1e-9, "A5s = 3 valid combos (Ah5h blocked) x 0.5"

# later token overrides earlier
w = parse_range("AK, AKs:0.25", combos)
aks_mask = parse_range("AKs", combos) > 0
ako_mask = parse_range("AKo", combos) > 0
assert np.all(w[aks_mask] == 0.25) and np.all(w[ako_mask] == 1.0)

print("all range checks passed")

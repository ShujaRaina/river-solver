"""Gold-standard cross-check: CFR value == exact LP value (Phase 4+).

Run: python test_lp.py
"""

import numpy as np

from cards import parse_cards
from solver import Solver
from lp_verify import lp_value, _key

BOARD = "Ts Jh Qc 2d 7s"
P0 = ["Ah Kh", "3c 3d"]          # nut straight (A-K-Q-J-T) and a weak pair
P1 = ["Ah Ks", "8c 8d"]          # nut straight (shares Ah with P0's -> a blocked pair)
POT, STACK, FRACS = 10.0, 10.0, [1.0]     # SPR 1 -> a single all-in; shallow tree

# --- exact value via LP over pure strategies ---
lp_v, r, c = lp_value(BOARD, P0, P1, POT, STACK, FRACS)
print(f"LP exact  value to P0 = {lp_v:+.5f}   ({r}x{c} pure-strategy matrix)")

# --- CFR value on the same instance ---
board = parse_cards(BOARD)
s = Solver(board, base_pot=POT, stack=STACK, fractions=tuple(FRACS))
idx = {cmb: i for i, cmb in enumerate(s.combos)}
r0 = np.zeros(s.N); r1 = np.zeros(s.N)
for h in P0:
    r0[idx[_key(h)]] = 1.0
for h in P1:
    r1[idx[_key(h)]] = 1.0
s.train(4000, range0=r0, range1=r1, plus=True)
Z = float(r0 @ (np.asarray(s.C, float) @ r1))
cfr_v = float(s.game_values()[0]) / Z
print(f"CFR       value to P0 = {cfr_v:+.5f}   (4000 CFR+ iters, {int(Z)} live pairs)")

gap = abs(cfr_v - lp_v)
print(f"gap = {gap:.6f}")
assert gap < 5e-3, f"CFR must converge to the exact LP value (gap {gap})"
print("PASS: CFR converges to the exact equilibrium value")

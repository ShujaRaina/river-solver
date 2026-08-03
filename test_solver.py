"""Checks for the vectorized CFR solver (Phase 3). Run: python test_solver.py

The primary check is the one that mattered for Leduc: exploitability -> 0.
Trains a real spot briefly (~200 CFR+ iters) and asserts the solve is dropping
and well-behaved, plus two structural invariants.
"""

import numpy as np

from cards import parse_cards
from solver import Solver

board = parse_cards("Ah Kd 7s 2c 9h")
s = Solver(board)

# Exploitability starts high, stays >= 0, and falls with training.
s.train(5, plus=True)
e_early = s.exploitability()
s.train(195, plus=True)                      # 200 iterations total
e_late = s.exploitability()

assert e_late >= 0.0, "exploitability is a nonneg gap"
assert e_late < e_early, "exploitability must fall with training"
assert e_late < 0.05 * s.base_pot, f"expected < 5% of pot, got {100*e_late/s.base_pot:.2f}%"

# Constant-sum invariant: under any profile, u0 + u1 == base_pot * Z, where Z is
# the total compatible-pair mass. The dead money is split among valid matchups.
u0, u1 = s.game_values()
Z = s.range0 @ (s.C @ s.range1)
assert abs((u0 + u1) - s.base_pot * Z) < 1e-6 * abs(s.base_pot * Z), "constant-sum"

# A best response can never do worse than playing the strategy itself.
b0, b1 = s.best_response_values()
assert b0 >= u0 - 1e-6 and b1 >= u1 - 1e-6, "best response >= strategy value"

print(f"all solver checks passed  (exploitability {100*e_late/s.base_pot:.2f}% of pot @ 200 iters)")

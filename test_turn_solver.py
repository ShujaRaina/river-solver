"""Convergence check for the turn solver (Phase T2). Run: python test_turn_solver.py

The definitive correctness test, same as everywhere else: exploitability -> 0.
Trains a small turn instance (single all-in bet, so the tree is shallow) and
asserts the solve is dropping and well-behaved.
"""

from cards import parse_cards
from turn_solver import TurnSolver

turn = parse_cards("Ts 9h 2c 5d")
s = TurnSolver(turn, base_pot=20.0, stack=80.0, fractions=(1.0,))

s.train(5)
e_early = s.exploitability()
s.train(25)                        # 30 iterations total
e_late = s.exploitability()

assert e_late >= 0.0, "exploitability is a nonneg gap"
assert e_late < e_early, "exploitability must fall with training"
assert e_late < 1.0, f"expected well under 1 bb, got {e_late:.3f}"

print(f"turn solver converges: exploitability {e_early:.2f} -> {e_late:.3f} bb "
      f"(5 -> 30 iters, {s.n_river} river runouts)")

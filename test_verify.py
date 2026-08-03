"""Phase 4 verification. Run: python test_verify.py

Two independent guards on the solve, mirroring the Leduc discipline:
  1. the exploitability metric must DISCRIMINATE -- a uniform (untrained)
     strategy scores far higher than the solved one (a metric that only ever
     reads ~0 is untrustworthy);
  2. an independent Monte-Carlo playout of the average strategy must reproduce
     the solver's computed game value.
"""

import numpy as np

from cards import parse_cards
from solver import Solver
from verify import mc_game_value

board = parse_cards("Ah Kd 7s 2c 9h")

s = Solver(board)
s.train(0)                       # sets ranges, no learning -> uniform average
e_uniform = s.exploitability()
s.train(150)                     # now solve
e_solved = s.exploitability()

assert e_solved >= 0.0
assert e_uniform > 3 * e_solved, \
    f"metric must discriminate: uniform {e_uniform:.2f}, solved {e_solved:.2f}"

# Independent MC value must match the computed game value.
Z = s.range0 @ (s.C @ s.range1)
expected = s.game_values()[0] / Z
mc_mean, mc_se = mc_game_value(s, n_samples=100_000, seed=1)
assert abs(mc_mean - expected) < 4 * mc_se, \
    f"MC {mc_mean:.4f} vs computed {expected:.4f} (4se {4*mc_se:.4f})"

print(f"all verify checks passed  "
      f"(expl uniform {e_uniform:.2f} vs solved {e_solved:.2f} chips/hand;  "
      f"MC value {mc_mean:.3f} == computed {expected:.3f} +/- {4*mc_se:.3f})")

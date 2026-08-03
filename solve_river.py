"""Solve a river spot and print the strategy. Run: python solve_river.py

The one-command entry point: pick a board, pot/stack, and two ranges (in poker
notation), run CFR+ to equilibrium, and read out the result.
"""

from cards import parse_cards
from solver import Solver
from ranges import parse_range
from readout import describe

BOARD = "Ah Kd 7s 2c 9h"
POT, STACK = 20.0, 80.0
ITERS = 800

# Real, hand-specified ranges (not uniform). These are plausible inputs for the
# line, not equilibria derived from earlier streets -- that's the caller's job.
RANGE0 = "AA,KK,QQ,AK,AQs,AJs,ATs,KQs,KJs,QJs,JTs,99,88,77,A5s,A4s"   # out of position
RANGE1 = "AA,KK,99,77,AK,AQ,AJ,KQ,QJs,JTs,T9s,T8s,98s,A5s"           # in position


def main():
    board = parse_cards(BOARD)
    s = Solver(board, base_pot=POT, stack=STACK, fractions=(0.33, 0.66))
    r0 = parse_range(RANGE0, s.combos)
    r1 = parse_range(RANGE1, s.combos)
    print(f"river {BOARD}   (pot {POT:g}, stack {STACK:g})")
    print(f"P0 range: {int((r0 > 0).sum())} combos   P1 range: {int((r1 > 0).sum())} combos")
    print(f"solving ({ITERS} CFR+ iters)...\n")
    s.train(ITERS, range0=r0, range1=r1, plus=True)
    describe(s, hands=["As Ac", "Kc Qc", "Js Ts", "As 5s", "9s 9c"])


if __name__ == "__main__":
    main()

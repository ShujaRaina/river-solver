"""Solve a river spot and print the strategy. Run: python solve_river.py

The one-command entry point: build the game for a board + pot/stack, run CFR+ to
equilibrium, and read out the result.
"""

from cards import parse_cards
from solver import Solver
from readout import describe

BOARD = "Ah Kd 7s 2c 9h"
POT, STACK = 20.0, 80.0
ITERS = 1000


def main():
    board = parse_cards(BOARD)
    s = Solver(board, base_pot=POT, stack=STACK, fractions=(0.33, 0.66))
    print(f"solving river {BOARD}  (pot {POT:g}, stack {STACK:g}, uniform ranges, "
          f"{ITERS} CFR+ iters)...\n")
    s.train(ITERS, plus=True)
    describe(s, hands=["As Ac", "Ac 5c", "Kc Qc", "Js Ts", "5d 4d"])


if __name__ == "__main__":
    main()

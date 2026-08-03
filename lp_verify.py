"""Exact equilibrium via linear programming, for cross-checking CFR (Phase 4+).

An independent gold-standard: take a SMALL river instance and compute its exact
game value a completely different way -- enumerate every pure strategy of each
player, build the normal-form payoff matrix, and solve the zero-sum matrix game
with an LP (minimax). If the CFR solver's converged value matches this exact
value, CFR is finding the true equilibrium (not just reporting a low
exploitability from its own -- possibly buggy -- best-response code).

We keep it a two-player zero-sum in P0's payoff: the game is constant-sum
(u0 + u1 = dead money), so P1 maximizing its own payoff == minimizing P0's.
Small instance = 2 hands each + SPR 1 (a single all-in), so the tree is shallow
and the pure-strategy sets stay tiny.
"""

from itertools import product

import numpy as np
from scipy.optimize import linprog

from cards import parse_cards
from board import board_strengths
from betting import build_tree, walk
from verify import _p0_payoff


def _key(hand):
    a, b = parse_cards(hand)
    return (a, b) if a < b else (b, a)


def _solve_matrix_game(M):
    """Value to the row player (P0) of the zero-sum matrix game M (P0's payoff),
    i.e. max_p min_b (p^T M)_b, via LP."""
    R, C = M.shape
    c = np.zeros(R + 1); c[-1] = -1.0                  # maximize v == minimize -v
    A_ub = np.zeros((C, R + 1))                         # (p^T M)_b >= v  for all b
    A_ub[:, :R] = -M.T
    A_ub[:, R] = 1.0
    A_eq = np.zeros((1, R + 1)); A_eq[0, :R] = 1.0      # sum p = 1
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(C), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * R + [(None, None)], method="highs")
    if not res.success:
        raise RuntimeError("LP failed: " + res.message)
    return res.x[-1]


def lp_value(board_str, p0_hands, p1_hands, base_pot, stack, fractions):
    board = parse_cards(board_str)
    combos, strengths = board_strengths(board)
    sidx = {c: i for i, c in enumerate(combos)}
    p0c = [_key(h) for h in p0_hands]
    p1c = [_key(h) for h in p1_hands]
    strength = {c: strengths[sidx[c]] for c in p0c + p1c}

    root = build_tree(base_pot, stack, tuple(fractions), first_actor=0)
    nodes0 = [n for n in walk(root) if not n.is_terminal() and n.player == 0]
    nodes1 = [n for n in walk(root) if not n.is_terminal() and n.player == 1]
    info0 = [(n, c) for n in nodes0 for c in p0c]       # P0's information sets
    info1 = [(n, c) for n in nodes1 for c in p1c]
    pures0 = list(product(*[range(len(n.actions)) for (n, _) in info0]))
    pures1 = list(product(*[range(len(n.actions)) for (n, _) in info1]))

    pairs = [(i, j) for i in p0c for j in p1c if not (set(i) & set(j))]
    Z = len(pairs)                                       # uniform over compatible pairs

    def playout(m0, m1, i, j):
        node = root
        while not node.is_terminal():
            a = m0[(node, i)] if node.player == 0 else m1[(node, j)]
            node = node.actions[a][1]
        return _p0_payoff(node, strength[i], strength[j])

    M = np.zeros((len(pures0), len(pures1)))
    for a, pu0 in enumerate(pures0):
        m0 = dict(zip(info0, pu0))
        for b, pu1 in enumerate(pures1):
            m1 = dict(zip(info1, pu1))
            M[a, b] = sum(playout(m0, m1, i, j) for (i, j) in pairs) / Z

    return _solve_matrix_game(M), len(pures0), len(pures1)

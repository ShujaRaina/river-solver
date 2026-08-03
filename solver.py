"""Vectorized CFR for the river subgame (Phase 3).

Ties Phases 0-2 together. The betting tree's decision nodes are public
information sets; at each we keep a per-combo strategy -- an array of shape
(N_combos, N_actions) -- and run Counterfactual Regret Minimization as array
math over the whole range at once, instead of walking one deal at a time.

Reach vectors
-------------
Down the tree we carry reach0, reach1: reach_p[i] is the probability player p
reaches this node holding combo i (its range weight times its own action
probabilities along the path). At the root, reach = the input ranges. Card
removal between the two ranges is handled at terminals by the M / C matrices
(blocked pairs zeroed), so the reaches themselves need no joint normalization.

The vector-CFR update at a decision node for the acting player p:
  * node value (per combo) = sum_a sigma[:,a] * cfv_a
  * regret[:,a] += cfv_a - node_value        (opponent's reach is already baked
                                              into cfv_a, so no extra weighting)
  * strategy_sum += weight * reach_p * sigma  (weight = t for CFR+ averaging)
For the NON-acting player, the value returned up is the plain sum over the
acting player's actions (their strategy weighting rides along in the reach).

Terminal values are dead-money-correct (you win the pot at showdown, not just
your stake) -- see `_terminal_value`.

Correctness is judged the way Leduc was: exploitability -> 0.
"""

import numpy as np

from board import board_strengths
from terminal import showdown_matrix, compatibility_mask
from betting import build_tree, walk


class Solver:
    def __init__(self, board, base_pot=20.0, stack=80.0,
                 fractions=(0.33, 0.66), first_actor=0):
        self.combos, strengths = board_strengths(board)
        self.N = len(self.combos)
        self.M = showdown_matrix(self.combos, strengths)   # +1/0/-1, blocked -> 0
        self.C = compatibility_mask(self.combos)           # 1 if no shared card
        self.base_pot = base_pot
        self.root = build_tree(base_pot, stack, fractions, first_actor)

        self.regret, self.strat_sum = {}, {}
        for n in walk(self.root):
            if not n.is_terminal():
                shape = (self.N, len(n.actions))
                self.regret[n] = np.zeros(shape)
                self.strat_sum[n] = np.zeros(shape)

    # --- strategies -------------------------------------------------------
    def strategy(self, node):
        """Current strategy via regret matching, per combo."""
        r = np.maximum(self.regret[node], 0.0)
        total = r.sum(axis=1, keepdims=True)
        A = r.shape[1]
        return np.where(total > 0, r / np.where(total > 0, total, 1.0), 1.0 / A)

    def average_strategy(self, node):
        s = self.strat_sum[node]
        total = s.sum(axis=1, keepdims=True)
        A = s.shape[1]
        return np.where(total > 0, s / np.where(total > 0, total, 1.0), 1.0 / A)

    # --- terminal values (per hero combo, weighted by opponent reach) -----
    def _terminal_value(self, node, opp_reach, hero):
        if node.kind == "showdown":
            pot, stake = node.pot, node.stake
            return 0.5 * pot * (self.M @ opp_reach) + \
                (0.5 * pot - stake) * (self.C @ opp_reach)
        # fold: hero wins the pot (minus own contribution) unless hero folded
        c = node.contrib[hero]
        amount = (node.pot - c) if node.folder != hero else -c
        return amount * (self.C @ opp_reach)

    # --- CFR traversal (updates regrets + strategy sums) ------------------
    def _traverse(self, node, reach0, reach1):
        if node.is_terminal():
            return (self._terminal_value(node, reach1, 0),
                    self._terminal_value(node, reach0, 1))

        p = node.player
        sigma = self.strategy(node)
        reach_p = reach0 if p == 0 else reach1
        self.strat_sum[node] += self._weight * reach_p[:, None] * sigma

        cfv0_a, cfv1_a = [], []
        for a, (_, child) in enumerate(node.actions):
            if p == 0:
                c0, c1 = self._traverse(child, reach0 * sigma[:, a], reach1)
            else:
                c0, c1 = self._traverse(child, reach0, reach1 * sigma[:, a])
            cfv0_a.append(c0)
            cfv1_a.append(c1)
        cfv0_a = np.stack(cfv0_a, axis=1)
        cfv1_a = np.stack(cfv1_a, axis=1)

        if p == 0:
            node_cfv0 = (sigma * cfv0_a).sum(axis=1)
            self.regret[node] += cfv0_a - node_cfv0[:, None]
            if self._plus:
                np.maximum(self.regret[node], 0.0, out=self.regret[node])
            return node_cfv0, cfv1_a.sum(axis=1)
        else:
            node_cfv1 = (sigma * cfv1_a).sum(axis=1)
            self.regret[node] += cfv1_a - node_cfv1[:, None]
            if self._plus:
                np.maximum(self.regret[node], 0.0, out=self.regret[node])
            return cfv0_a.sum(axis=1), node_cfv1

    def train(self, iters, range0=None, range1=None, plus=True):
        self.range0 = np.ones(self.N) if range0 is None else np.asarray(range0, float)
        self.range1 = np.ones(self.N) if range1 is None else np.asarray(range1, float)
        self._plus = plus
        for t in range(1, iters + 1):
            self._weight = float(t) if plus else 1.0
            self._traverse(self.root, self.range0, self.range1)

    # --- evaluation: values under avg strategy, best responses, expl ------
    def _values_avg(self, node, reach0, reach1):
        if node.is_terminal():
            return (self._terminal_value(node, reach1, 0),
                    self._terminal_value(node, reach0, 1))
        p = node.player
        sigma = self.average_strategy(node)
        cfv0_a, cfv1_a = [], []
        for a, (_, child) in enumerate(node.actions):
            if p == 0:
                c0, c1 = self._values_avg(child, reach0 * sigma[:, a], reach1)
            else:
                c0, c1 = self._values_avg(child, reach0, reach1 * sigma[:, a])
            cfv0_a.append(c0)
            cfv1_a.append(c1)
        cfv0_a = np.stack(cfv0_a, axis=1)
        cfv1_a = np.stack(cfv1_a, axis=1)
        if p == 0:
            return (sigma * cfv0_a).sum(axis=1), cfv1_a.sum(axis=1)
        return cfv0_a.sum(axis=1), (sigma * cfv1_a).sum(axis=1)

    def _best_response(self, node, opp_reach, exploiter):
        """Value to `exploiter` (per combo) when it best-responds and the
        opponent plays its average strategy."""
        if node.is_terminal():
            return self._terminal_value(node, opp_reach, exploiter)
        if node.player == exploiter:
            vals = [self._best_response(child, opp_reach, exploiter)
                    for _, child in node.actions]
            return np.max(np.stack(vals, axis=1), axis=1)   # best action per combo
        sigma = self.average_strategy(node)
        total = np.zeros(self.N)
        for a, (_, child) in enumerate(node.actions):
            total += self._best_response(child, opp_reach * sigma[:, a], exploiter)
        return total

    def game_values(self):
        u0, u1 = self._values_avg(self.root, self.range0, self.range1)
        return self.range0 @ u0, self.range1 @ u1

    def best_response_values(self):
        b0 = self.range0 @ self._best_response(self.root, self.range1, 0)
        b1 = self.range1 @ self._best_response(self.root, self.range0, 1)
        return b0, b1

    def exploitability(self):
        """NashConv per hand: how much both players gain by best-responding to
        the current average strategy, normalized by total compatible mass.
        -> 0 at equilibrium; always >= 0."""
        u0, u1 = self.game_values()
        b0, b1 = self.best_response_values()
        Z = self.range0 @ (self.C @ self.range1)
        return ((b0 - u0) + (b1 - u1)) / Z

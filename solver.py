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
        self.strengths = np.asarray(strengths)
        self.N = len(self.combos)
        # float32 for the hot matmuls (M/C @ reaches); accumulators stay float64.
        self.M = showdown_matrix(self.combos, strengths).astype(np.float32)
        self.C = compatibility_mask(self.combos).astype(np.float32)
        self.base_pot = base_pot
        self.root = build_tree(base_pot, stack, fractions, first_actor)

        self.regret, self.strat_sum = {}, {}
        for n in walk(self.root):
            if not n.is_terminal():
                shape = (self.N, len(n.actions))
                self.regret[n] = np.zeros(shape)
                self.strat_sum[n] = np.zeros(shape)

        # Terminal batching (the speed path): index every terminal and precompute
        # its value coefficients, so a whole iteration's terminal values are a few
        # big matmuls (M/C read once) instead of a matvec per terminal.
        self.terminals = [n for n in walk(self.root) if n.is_terminal()]
        self.tindex = {n: i for i, n in enumerate(self.terminals)}
        T = self.T = len(self.terminals)
        a, b = np.zeros(T, np.float32), np.zeros(T, np.float32)      # showdown coeffs
        amt0, amt1 = np.zeros(T, np.float32), np.zeros(T, np.float32)  # fold coeffs
        sd, fd = [], []
        for i, n in enumerate(self.terminals):
            if n.kind == "showdown":
                sd.append(i)
                a[i] = 0.5 * n.pot
                b[i] = 0.5 * n.pot - n.stake
            else:
                fd.append(i)
                c0, c1 = n.contrib
                amt0[i] = (n.pot - c0) if n.folder != 0 else -c0
                amt1[i] = (n.pot - c1) if n.folder != 1 else -c1
        self._sd, self._fd = np.array(sd, dtype=int), np.array(fd, dtype=int)
        self._a, self._b, self._amt0, self._amt1 = a, b, amt0, amt1
        self._R0 = np.empty((self.N, T), np.float32)
        self._R1 = np.empty((self.N, T), np.float32)
        self._cfv0 = np.empty((self.N, T), np.float32)
        self._cfv1 = np.empty((self.N, T), np.float32)
        self._t = 0                        # persistent iteration count (for chunked training)

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

    # --- reference recursive CFR iteration (a matvec per terminal) --------
    # Kept as the readable spec and differential oracle; train() uses the
    # batched _down/_terminal_cfvs/_up path, which is ~5x faster and identical.
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

    # --- batched iteration (down: reaches -> batch matmuls -> up: regrets) ---
    def _down(self, node, reach0, reach1, sig):
        """Propagate reaches to terminals (store their columns), accumulate the
        strategy sum, and cache each node's current strategy for the up-pass."""
        if node.is_terminal():
            t = self.tindex[node]
            self._R0[:, t] = reach0
            self._R1[:, t] = reach1
            return
        p = node.player
        sigma = self.strategy(node)
        sig[node] = sigma
        reach_p = reach0 if p == 0 else reach1
        self.strat_sum[node] += self._weight * reach_p[:, None] * sigma
        for a, (_, child) in enumerate(node.actions):
            if p == 0:
                self._down(child, reach0 * sigma[:, a], reach1, sig)
            else:
                self._down(child, reach0, reach1 * sigma[:, a], sig)

    def _terminal_cfvs(self):
        """All terminal counterfactual values in a few matmuls (M/C read once)."""
        CR0, CR1 = self.C @ self._R0, self.C @ self._R1
        sd, fd = self._sd, self._fd
        if sd.size:
            MR0, MR1 = self.M @ self._R0[:, sd], self.M @ self._R1[:, sd]
            self._cfv0[:, sd] = self._a[sd] * MR1 + self._b[sd] * CR1[:, sd]
            self._cfv1[:, sd] = self._a[sd] * MR0 + self._b[sd] * CR0[:, sd]
        if fd.size:
            self._cfv0[:, fd] = self._amt0[fd] * CR1[:, fd]
            self._cfv1[:, fd] = self._amt1[fd] * CR0[:, fd]

    def _up(self, node, sig):
        """Combine terminal cfvs up the tree and update the acting player's
        regrets. `sig` holds each node's strategy cached from the down-pass."""
        if node.is_terminal():
            t = self.tindex[node]
            return self._cfv0[:, t], self._cfv1[:, t]
        p = node.player
        sigma = sig[node]
        c0a, c1a = [], []
        for _, child in node.actions:
            v0, v1 = self._up(child, sig)
            c0a.append(v0)
            c1a.append(v1)
        c0a, c1a = np.stack(c0a, axis=1), np.stack(c1a, axis=1)
        if p == 0:
            nv0 = (sigma * c0a).sum(axis=1)
            self.regret[node] += c0a - nv0[:, None]
            if self._plus:
                np.maximum(self.regret[node], 0.0, out=self.regret[node])
            return nv0, c1a.sum(axis=1)
        nv1 = (sigma * c1a).sum(axis=1)
        self.regret[node] += c1a - nv1[:, None]
        if self._plus:
            np.maximum(self.regret[node], 0.0, out=self.regret[node])
        return c0a.sum(axis=1), nv1

    def train(self, iters, range0=None, range1=None, plus=True):
        self.range0 = np.ones(self.N) if range0 is None else np.asarray(range0, float)
        self.range1 = np.ones(self.N) if range1 is None else np.asarray(range1, float)
        self._plus = plus
        for _ in range(iters):
            self._t += 1                   # persists across calls: chunked training stays correct
            self._weight = float(self._t) if plus else 1.0
            sig = {}
            self._down(self.root, self.range0, self.range1, sig)
            self._terminal_cfvs()
            self._up(self.root, sig)

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

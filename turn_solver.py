"""Vectorized CFR+ for the turn subgame (Phase T2).

Turn betting -> chance (river) -> river betting -> showdown, solved over the
1128-combo turn universe. The new piece vs the river solver is the chance node.

Chance node (the Leduc trap, vectorized). When the river r lands, combos holding
r are impossible, and the true value to a hero holding i is

    cfv(i) = (1/44) * sum_{r not in i}  sum_{j: compat(i,j), r not in j}  r_opp(j) v(i,j,r)

(44 = 52 - 4 board - 4 hole cards; the river is uniform over what's left). That
equals: weight every river by 1/44, mask combos containing r on BOTH sides
(valid[r]), recurse into the river subgame with per-river strengths, and sum.

Efficiency: the river betting structure is card-independent, so we carry reaches
as (n_river, N) and batch all 48 rivers through one river-subtree pass. Regrets
in the river part are indexed by river card -- (n_river, N, actions) -- since a
player sees the river and plays it differently.

Memory scales with (turn-close points) x 48 x combos x actions, so this fits a
laptop only for a modest bet abstraction (few sizes); float32 matrices help.
Correctness is judged the usual way: exploitability -> 0.
"""

import numpy as np

from terminal import compatibility_mask
from turn_eval import turn_strengths
from turn_tree import build_turn_tree
from showdown_fast import sorted_showdown_batched


def _regret_match(reg):
    """Regret matching over the last axis; works for (N,A) and (n_river,N,A)."""
    pos = np.maximum(reg, 0.0)
    total = pos.sum(axis=-1, keepdims=True)
    A = reg.shape[-1]
    return np.where(total > 0, pos / np.where(total > 0, total, 1.0), 1.0 / A)


class TurnSolver:
    def __init__(self, turn_board, base_pot=20.0, stack=80.0,
                 fractions=(1.0,), first_actor=0):
        self.turn_combos, per_river = turn_strengths(turn_board)
        self.N = len(self.turn_combos)
        self.rivers = list(per_river.keys())
        self.n_river = len(self.rivers)
        self.base_pot = base_pot
        self.chance_weight = 1.0 / (52 - len(turn_board) - 4)      # 1/44

        self.C = compatibility_mask(self.turn_combos).astype(np.float32)
        # Showdowns use the O(N) sorted method (no per-river N*N matrix), so we
        # keep only the per-river strengths + each combo's two cards.
        self.strengths = np.stack([per_river[r] for r in self.rivers])   # (n_river, N)
        self.valid = (self.strengths >= 0).astype(np.float32)            # (n_river, N)
        self.card_a = np.array([c[0] for c in self.turn_combos])
        self.card_b = np.array([c[1] for c in self.turn_combos])

        self.root = build_turn_tree(base_pot, stack, fractions, first_actor)
        self.regret, self.strat = {}, {}
        self._alloc(self.root, river=False)

    def _alloc(self, node, river):
        if node.is_terminal():
            return
        if node.is_chance():
            self._alloc(node.river_root, river=True)
            return
        A = len(node.actions)
        shape = (self.n_river, self.N, A) if river else (self.N, A)
        self.regret[node] = np.zeros(shape, np.float32)
        self.strat[node] = np.zeros(shape, np.float32)
        for _, ch in node.actions:
            self._alloc(ch, river)

    def _showdown(self, R):
        """(M@R, C@R) per river via the sorted-prefix-sum showdown -- no N*N matrix."""
        return sorted_showdown_batched(self.strengths, self.card_a, self.card_b, R)

    # --- terminal values -------------------------------------------------
    def _fold(self, node, opp_reach, hero, C_apply):
        c = node.contrib[hero]
        amount = (node.pot - c) if node.folder != hero else -c
        return amount * C_apply(opp_reach)

    # --- CFR traversal: turn part (reaches are (N,)) ----------------------
    def _turn(self, node, r0, r1):
        if node.is_terminal():                                     # only folds on the turn
            return (self._fold(node, r1, 0, lambda x: self.C @ x),
                    self._fold(node, r0, 1, lambda x: self.C @ x))
        if node.is_chance():
            R0 = self.valid * r0[None, :]                          # (n_river, N)
            R1 = self.valid * r1[None, :]
            c0, c1 = self._river(node.river_root, R0, R1)
            w = self.chance_weight
            return w * (self.valid * c0).sum(0), w * (self.valid * c1).sum(0)

        reg = self.regret[node]
        sigma = _regret_match(reg)
        p = node.player
        reach_p = r0 if p == 0 else r1
        self.strat[node] += self._weight * reach_p[:, None] * sigma
        c0a, c1a = [], []
        for a, (_, ch) in enumerate(node.actions):
            a0, a1 = (self._turn(ch, r0 * sigma[:, a], r1) if p == 0
                      else self._turn(ch, r0, r1 * sigma[:, a]))
            c0a.append(a0); c1a.append(a1)
        c0a, c1a = np.stack(c0a, 1), np.stack(c1a, 1)
        if p == 0:
            nv = (sigma * c0a).sum(1)
            reg += c0a - nv[:, None]; np.maximum(reg, 0, out=reg)
            return nv, c1a.sum(1)
        nv = (sigma * c1a).sum(1)
        reg += c1a - nv[:, None]; np.maximum(reg, 0, out=reg)
        return c0a.sum(1), nv

    # --- CFR traversal: river part (reaches are (n_river, N)) ------------
    def _river(self, node, R0, R1):
        if node.is_terminal():
            if node.kind == "showdown":
                pot, stake = node.pot, node.stake
                MR1, CR1 = self._showdown(R1)
                MR0, CR0 = self._showdown(R0)
                return (0.5 * pot * MR1 + (0.5 * pot - stake) * CR1,
                        0.5 * pot * MR0 + (0.5 * pot - stake) * CR0)
            return (self._fold(node, R1, 0, lambda x: x @ self.C),
                    self._fold(node, R0, 1, lambda x: x @ self.C))

        reg = self.regret[node]
        sigma = _regret_match(reg)
        p = node.player
        reach_p = R0 if p == 0 else R1
        self.strat[node] += self._weight * reach_p[:, :, None] * sigma
        c0a, c1a = [], []
        for a, (_, ch) in enumerate(node.actions):
            a0, a1 = (self._river(ch, R0 * sigma[:, :, a], R1) if p == 0
                      else self._river(ch, R0, R1 * sigma[:, :, a]))
            c0a.append(a0); c1a.append(a1)
        c0a, c1a = np.stack(c0a, -1), np.stack(c1a, -1)
        if p == 0:
            nv = (sigma * c0a).sum(-1)
            reg += c0a - nv[..., None]; np.maximum(reg, 0, out=reg)
            return nv, c1a.sum(-1)
        nv = (sigma * c1a).sum(-1)
        reg += c1a - nv[..., None]; np.maximum(reg, 0, out=reg)
        return c0a.sum(-1), nv

    def train(self, iters, range0=None, range1=None):
        self.range0 = (np.ones(self.N, np.float32) if range0 is None
                       else np.asarray(range0, np.float32))
        self.range1 = (np.ones(self.N, np.float32) if range1 is None
                       else np.asarray(range1, np.float32))
        for t in range(1, iters + 1):
            self._weight = float(t)
            self._turn(self.root, self.range0, self.range1)

    # --- evaluation: average-strategy values and best responses ----------
    def _avg(self, node, river):
        s = self.strat[node]
        total = s.sum(axis=-1, keepdims=True)
        A = s.shape[-1]
        return np.where(total > 0, s / np.where(total > 0, total, 1.0), 1.0 / A)

    def _val_turn(self, node, r0, r1):
        if node.is_terminal():
            return (self._fold(node, r1, 0, lambda x: self.C @ x),
                    self._fold(node, r0, 1, lambda x: self.C @ x))
        if node.is_chance():
            R0, R1 = self.valid * r0[None, :], self.valid * r1[None, :]
            c0, c1 = self._val_river(node.river_root, R0, R1)
            w = self.chance_weight
            return w * (self.valid * c0).sum(0), w * (self.valid * c1).sum(0)
        sigma = self._avg(node, False)
        p = node.player
        c0a, c1a = [], []
        for a, (_, ch) in enumerate(node.actions):
            a0, a1 = (self._val_turn(ch, r0 * sigma[:, a], r1) if p == 0
                      else self._val_turn(ch, r0, r1 * sigma[:, a]))
            c0a.append(a0); c1a.append(a1)
        c0a, c1a = np.stack(c0a, 1), np.stack(c1a, 1)
        if p == 0:
            return (sigma * c0a).sum(1), c1a.sum(1)
        return c0a.sum(1), (sigma * c1a).sum(1)

    def _val_river(self, node, R0, R1):
        if node.is_terminal():
            if node.kind == "showdown":
                pot, stake = node.pot, node.stake
                MR1, CR1 = self._showdown(R1)
                MR0, CR0 = self._showdown(R0)
                return (0.5 * pot * MR1 + (0.5 * pot - stake) * CR1,
                        0.5 * pot * MR0 + (0.5 * pot - stake) * CR0)
            return (self._fold(node, R1, 0, lambda x: x @ self.C),
                    self._fold(node, R0, 1, lambda x: x @ self.C))
        sigma = self._avg(node, True)
        p = node.player
        c0a, c1a = [], []
        for a, (_, ch) in enumerate(node.actions):
            a0, a1 = (self._val_river(ch, R0 * sigma[:, :, a], R1) if p == 0
                      else self._val_river(ch, R0, R1 * sigma[:, :, a]))
            c0a.append(a0); c1a.append(a1)
        c0a, c1a = np.stack(c0a, -1), np.stack(c1a, -1)
        if p == 0:
            return (sigma * c0a).sum(-1), c1a.sum(-1)
        return c0a.sum(-1), (sigma * c1a).sum(-1)

    def _br_turn(self, node, opp_reach, ex):
        if node.is_terminal():
            return self._fold(node, opp_reach, ex, lambda x: self.C @ x)
        if node.is_chance():
            Ropp = self.valid * opp_reach[None, :]
            v = self._br_river(node.river_root, Ropp, ex)
            return self.chance_weight * (self.valid * v).sum(0)
        if node.player == ex:
            vals = [self._br_turn(ch, opp_reach, ex) for _, ch in node.actions]
            return np.max(np.stack(vals, 1), axis=1)
        sigma = self._avg(node, False)
        return sum(self._br_turn(ch, opp_reach * sigma[:, a], ex)
                   for a, (_, ch) in enumerate(node.actions))

    def _br_river(self, node, Ropp, ex):
        if node.is_terminal():
            if node.kind == "showdown":
                pot, stake = node.pot, node.stake
                MR, CR = self._showdown(Ropp)
                return 0.5 * pot * MR + (0.5 * pot - stake) * CR
            return self._fold(node, Ropp, ex, lambda x: x @ self.C)
        if node.player == ex:
            vals = [self._br_river(ch, Ropp, ex) for _, ch in node.actions]
            return np.max(np.stack(vals, -1), axis=-1)
        sigma = self._avg(node, True)
        return sum(self._br_river(ch, Ropp * sigma[:, :, a], ex)
                   for a, (_, ch) in enumerate(node.actions))

    def game_values(self):
        u0, u1 = self._val_turn(self.root, self.range0, self.range1)
        return self.range0 @ u0, self.range1 @ u1

    def exploitability(self):
        u0, u1 = self.game_values()
        b0 = self.range0 @ self._br_turn(self.root, self.range1, 0)
        b1 = self.range1 @ self._br_turn(self.root, self.range0, 1)
        Z = self.range0 @ (np.asarray(self.C, float) @ self.range1)
        return ((b0 - u0) + (b1 - u1)) / Z

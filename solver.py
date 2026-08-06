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
                 fractions=(0.33, 0.66), first_actor=0, max_nodes=None):
        self.combos, strengths = board_strengths(board)
        self.strengths = np.asarray(strengths)
        self.N = len(self.combos)
        # float32 for the hot matmuls (M/C @ reaches); accumulators stay float64.
        self.M = showdown_matrix(self.combos, strengths).astype(np.float32)
        self.C = compatibility_mask(self.combos).astype(np.float32)
        self.base_pot = base_pot
        self.root = build_tree(base_pot, stack, fractions, first_actor,
                               max_nodes=max_nodes)

        # regret / strat_sum are allocated in _build_flat() below, as batched
        # per-(depth, action-count) group arrays; the per-node dicts point at
        # views into them so the recursive reference path keeps working.
        self.regret, self.strat_sum = {}, {}

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
        self._dcfr = False                 # set per-train() call; CFR+ is the default
        self._build_flat()                 # batched-iteration schedule + storage

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

    # --- regret / strategy accumulation: CFR+ (default) or Discounted CFR -----
    # The per-iteration discount scalars (_cp, _cn, _cg) are set in train(); they
    # depend only on the iteration count, so they're the same for every node.
    def _accumulate_strategy(self, node, reach_p, sigma):
        ss = self.strat_sum[node]
        if self._dcfr:
            ss *= self._cg                          # discount old strategy mass
            ss += reach_p[:, None] * sigma          # add this iter at weight 1
        else:
            ss += self._weight * reach_p[:, None] * sigma   # CFR+ linear averaging

    def _update_regret(self, node, delta):
        R = self.regret[node]
        if self._dcfr:
            R *= np.where(R > 0, self._cp, self._cn)  # two-sided discount, then add
            R += delta
        else:
            R += delta
            if self._plus:
                np.maximum(R, 0.0, out=R)             # regret-matching+ flooring

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
        self._accumulate_strategy(node, reach_p, sigma)
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
            self._update_regret(node, c0a - nv0[:, None])
            return nv0, c1a.sum(axis=1)
        nv1 = (sigma * c1a).sum(axis=1)
        self._update_regret(node, c1a - nv1[:, None])
        return c0a.sum(axis=1), nv1

    # --- flattened, level-batched iteration -------------------------------
    # The recursive _down/_up fire ~5 tiny numpy calls per decision node; on a
    # small river tree that's dominated by per-call Python/numpy dispatch (~88%
    # of runtime is overhead, not float math). Here we group decision nodes by
    # (depth, action-count) -- players alternate strictly by depth, so a group
    # has one player -- and process a whole group in a few batched ops over a
    # (G, N, A) array. Reaches/cfvs flow through unified (n_nodes, N) buffers via
    # gather/scatter. Same math as _down/_up (verified equal), far fewer calls.
    def _build_flat(self):
        from collections import deque, defaultdict
        depth, order, dq = {self.root: 0}, [], deque([self.root])
        while dq:
            n = dq.popleft()
            order.append(n)
            if not n.is_terminal():
                for _, c in n.actions:
                    depth[c] = depth[n] + 1
                    dq.append(c)
        decisions = [n for n in order if not n.is_terminal()]
        D = len(decisions)
        gid = {n: i for i, n in enumerate(decisions)}      # decisions: 0..D-1
        for n, t in self.tindex.items():
            gid[n] = D + t                                 # terminals: D..D+T-1
        self._D = D
        self._nnodes = D + self.T
        self._term_gids = np.arange(D, D + self.T)
        self._root_gid = gid[self.root]
        self._REACH0 = np.zeros((self._nnodes, self.N))
        self._REACH1 = np.zeros((self._nnodes, self.N))
        self._CFV0 = np.zeros((self._nnodes, self.N))
        self._CFV1 = np.zeros((self._nnodes, self.N))

        buckets = defaultdict(list)
        for n in decisions:
            buckets[(depth[n], len(n.actions))].append(n)
        self._groups = []
        for key in sorted(buckets):                        # ascending depth (down order)
            nodes = buckets[key]
            _, A = key
            players = {n.player for n in nodes}
            assert len(players) == 1, "group spans both players (depth parity broke)"
            ids = np.array([gid[n] for n in nodes])
            child = np.array([[gid[c] for _, c in n.actions] for n in nodes])  # (G, A)
            R = np.zeros((len(nodes), self.N, A))
            S = np.zeros((len(nodes), self.N, A))
            for row, n in enumerate(nodes):
                self.regret[n] = R[row]                    # view: shared with batched R
                self.strat_sum[n] = S[row]                 # view
            self._groups.append({"ids": ids, "child": child,
                                 "player": nodes[0].player, "A": A, "R": R, "S": S})

    def _apply_regret(self, R, delta):
        """Batched regret update on a group's (G, N, A) array, in place."""
        if self._dcfr:
            R *= np.where(R > 0, self._cp, self._cn)
            R += delta
        else:
            R += delta
            if self._plus:
                np.maximum(R, 0.0, out=R)

    def _iterate_flat(self):
        R0, R1 = self._REACH0, self._REACH1
        C0, C1 = self._CFV0, self._CFV1
        R0[self._root_gid] = self.range0
        R1[self._root_gid] = self.range1

        sig = []
        for g in self._groups:                             # down: ascending depth
            ids, child, p, A = g["ids"], g["child"], g["player"], g["A"]
            pos = np.maximum(g["R"], 0.0)
            tot = pos.sum(axis=2, keepdims=True)
            sigma = np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), 1.0 / A)
            sig.append(sigma)
            RP, RO = (R0, R1) if p == 0 else (R1, R0)
            reach_p, reach_o = RP[ids], RO[ids]            # (G, N)
            if self._dcfr:
                g["S"] *= self._cg
                g["S"] += reach_p[:, :, None] * sigma
            else:
                g["S"] += self._weight * reach_p[:, :, None] * sigma
            child_reach_p = reach_p[:, :, None] * sigma    # (G, N, A)
            for a in range(A):
                ca = child[:, a]
                RP[ca] = child_reach_p[:, :, a]
                RO[ca] = reach_o

        tg = self._term_gids                               # terminals -> batched matmuls
        self._R0 = R0[tg].T.astype(np.float32)
        self._R1 = R1[tg].T.astype(np.float32)
        self._terminal_cfvs()
        C0[tg] = self._cfv0.T
        C1[tg] = self._cfv1.T

        for gi in range(len(self._groups) - 1, -1, -1):    # up: descending depth
            g = self._groups[gi]
            ids, child, p = g["ids"], g["child"], g["player"]
            sigma = sig[gi]
            c0 = C0[child].transpose(0, 2, 1)              # (G, N, A)
            c1 = C1[child].transpose(0, 2, 1)
            if p == 0:
                nv0 = (sigma * c0).sum(axis=2)
                self._apply_regret(g["R"], c0 - nv0[:, :, None])
                C0[ids], C1[ids] = nv0, c1.sum(axis=2)
            else:
                nv1 = (sigma * c1).sum(axis=2)
                self._apply_regret(g["R"], c1 - nv1[:, :, None])
                C1[ids], C0[ids] = nv1, c0.sum(axis=2)

    def train(self, iters, range0=None, range1=None, plus=True,
              variant="cfr+", dcfr_params=(1.5, 0.0, 2.0), engine="flat"):
        """Run `iters` CFR iterations. variant="cfr+" (default, verified) uses
        regret-matching+ with linear averaging; variant="dcfr" uses Discounted
        CFR (Brown & Sandholm 2019) with (alpha, beta, gamma) = dcfr_params:
        positive regrets *= t^a/(t^a+1), negative *= t^b/(t^b+1), strategy sum
        *= (t/(t+1))^g each iteration. Defaults (1.5, 0, 2) are the paper's.

        engine="flat" (default) is the level-batched iteration; engine="recursive"
        is the per-node reference kept as the differential oracle -- both share
        storage and produce equal results."""
        self.range0 = np.ones(self.N) if range0 is None else np.asarray(range0, float)
        self.range1 = np.ones(self.N) if range1 is None else np.asarray(range1, float)
        self._plus = plus
        self._dcfr = (variant == "dcfr")
        alpha, beta, gamma = dcfr_params
        for _ in range(iters):
            self._t += 1                   # persists across calls: chunked training stays correct
            t = self._t
            if self._dcfr:
                self._cp = (t ** alpha) / (t ** alpha + 1.0)   # positive-regret discount
                self._cn = (t ** beta) / (t ** beta + 1.0)     # negative-regret discount
                self._cg = (t / (t + 1.0)) ** gamma            # strategy-sum discount
            else:
                self._weight = float(t) if plus else 1.0
            if engine == "flat":
                self._iterate_flat()
            else:
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

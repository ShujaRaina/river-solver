"""Betting tree / action abstraction (Phase 2).

Builds the finite river betting tree for a heads-up spot defined by a starting
pot, an effective stack, and a set of bet/raise sizes. No raise cap: the stack
is the natural cap, so raises escalate until someone is all-in and the tree ends
itself (each raise strictly increases a player's contribution, bounded by the
stack). The betting node IS the public information set the Phase-3 CFR will attach
a per-combo strategy to.

Conventions
-----------
- P0 acts first (out of position). Amounts are chips this street; the starting
  pot is prior-street "dead money" already in the middle.
- A *bet* is `fraction * current_pot`. A *raise* is `fraction * pot_after_call`
  (pot-relative, the standard). Anything >= the remaining stack becomes all-in.
- A call always closes the river -> showdown. A fold ends the hand. Two checks
  in a row -> showdown.

Terminal nodes carry everything a value needs later: pot, each player's
contribution, and (for a fold) who folded. At a showdown the two contributions
are always equal (a call equalized them), so `stake` = each player's at-risk
chips. The dead-money-correct showdown value for Phase 3 is

    value_to_hero(i) = 0.5*pot*(M @ opp_reach)[i] + (0.5*pot - stake)*(C @ opp_reach)[i]

(derived from "hero nets pot on a win, pot/2 on a chop, 0 on a loss, minus its
own stake"); noted here so Phase 3 can just use it.
"""


class Node:
    __slots__ = ("player", "contrib", "actions", "kind", "folder", "pot",
                 "stake", "river_root")

    def __init__(self):
        self.player = None       # player to act at a decision node, else None
        self.contrib = None      # (c0, c1) chips in this street
        self.actions = []        # decision node: list of (label, child Node)
        self.kind = None         # "fold" / "showdown" terminal, or "chance"
        self.folder = None       # terminal fold: which player folded
        self.pot = None          # total pot (dead money + both contribs)
        self.stake = None        # terminal showdown: each player's at-risk chips
        self.river_root = None   # chance node: the river betting subtree

    def is_terminal(self):
        return self.kind in ("fold", "showdown")

    def is_chance(self):
        return self.kind == "chance"

    def child(self, label):
        for lab, nxt in self.actions:
            if lab == label:
                return nxt
        raise KeyError(f"no action {label!r} at this node (have {self.labels()})")

    def labels(self):
        return [lab for lab, _ in self.actions]


class TreeTooLarge(ValueError):
    """Betting tree exceeded the node budget (deep SPR / too many sizes).

    Raised DURING construction -- deep-stack trees grow to millions of nodes
    and would OOM before a post-hoc count could catch them, so we abort as
    soon as the running node count crosses the cap.
    """


def build_tree(base_pot=20.0, stack=80.0, fractions=(0.33, 0.66),
               first_actor=0, round_to=4, on_close=None, max_nodes=None):
    """Build one betting round. When betting closes (a call, or check-check)
    without a fold, `on_close(contrib)` decides the continuation -- a showdown
    terminal by default, or (for the turn) a chance node into a river round.

    `max_nodes` (if set) aborts construction with TreeTooLarge once that many
    nodes have been created -- a DoS guard for deep-stack / many-size spots."""
    eps = 1e-9
    R = lambda x: round(x, round_to)
    n_built = [0]

    def _count():
        n_built[0] += 1
        if max_nodes is not None and n_built[0] > max_nodes:
            raise TreeTooLarge(
                f"betting tree exceeds {max_nodes} nodes "
                f"(stack/pot too deep or too many bet sizes)")

    def showdown(contrib):
        # reached only after equalized betting, so contrib[0] == contrib[1].
        # Derive pot from the rounded contribs so pot == dead money + contribs exactly.
        _count()
        rc0, rc1 = R(contrib[0]), R(contrib[1])
        n = Node()
        n.kind = "showdown"
        n.contrib = (rc0, rc1)
        n.pot = R(base_pot + rc0 + rc1)
        n.stake = rc0
        return n

    def fold(contrib, folder):
        _count()
        rc0, rc1 = R(contrib[0]), R(contrib[1])
        n = Node()
        n.kind = "fold"
        n.folder = folder
        n.contrib = (rc0, rc1)
        n.pot = R(base_pot + rc0 + rc1)
        return n

    close = on_close or showdown          # what a closed round leads to

    def build(contrib, to_act, prev_checked):
        _count()
        other = 1 - to_act
        to_call = contrib[other] - contrib[to_act]
        remaining = stack - contrib[to_act]
        node = Node()
        node.player = to_act
        node.contrib = (R(contrib[0]), R(contrib[1]))

        if to_call <= eps:
            # No bet to face: check, or bet.
            if prev_checked:
                node.actions.append(("check", close(contrib)))
            else:
                node.actions.append(("check", build(contrib, other, True)))
            pot_now = base_pot + contrib[0] + contrib[1]
            sized = []
            for f in fractions:
                amt = f * pot_now
                if amt < remaining - eps:          # a real bet, not all-in
                    sized.append((f"bet{int(round(f * 100))}", amt))
            if remaining > eps:
                sized.append(("allin", remaining))
            _add_sized(node, contrib, to_act, other, sized, R, eps)
        else:
            # Facing a bet: fold, call (closes -> showdown), or raise.
            node.actions.append(("fold", fold(contrib, to_act)))
            called = list(contrib)
            called[to_act] = contrib[other]           # a call equalizes contribs
            node.actions.append(("call", close(tuple(called))))
            pot_after_call = base_pot + 2 * contrib[other]
            sized = []
            for f in fractions:
                new_c = contrib[other] + f * pot_after_call
                if new_c < stack - eps and new_c > contrib[other] + eps:
                    sized.append((f"raise{int(round(f * 100))}", new_c - contrib[to_act]))
            if stack - contrib[other] > eps:          # all-in shove (always legal)
                sized.append(("allin", stack - contrib[to_act]))
            _add_sized(node, contrib, to_act, other, sized, R, eps)
        return node

    def _add_sized(node, contrib, to_act, other, sized, R, eps):
        """Attach bet/raise children, deduped by resulting contribution."""
        seen = set()
        for label, added in sized:
            if added <= eps:
                continue
            new_contrib = list(contrib)
            new_contrib[to_act] += added
            key = R(new_contrib[to_act])
            if key in seen:
                continue
            seen.add(key)
            node.actions.append((label, build(tuple(new_contrib), other, False)))

    return build((0.0, 0.0), first_actor, False)


def walk(node):
    """Yield every node in the tree (pre-order), descending through chance
    nodes into their river subtrees."""
    yield node
    if node.is_chance():
        yield from walk(node.river_root)
    for _, child in node.actions:
        yield from walk(child)


def counts(root):
    c = dict(decisions=0, chance=0, terminals=0, folds=0, showdowns=0)
    for n in walk(root):
        if n.is_terminal():
            c["terminals"] += 1
            c["folds"] += (n.kind == "fold")
            c["showdowns"] += (n.kind == "showdown")
        elif n.is_chance():
            c["chance"] += 1
        else:
            c["decisions"] += 1
    return c

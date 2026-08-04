"""Checks for the turn betting tree (Phase T1). Run: python test_turn_tree.py"""

from betting import walk, counts
from turn_tree import build_turn_tree

BASE, STACK = 20.0, 80.0
root = build_turn_tree(base_pot=BASE, stack=STACK, fractions=(1.0,), first_actor=0)

c = counts(root)
print("node counts:", c)
assert c["chance"] > 0, "the turn must contain river chance nodes"
assert c["decisions"] > 0 and c["showdowns"] > 0

# root is the turn's first decision, facing no bet
assert root.player == 0 and set(root.labels()) == {"check", "bet100", "allin"}

# --- check-check on the turn -> chance node, river pot = dead money only ---
cc = root.child("check").child("check")
assert cc.is_chance(), "turn check-check leads to the river chance node"
assert abs(cc.pot - BASE) < 1e-6, "no turn betting -> river pot is just the dead money"
rr = cc.river_root
assert rr.player == 0 and not rr.is_terminal() and not rr.is_chance()

# --- turn bet(pot) + call -> chance node with the pot carried forward ---
bc = root.child("bet100").child("call")
assert bc.is_chance()
assert abs(bc.pot - (BASE + 2 * BASE)) < 1e-6, "pot 20 + bets 20 + 20 = 60"

# --- structural invariants over the whole tree ---
for n in walk(root):
    if n.is_chance():
        # a river subtree is a plain betting round: no nested chance nodes
        assert counts(n.river_root)["chance"] == 0, "river round has no chance"
        # its stacks are bounded: no showdown invests more than the full stack
        for m in walk(n.river_root):
            if m.kind == "showdown":
                # total invested never exceeds the stack (turn part + river part)
                assert m.contrib[0] <= STACK - n.contrib[0] + 1e-6, "river stack carried down"

print("all turn-tree checks passed")

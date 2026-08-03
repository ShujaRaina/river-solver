"""Checks for the betting tree (Phase 2). Run: python test_betting.py"""

from betting import build_tree, walk, counts

BASE, STACK = 20.0, 80.0
root = build_tree(base_pot=BASE, stack=STACK, fractions=(0.33, 0.66), first_actor=0)

# --- finiteness + shape ---
c = counts(root)
assert c["decisions"] > 0 and c["terminals"] > 0
assert c["terminals"] == c["folds"] + c["showdowns"]
print("node counts:", c)

# --- root is P0, facing no bet: check + three bet sizes ---
assert root.player == 0
assert set(root.labels()) == {"check", "bet33", "bet66", "allin"}

# --- every terminal is well-formed and pot-accounted ---
for n in walk(root):
    if not n.is_terminal():
        # a decision node's contribs never exceed the stack
        assert 0 <= n.contrib[0] <= STACK + 1e-6 and 0 <= n.contrib[1] <= STACK + 1e-6
        continue
    assert abs(n.pot - (BASE + n.contrib[0] + n.contrib[1])) < 1e-6, "pot = dead money + contribs"
    if n.kind == "showdown":
        assert abs(n.contrib[0] - n.contrib[1]) < 1e-6, "showdown contribs equalized"
        assert abs(n.stake - n.contrib[0]) < 1e-6
    else:
        assert n.folder in (0, 1)

# --- specific lines ---
# check-check -> showdown for the dead money only
cc = root.child("check").child("check")
assert cc.kind == "showdown" and abs(cc.pot - BASE) < 1e-6 and abs(cc.stake) < 1e-6

# P0 bets 66%, P1 folds -> P1 is the folder, P0 wins pot = 20 + 13.2
bf = root.child("bet66").child("fold")
assert bf.kind == "fold" and bf.folder == 1
assert abs(bf.pot - (BASE + 0.66 * BASE)) < 1e-6

# facing an all-in, the only options are fold or call (no raise beyond the stack)
facing_shove = root.child("allin")
assert facing_shove.player == 1
assert set(facing_shove.labels()) == {"fold", "call"}
# and calling the shove is a showdown at full stacks
allin_sd = facing_shove.child("call")
assert allin_sd.kind == "showdown" and abs(allin_sd.stake - STACK) < 1e-6

# --- max contribution never exceeds the stack (no-cap is stack-bounded) ---
assert max(max(n.contrib) for n in walk(root)) <= STACK + 1e-6

print("all betting checks passed")

"""Head-to-head: iters + wall-time to reach 0.02 bb, CFR+ vs Discounted CFR
(a couple param sets), same spots. Answers whether DCFR converges faster on
these river endgames or whether CFR+ is already near-optimal here."""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
from cards import parse_card
from solver import Solver
from ranges import range_from_classes

board = [parse_card(c) for c in "As Kd 7s 2c 9h".split()]
r0c = {c: 1 for c in "AA KK QQ AKs AQs AJs KQs KJs QJs JTs T9s 99 88 77 A5s A4s".split()}
r1c = {c: 1 for c in "AA KK 99 77 AKs AKo AQo AJo KQo QJs JTs T9s T8s 54s".split()}
FRACS = (0.33, 0.66, 1.0, 1.5, 2.0)

VARIANTS = [
    ("CFR+          ", "cfr+", None),
    ("DCFR(1.5,0,2) ", "dcfr", (1.5, 0.0, 2.0)),
    ("DCFR(3,0,2)   ", "dcfr", (3.0, 0.0, 2.0)),
    ("LCFR(1,1,1)   ", "dcfr", (1.0, 1.0, 1.0)),
]
SPRS = [(20, 40, "2"), (20, 60, "3"), (20, 80, "4")]
TARGET, CHUNK, MAX_ITERS, WALL = 0.02, 100, 6000, 200.0


def iters_to_target(variant, params, pot, stack):
    s = Solver(board, float(pot), float(stack), FRACS, max_nodes=5000)
    r0 = range_from_classes(r0c, s.combos)
    r1 = range_from_classes(r1c, s.combos)
    kw = {"variant": variant}
    if params:
        kw["dcfr_params"] = params
    t0 = time.monotonic()
    while s._t < MAX_ITERS and (time.monotonic() - t0) < WALL:
        s.train(CHUNK, range0=r0, range1=r1, **kw)
        if float(s.exploitability()) < TARGET:
            return s._t, time.monotonic() - t0, float(s.exploitability())
    return None, time.monotonic() - t0, float(s.exploitability())


print(f"{'variant':14} " + "  ".join(f"SPR{spr:>2}" .rjust(16) for *_, spr in SPRS), flush=True)
print("-" * 66, flush=True)
for name, variant, params in VARIANTS:
    cells = []
    for pot, stack, spr in SPRS:
        it, sec, expl = iters_to_target(variant, params, pot, stack)
        cells.append(f"{it}it {sec:.0f}s" if it else f">{MAX_ITERS} ({expl:.3f})")
    print(f"{name} " + "  ".join(c.rjust(16) for c in cells), flush=True)
print("done", flush=True)

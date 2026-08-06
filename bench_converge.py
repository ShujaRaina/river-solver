"""Iters-to-converge benchmark: how many CFR+ iterations to drive exploitability
below 0.05 / 0.02 / 0.01 bb, across board textures and SPRs. Feeds the rate
limit / solve-backstop / max-iters decision. Streams one row per spot."""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
from cards import parse_card
from solver import Solver
from ranges import range_from_classes

BOARDS = {
    "dry   ": "As Kd 7s 2c 9h",
    "wet   ": "9h 8h 7s 6c 2d",
    "paired": "Kh Kd 7s 4c 2h",
}
# realistic river SPR range (most money is already in by the river)
SPRS = [(20, 40, "2"), (20, 60, "3"), (20, 80, "4"), (20, 120, "6")]
FRACS = (0.33, 0.66, 1.0, 1.5, 2.0)
THRESH = [0.05, 0.02]                    # 0.02 bb (0.1% pot) is the target
CHUNK, MAX_ITERS, WALL_CAP = 100, 5000, 180.0

r0c = {c: 1 for c in "AA KK QQ AKs AQs AJs KQs KJs QJs JTs T9s 99 88 77 A5s A4s".split()}
r1c = {c: 1 for c in "AA KK 99 77 AKs AKo AQo AJo KQo QJs JTs T9s T8s 54s".split()}

print(f"{'board':6}  {'SPR':>3}  {'nodes':>5}  "
      f"{'<0.05':>13}  {'<0.02 (target)':>15}  {'final expl':>10}", flush=True)
print("-" * 62, flush=True)

for name, bstr in BOARDS.items():
    board = [parse_card(c) for c in bstr.split()]
    for pot, stack, spr in SPRS:
        s = Solver(board, float(pot), float(stack), FRACS, max_nodes=5000)
        from betting import counts
        nodes = sum(counts(s.root).values())
        r0 = range_from_classes(r0c, s.combos)
        r1 = range_from_classes(r1c, s.combos)
        hit = {t: None for t in THRESH}          # iters at first crossing
        t_hit = {t: None for t in THRESH}        # wall-seconds at first crossing
        t0 = time.monotonic()
        expl = None
        while s._t < MAX_ITERS and (time.monotonic() - t0) < WALL_CAP:
            s.train(CHUNK, range0=r0, range1=r1, plus=True)
            expl = float(s.exploitability())
            el = time.monotonic() - t0
            for t in THRESH:
                if hit[t] is None and expl < t:
                    hit[t], t_hit[t] = s._t, el
            if all(hit[t] is not None for t in THRESH):
                break

        def cell(t):
            return f"{hit[t]:>4}it {t_hit[t]:6.1f}s" if hit[t] else f"{'>'+str(s._t)+'it':>12}"
        print(f"{name}  {spr:>3}  {nodes:>5}  "
              f"{cell(0.05):>13}  {cell(0.02):>15}  {expl:>10.4f}", flush=True)

print("done", flush=True)

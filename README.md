# River subgame solver

A heads-up no-limit **river** solver: given a fixed 5-card board, two player
ranges, a pot/stack, and a bet-size tree, solve the betting subgame to
equilibrium with vectorized CFR and report exploitability. The river (no future
cards, no chance nodes after the start) is the smallest game that is genuinely
"an NL solver," and the proving ground for the range-based, vectorized CFR that
larger postflop solves need.

Descends from a tabular Kuhn/Leduc CFR solver. The betting logic generalizes
Leduc's; the verification (exploitability → 0) is the same method; the new idea
is carrying a **vector over ~1,000 hand combos** and doing CFR as array math
instead of walking one deal at a time.

## Phases

- **Phase 0 — cards + fixed-board evaluator** ✅
  `cards.py` (int 0..51 representation, parsing, combos), `evaluator.py` (5- and
  7-card hand scoring), `board.py` (dense per-combo showdown strength on a board).
  Hand-checked in `test_cards.py`, `test_evaluator.py`, `test_board.py`.
- **Phase 1 — ranges + terminal EV** ✅
  `ranges.py`, `terminal.py`. Card-removal/blockers, fold + showdown EV over
  ranges via a precomputed N×N showdown matrix (blocked pairs zeroed), each
  terminal EV a single matrix-vector product. Differentially verified against a
  pure-Python O(N²) brute force. (The O(N log N) prefix-sum showdown is deferred
  to when turn/flop runouts make an N² matrix per board too big to store.)
- **Phase 2 — betting tree / action abstraction** ✅
  `betting.py`. Finite tree from (starting pot, effective stack, bet-size set) —
  **no raise cap**; the stack bounds it (raises escalate to all-in). Default spot:
  pot 20, stack 80 (SPR 4), sizes {33%, 66%, all-in}, P0 out of position.
  Pot-relative sizing, terminal classification, and pot accounting; verified in
  `test_betting.py` (60 decision nodes, 117 terminals).
- **Phase 3 — vectorized CFR engine** ✅
  `solver.py`. Per-combo regret/strategy arrays over the betting tree; reach
  vectors propagated down, vector-CFR updates, dead-money-correct terminal
  values; CFR+ (regret floored at 0, linear averaging). Verified by
  exploitability → 0 — CFR+ reaches **~0.3% of pot at 2000 iters** on a sample
  spot — and the constant-sum invariant `u0 + u1 = base_pot · Z` holds exactly.
  `test_solver.py`.
- **Phase 4 — verification + readout** ✅
  `verify.py` — an independent Monte-Carlo playout of the average strategy
  reproduces the solver's computed game value (a separate code path must agree,
  the Leduc-vs-OpenSpiel discipline). `test_verify.py` — the exploitability
  metric discriminates (uniform ~34 vs solved <1 chips/hand). `readout.py` /
  `solve_river.py` — range-weighted action mix and per-hand strategy. The solved
  strategy is textbook-**polarized** (value-bet the nuts, bluff the worst hands,
  check medium) — discovered from scratch.
- **Phase 5 — usability & scale** (next)
  Real input ranges, richer bet trees, faster CFR (the O(N log N) showdown for
  turn/flop), and a cleaner solve API.

## Run the tests

```bash
python test_cards.py && python test_evaluator.py && python test_board.py
python test_terminal.py            # Phase 1 (needs numpy)
python test_betting.py             # Phase 2
python test_solver.py              # Phase 3 (~11s: trains a real spot)
python test_verify.py              # Phase 4 (metric + Monte-Carlo cross-check)

python solve_river.py              # solve a spot and print the strategy
```

Phase 0 is pure standard library; numpy enters at Phase 1 for the vectorized
terminal EV (`pip install -r requirements.txt`).

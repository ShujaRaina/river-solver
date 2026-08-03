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
- **Phase 2 — betting tree / action abstraction**
  Finite tree from (pot, stack, bet sizes, raise cap): nodes, legal actions,
  pot accounting.
- **Phase 3 — vectorized CFR engine**
  Per-combo regret/strategy arrays; vanilla CFR first, then CFR+/DCFR.
- **Phase 4 — verification**
  Exploitability → 0 (% of pot); cross-check small spots against brute force.
- **Phase 5 — usability**
  Real input ranges, richer bet trees, strategy/EV readouts, numpy perf pass.

## Run the tests

```bash
python test_cards.py && python test_evaluator.py && python test_board.py
python test_terminal.py            # Phase 1 (needs numpy)
```

Phase 0 is pure standard library; numpy enters at Phase 1 for the vectorized
terminal EV (`pip install -r requirements.txt`).

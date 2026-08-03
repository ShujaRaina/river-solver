"""Ranges over a board's hole combos (Phase 1, + range notation in Phase 5).

A range is a nonnegative weight vector aligned with the combo list returned by
board.board_strengths(board): weights[i] is how often the player holds combos[i].
Weights need not sum to 1 -- the solver only cares about relative reach.

`parse_range` turns standard poker notation into that vector, so you can feed the
solver a *real* range ("AA-TT, AKs, A5s:0.5") instead of uniform. Card removal is
automatic: combos using a board card just aren't in `combos`, so they get 0.
Supported: pairs (AA, 77), suited/offsuit (AKs, AKo), both (AK), plus (77+, ATs+),
dashes (AA-TT, AKs-ATs), and per-combo weights (A5s:0.5). Realism is the caller's
job -- these are plausible inputs, not equilibria derived from earlier streets.
"""

import numpy as np

from cards import RANK_CHARS, make_card


def uniform_range(n):
    return np.ones(n)


def random_range(n, seed):
    return np.random.default_rng(seed).random(n)


def range_from_weights(combos, weights_by_combo, default=0.0):
    """Build an aligned weight vector from a {(card_a, card_b): weight} dict."""
    index = {c: i for i, c in enumerate(combos)}
    r = np.full(len(combos), float(default))
    for combo, w in weights_by_combo.items():
        a, b = combo
        key = (a, b) if a < b else (b, a)
        r[index[key]] = w
    return r


# --- poker range notation --------------------------------------------------

def _parse_class(s):
    """'AKs' -> (rank_A, rank_K, 's'); 'AK' -> (.., .., None); 'AA' -> (A, A, None).
    Ranks are returned high-first."""
    s = s.strip()
    suited = None
    if len(s) == 3 and s[2] in "so":
        suited = s[2]
        s = s[:2]
    if len(s) != 2:
        raise ValueError(f"bad hand class {s!r}")
    r1, r2 = RANK_CHARS.find(s[0].upper()), RANK_CHARS.find(s[1].upper())
    if r1 < 0 or r2 < 0:
        raise ValueError(f"bad hand class {s!r}")
    return (max(r1, r2), min(r1, r2), suited)


def _class_combos(r1, r2, suited):
    """All 2-card combos for one hand class (over the full 52-card deck)."""
    if r1 == r2:                                   # a pocket pair: C(4,2) = 6
        cards = [make_card(r1, s) for s in range(4)]
        return [(a, b) for i, a in enumerate(cards) for b in cards[i + 1:]]
    out = []
    if suited in ("s", None):                      # 4 suited
        for s in range(4):
            a, b = make_card(r1, s), make_card(r2, s)
            out.append((min(a, b), max(a, b)))
    if suited in ("o", None):                      # 12 offsuit
        for s1 in range(4):
            for s2 in range(4):
                if s1 != s2:
                    a, b = make_card(r1, s1), make_card(r2, s2)
                    out.append((min(a, b), max(a, b)))
    return out


def _expand(expr):
    """Expand one token ('77+', 'AKs-ATs', 'AA', ...) into 52-card combos."""
    expr = expr.strip()
    if "-" in expr:
        left, right = (x.strip() for x in expr.split("-"))
        h1, l1, s1 = _parse_class(left)
        h2, l2, s2 = _parse_class(right)
        out = []
        if h1 == l1 and h2 == l2:                  # pair range, e.g. AA-TT
            lo, hi = sorted((h1, h2))
            for r in range(lo, hi + 1):
                out += _class_combos(r, r, None)
            return out
        if h1 != h2 or s1 != s2:                    # e.g. AKs-ATs: fixed high, vary low
            raise ValueError(f"bad range {expr!r}")
        lo, hi = sorted((l1, l2))
        for r in range(lo, hi + 1):
            out += _class_combos(h1, r, s1)
        return out
    if expr.endswith("+"):
        h, l, su = _parse_class(expr[:-1])
        out = []
        if h == l:                                  # pair+, e.g. 77+ -> 77..AA
            for r in range(h, len(RANK_CHARS)):
                out += _class_combos(r, r, None)
            return out
        for r in range(l, h):                       # ATs+ -> AT, AJ, AQ, AK
            out += _class_combos(h, r, su)
        return out
    h, l, su = _parse_class(expr)
    return _class_combos(h, l, su)


def range_from_classes(class_weights, combos):
    """Build a weight vector from a {hand_class: weight} dict, e.g. from a 13x13
    range grid: {"AA": 1.0, "AKs": 0.5, "AKo": 1.0, ...}. Card removal automatic."""
    index = {c: i for i, c in enumerate(combos)}
    weights = np.zeros(len(combos))
    for label, w in class_weights.items():
        if w <= 0:
            continue
        for combo in _expand(label):
            if combo in index:
                weights[index[combo]] = float(w)
    return weights


def parse_range(spec, combos):
    """Turn a range string into a weight vector aligned with `combos`.
    Combos using a board card aren't in `combos`, so they're dropped (card
    removal). A later token overrides an earlier one for the same combo."""
    index = {c: i for i, c in enumerate(combos)}
    weights = np.zeros(len(combos))
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        w = 1.0
        if ":" in token:
            token, ws = token.split(":")
            w = float(ws)
        for combo in _expand(token):
            if combo in index:
                weights[index[combo]] = w
    return weights

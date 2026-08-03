"""Human-readable strategy readout (Phase 4).

Turns a solved spot into something you can actually read: the exploitability,
the range-weighted action mix at the root, and the strategy for specific hands.
This is the seed of "explain what the solver is doing" -- the interface a coach
product would build on.
"""

from cards import parse_cards, cards_str


def _range_weighted_mix(solver, node):
    avg = solver.average_strategy(node)          # (N, actions)
    w = solver.range0 if node.player == 0 else solver.range1
    w = w / w.sum()
    return w @ avg                               # (actions,)


def describe(solver, hands=()):
    pot = solver.base_pot
    print(f"exploitability : {100 * solver.exploitability() / pot:.2f}% of pot")
    root = solver.root
    print(f"P{root.player} root action mix (range-weighted):")
    for label, freq in zip(root.labels(), _range_weighted_mix(solver, root)):
        print(f"    {label:8s} {100 * freq:5.1f}%")

    if hands:
        index = {c: i for i, c in enumerate(solver.combos)}
        avg = solver.average_strategy(root)
        print(f"specific hands at the root (P{root.player}):")
        for hand in hands:
            a, b = parse_cards(hand)
            i = index[(a, b) if a < b else (b, a)]
            mix = ", ".join(f"{lab} {100 * p:.0f}%"
                            for lab, p in zip(root.labels(), avg[i]))
            print(f"    {cards_str(parse_cards(hand)):6s} -> {mix}")

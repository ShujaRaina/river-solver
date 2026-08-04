"""Two-round turn betting tree with the river chance node (Phase T1).

Turn betting -> (chance: deal the river) -> river betting -> showdown. Both rounds
reuse betting.build_tree; the turn round's `on_close` hook, instead of a showdown,
builds a chance node whose river subtree carries the pot and stacks forward:

    river pot   = dead money + both turn contributions
    river stack = starting stack - this street's contribution

The chance node holds a SINGLE river betting subtree (the betting structure is the
same whatever card falls); the per-river-card strengths and card removal are
applied by the solver in T2, which is also where the ~48x work/memory lands.
"""

from betting import build_tree, Node


def build_turn_tree(base_pot=20.0, stack=80.0, fractions=(0.33, 0.66),
                    first_actor=0, round_to=4, river_fractions=None):
    # The river round may use a leaner bet abstraction than the turn -- the turn
    # decision is what's being studied, and each turn line spawns a whole river
    # subtree, so trimming river sizes cuts the tree (and solve time) sharply.
    river_fractions = fractions if river_fractions is None else river_fractions

    def make_chance(contrib):
        # contribs are equal at a close; carry pot and remaining stacks forward.
        turn_in = contrib[0]
        node = Node()
        node.kind = "chance"
        node.contrib = (round(contrib[0], round_to), round(contrib[1], round_to))
        node.pot = round(base_pot + contrib[0] + contrib[1], round_to)   # pot into the river
        node.river_root = build_tree(node.pot, stack - turn_in, river_fractions,
                                     first_actor, round_to)
        return node

    return build_tree(base_pot, stack, fractions, first_actor, round_to,
                      on_close=make_chance)

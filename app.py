"""Web backend for the river solver.

A tiny Flask app: serves the single-page frontend and exposes POST /solve, which
turns a board + two 13x13 range grids into a CFR+ solve and returns the acting
player's strategy aggregated back onto the grid (so the frontend can colour it).

Run:  python app.py   ->  http://127.0.0.1:5000
"""

import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from cards import parse_card
from solver import Solver
from ranges import range_from_classes, _expand

RANKS = "AKQJT98765432"          # grid order, index 0 = A (highest)


def all_classes():
    """The 169 hand-class labels in 13x13 grid order (row-major)."""
    out = []
    for row in range(13):
        for col in range(13):
            if row == col:
                out.append(RANKS[row] * 2)
            elif row < col:
                out.append(RANKS[row] + RANKS[col] + "s")
            else:
                out.append(RANKS[col] + RANKS[row] + "o")
    return out


CLASSES = all_classes()


def solve_spot(board_strs, r0_classes, r1_classes, pot, stack, fractions, iters):
    board = [parse_card(c) for c in board_strs]
    solver = Solver(board, base_pot=float(pot), stack=float(stack),
                    fractions=tuple(fractions))
    r0 = range_from_classes(r0_classes, solver.combos)
    r1 = range_from_classes(r1_classes, solver.combos)
    if r0.sum() <= 0 or r1.sum() <= 0:
        raise ValueError("both players need a non-empty range")
    solver.train(int(iters), range0=r0, range1=r1, plus=True)

    root = solver.root
    avg = solver.average_strategy(root)                 # (N, actions)
    labels = root.labels()
    acting_range = r0 if root.player == 0 else r1
    index = {c: i for i, c in enumerate(solver.combos)}

    # Aggregate the per-combo strategy up to each hand class (range-weighted).
    grid = {}
    for cls in CLASSES:
        idxs = [index[c] for c in _expand(cls) if c in index]
        w = np.array([acting_range[i] for i in idxs])
        if w.sum() <= 0:
            grid[cls] = None                            # not in range / blocked
            continue
        dist = (w[:, None] * avg[idxs]).sum(axis=0) / w.sum()
        grid[cls] = [round(float(x), 4) for x in dist]

    weights = acting_range / acting_range.sum()
    expl = float(solver.exploitability())         # chips/hand == bb (pot/stack are in bb)
    return {
        "actions": labels,
        "root_player": root.player,
        "mix": [round(float(x), 4) for x in weights @ avg],
        "exploitability_bb": round(expl, 3),
        "strategy": grid,
    }


app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/solve", methods=["POST"])
def solve():
    data = request.get_json(force=True)
    try:
        board = data["board"]
        if len(board) != 5 or len(set(board)) != 5:
            raise ValueError("need exactly 5 distinct board cards")
        result = solve_spot(
            board, data["range0"], data["range1"],
            data.get("pot", 20), data.get("stack", 80),
            data.get("fractions", [0.33, 0.66, 1.0]), data.get("iters", 250),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)

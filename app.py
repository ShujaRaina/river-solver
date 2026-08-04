"""Web backend for the river solver.

A tiny Flask app: serves the single-page frontend and exposes POST /solve, which
turns a board + two 13x13 range grids into a CFR+ solve and returns the acting
player's strategy aggregated back onto the grid (so the frontend can colour it).

Run:  python app.py   ->  http://127.0.0.1:5000
"""

import threading
import uuid

import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from cards import parse_card
from solver import Solver
from turn_solver import TurnSolver
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


# --- turn solve: slow, so run in a background thread and poll for live progress -

_turn_jobs = {}


def _root_grid(acting_range, combos, avg):
    """Aggregate the root's per-combo strategy onto the 169 hand classes."""
    index = {c: i for i, c in enumerate(combos)}
    grid = {}
    for cls in CLASSES:
        idxs = [index[c] for c in _expand(cls) if c in index]
        w = np.array([acting_range[i] for i in idxs])
        if w.sum() <= 0:
            grid[cls] = None
            continue
        dist = (w[:, None] * avg[idxs]).sum(0) / w.sum()
        grid[cls] = [round(float(x), 4) for x in dist]
    return grid


def _turn_worker(job, board, r0c, r1c, pot, stack, fractions, target):
    try:
        s = TurnSolver(board, float(pot), float(stack), tuple(fractions))
        r0 = range_from_classes(r0c, s.turn_combos)
        r1 = range_from_classes(r1c, s.turn_combos)
        if r0.sum() <= 0 or r1.sum() <= 0:
            job["error"] = "both players need a non-empty range"
            job["done"] = True
            return
        job["actions"] = s.root.labels()
        w = r0 / r0.sum()
        while s._t < target and not job.get("cancel"):
            s.train(min(3, target - s._t), range0=r0, range1=r1)   # a small chunk
            avg = s._avg(s.root, False)
            job["iter"] = s._t
            job["mix"] = [round(float(x), 4) for x in (w @ avg)]
            job["strategy"] = _root_grid(r0, s.turn_combos, avg)
        job["exploitability_bb"] = round(float(s.exploitability()), 3)
        job["done"] = True
    except Exception as e:                                          # noqa: BLE001
        job["error"] = str(e)
        job["done"] = True


@app.route("/turn/solve", methods=["POST"])
def turn_solve():
    data = request.get_json(force=True)
    board = data.get("board", [])
    if len(board) != 4 or len(set(board)) != 4:
        return jsonify({"error": "need exactly 4 turn board cards"}), 400
    target = int(data.get("iters", 60))
    jid = uuid.uuid4().hex[:8]
    job = {"iter": 0, "done": False, "strategy": {}, "mix": [],
           "actions": [], "exploitability_bb": None, "target": target}
    _turn_jobs[jid] = job
    threading.Thread(target=_turn_worker, daemon=True, args=(
        job, [parse_card(c) for c in board], data["range0"], data["range1"],
        data.get("pot", 20), data.get("stack", 80),
        tuple(data.get("fractions", [0.33, 0.66, 1.0])), target)).start()
    return jsonify({"id": jid, "target": target})


@app.route("/turn/progress/<jid>")
def turn_progress(jid):
    job = _turn_jobs.get(jid)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(job)


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)

"""Web backend for the river solver.

A small Flask app: serves the single-page frontend and exposes an async river
solve (POST /river/solve + GET /river/progress/<id>) that trains a CFR+ solve
in a background thread so the frontend can show a live ticking counter.

Hardened for a public deploy (see config.py for the knobs): every solve input
is clamped/validated, simultaneous solves are capped by a semaphore, jobs are
rate-limited per IP and evicted when finished, and each solve has a wall-clock
backstop. Run behind gunicorn in production (single worker -- job state is
in-process); `python app.py` is dev only.

Run (dev):  python app.py   ->  http://127.0.0.1:8000
"""

import os
# Cap BLAS to one thread per solve (must be set before numpy imports). numpy's
# matrix threading is pure overhead here -- the matrices are small, so all-core
# threading spends ~1100% CPU to do the work of one core at the same wall speed.
# Uncapped, one solve grabs every core, so N concurrent solves oversubscribe and
# thrash; capped, each solve is ~one core and users spread across cores cleanly.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import threading
import time
import uuid
from collections import OrderedDict

import numpy as np
from flask import Flask, request, jsonify, send_from_directory

import config
from cards import parse_card
from solver import Solver
from betting import TreeTooLarge
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


# --- input validation -------------------------------------------------------
# One place that turns an untrusted JSON body into clamped, bounded solve
# params -- the #1 DoS surface, since iters/pot/stack drive solve cost.

def _num(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be a number")
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return v


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _validate_range(raw, name):
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a map of hand-class -> weight")
    if len(raw) > config.MAX_RANGE_ENTRIES:
        raise ValueError(f"{name} has too many entries "
                         f"(max {config.MAX_RANGE_ENTRIES})")
    clean = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            raise ValueError(f"{name} keys must be hand-class strings")
        w = _num(v, f"{name}[{k}]")
        if w < 0:
            raise ValueError(f"{name}[{k}] weight must be >= 0")
        if w > 0:
            clean[k] = _clamp(w, 0.0, 1.0)
    return clean


def _validate_fractions(raw):
    if raw is None:
        return [0.33, 0.66, 1.0]
    if not isinstance(raw, list):
        raise ValueError("fractions must be a list of numbers")
    out, seen = [], set()
    for f in raw:
        v = round(_num(f, "fraction"), 4)
        if v < config.MIN_FRACTION or v > config.MAX_FRACTION:
            continue                     # silently drop out-of-range sizes
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
        if len(out) >= config.MAX_FRACTIONS:
            break
    if not out:
        raise ValueError(
            f"need at least one bet fraction in "
            f"[{config.MIN_FRACTION}, {config.MAX_FRACTION}]")
    return out


def validate_solve(data):
    """Return clamped/validated solve params, or raise ValueError."""
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
    board = data.get("board", [])
    if not isinstance(board, list) or len(board) != 5 or len(set(board)) != 5:
        raise ValueError("need exactly 5 distinct board cards")
    board_ints = [parse_card(c) for c in board]        # raises on bad card

    r0 = _validate_range(data.get("range0", {}), "range0")
    r1 = _validate_range(data.get("range1", {}), "range1")
    if not r0 or not r1:
        raise ValueError("both players need a non-empty range")

    pot = _clamp(_num(data.get("pot", 20), "pot"), config.MIN_POT, config.MAX_POT)
    stack = _clamp(_num(data.get("stack", 80), "stack"),
                   config.MIN_STACK, config.MAX_STACK)
    fractions = _validate_fractions(data.get("fractions"))
    iters = int(_clamp(_num(data.get("iters", 250), "iters"), 1, config.MAX_ITERS))

    return {"board": board_ints, "range0": r0, "range1": r1, "pot": pot,
            "stack": stack, "fractions": tuple(fractions), "iters": iters}


app = Flask(__name__, static_folder="static", static_url_path="")

# Per-IP rate limiting (R-DEP-4). flask-limiter is a soft dependency: it's in
# requirements.txt for the deploy, but if it isn't installed locally we degrade
# to no limiting rather than failing to boot (dev convenience).
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    _limiter = Limiter(get_remote_address, app=app, default_limits=[])
except ImportError:                                     # pragma: no cover
    _limiter = None


def rate_limited(f):
    """Apply the configured per-IP limit if flask-limiter is available."""
    if _limiter is not None:
        return _limiter.limit(config.RATE_LIMIT)(f)
    return f


# Cap simultaneous solves so a burst can't oversubscribe the box (R-DEP-3).
_solve_slots = threading.Semaphore(config.MAX_CONCURRENT_SOLVES)


# --- live river solve: chunked in a thread and polled, for a ticking counter.
# A new solve cancels the previous one, so threads don't pile up; jobs are
# evicted once finished/old so the store can't grow unbounded (R-DEP-6).

_river_jobs = OrderedDict()
_river_last = {"job": None}
_jobs_lock = threading.Lock()


def _monotonic():
    return time.monotonic()


def _evict_jobs():
    """Drop finished/expired jobs, then LRU-trim to MAX_JOBS. Caller holds lock."""
    now = _monotonic()
    for jid in list(_river_jobs):
        job = _river_jobs[jid]
        if job.get("done") and now - job["_ts"] > config.JOB_TTL_S:
            del _river_jobs[jid]
    while len(_river_jobs) > config.MAX_JOBS:
        # evict oldest finished job if any, else the oldest job outright
        victim = next((j for j in _river_jobs if _river_jobs[j].get("done")),
                      next(iter(_river_jobs)))
        del _river_jobs[victim]


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


def _river_worker(job, board, r0c, r1c, pot, stack, fractions, target):
    # Wait for a concurrency slot, bailing early if we've already been cancelled
    # (a newer solve superseded us while we queued).
    acquired = False
    try:
        while not job["cancel"]:
            if _solve_slots.acquire(timeout=0.5):
                acquired = True
                break
        if not acquired:
            job["done"] = True
            return

        s = Solver(board, float(pot), float(stack), tuple(fractions),
                   max_nodes=config.MAX_TREE_NODES)
        r0 = range_from_classes(r0c, s.combos)
        r1 = range_from_classes(r1c, s.combos)
        if r0.sum() <= 0 or r1.sum() <= 0:
            job["error"] = "both players need a non-empty range"
            job["done"] = True
            return
        job["actions"] = s.root.labels()
        w = r0 / r0.sum()
        deadline = _monotonic() + config.SOLVE_TIMEOUT_S
        while s._t < target and not job["cancel"]:
            s.train(min(25, target - s._t), range0=r0, range1=r1, plus=True)  # a chunk
            avg = s.average_strategy(s.root)
            job["iter"] = s._t
            job["mix"] = [round(float(x), 4) for x in (w @ avg)]
            job["strategy"] = _root_grid(r0, s.combos, avg)
            job["exploitability_bb"] = round(float(s.exploitability()), 3)
            if _monotonic() > deadline:            # wall-clock backstop (R-DEP-9)
                job["timeout"] = True
                break
        job["done"] = True
    except TreeTooLarge as e:
        job["error"] = str(e)
        job["done"] = True
    except Exception as e:                                          # noqa: BLE001
        job["error"] = str(e)
        job["done"] = True
    finally:
        if acquired:
            _solve_slots.release()


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "jobs": len(_river_jobs)})


@app.route("/river/solve", methods=["POST"])
@rate_limited
def river_solve():
    try:
        params = validate_solve(request.get_json(force=True, silent=True) or {})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    with _jobs_lock:
        if _river_last["job"] is not None:        # cancel any in-flight solve
            _river_last["job"]["cancel"] = True
        jid = uuid.uuid4().hex[:8]
        job = {"iter": 0, "done": False, "strategy": {}, "mix": [], "actions": [],
               "exploitability_bb": None, "target": params["iters"],
               "cancel": False, "_ts": _monotonic()}
        _river_jobs[jid] = job
        _river_last["job"] = job
        _evict_jobs()

    threading.Thread(target=_river_worker, daemon=True, args=(
        job, params["board"], params["range0"], params["range1"],
        params["pot"], params["stack"], params["fractions"],
        params["iters"])).start()
    return jsonify({"id": jid, "target": params["iters"]})


@app.route("/river/progress/<jid>")
def river_progress(jid):
    job = _river_jobs.get(jid)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    # don't leak internal bookkeeping (timestamp, cancel flag) to the client
    hidden = {"cancel"}
    return jsonify({k: v for k, v in job.items()
                    if not k.startswith("_") and k not in hidden})


if __name__ == "__main__":
    # dev server only; production uses gunicorn (see Dockerfile). port 8000, not
    # 5000: macOS uses 5000 for AirPlay Receiver (returns 403).
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, threaded=True)

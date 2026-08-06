"""Deployment hardening knobs (M1).

Every limit that protects the public deploy lives here, env-overridable so the
host can tighten/loosen without a code change. Defaults are the PRD's locked
values. Import this instead of hard-coding bounds in app.py / the engine.
"""

import os


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --- solve-cost clamps (R-DEP-1, R-DEP-2) ---------------------------------
MAX_ITERS = _int("SOLVER_MAX_ITERS", 3000)        # server-side cap on iters
MAX_POT = _float("SOLVER_MAX_POT", 2000.0)        # bb
MAX_STACK = _float("SOLVER_MAX_STACK", 2000.0)    # bb
MIN_POT = _float("SOLVER_MIN_POT", 1.0)
MIN_STACK = _float("SOLVER_MIN_STACK", 1.0)
MAX_FRACTIONS = _int("SOLVER_MAX_FRACTIONS", 5)   # bet sizes (all-in auto-added)
MIN_FRACTION = _float("SOLVER_MIN_FRACTION", 0.05)
MAX_FRACTION = _float("SOLVER_MAX_FRACTION", 3.0)
MAX_RANGE_ENTRIES = _int("SOLVER_MAX_RANGE_ENTRIES", 200)   # per player
MAX_TREE_NODES = _int("SOLVER_MAX_TREE_NODES", 5000)        # enforced in build

# --- concurrency / rate / time (R-DEP-3, R-DEP-4, R-DEP-9) -----------------
# Default concurrency ~= usable cores; capped so a burst can't oversubscribe.
try:
    _cpu = len(os.sched_getaffinity(0))          # linux: cores we may use
except AttributeError:
    _cpu = os.cpu_count() or 2
MAX_CONCURRENT_SOLVES = _int("SOLVER_MAX_CONCURRENT", max(1, _cpu))
RATE_LIMIT = os.environ.get("SOLVER_RATE_LIMIT", "10 per minute")
SOLVE_TIMEOUT_S = _float("SOLVER_TIMEOUT_S", 120.0)  # wall-clock backstop

# --- job store (R-DEP-6) ---------------------------------------------------
MAX_JOBS = _int("SOLVER_MAX_JOBS", 64)               # LRU-evict beyond this
JOB_TTL_S = _float("SOLVER_JOB_TTL_S", 300.0)        # drop finished jobs after

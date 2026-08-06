# Deploying the river solver

The solver is a single-instance Flask service (job state lives in-process, so it
is **not** horizontally scalable — one instance only). It ships as a Docker
image run under gunicorn. Target host: **Render** (container platform; free tier
is disqualified — 0.1 vCPU cannot run a CPU-bound CFR solve).

## One-time setup

1. Push this repo to GitHub (already at `github.com/ShujaRaina/river-solver`).
2. In the Render dashboard: **New +  ->  Blueprint**, point it at this repo.
   Render reads `render.yaml`, builds the `Dockerfile`, and provisions the web
   service on the `starter` plan with a health check on `/health`.
3. First build takes a few minutes (pip install numpy). When it's live you get
   an HTTPS URL like `https://river-solver.onrender.com`.

At M2 the URL is **live but unlinked** — smoke-test it before linking from the
portfolio (that's M4).

## Tier / cost

- `starter` (0.5 vCPU / 512 MB, ~$7/mo) — the default in `render.yaml`. Solves
  run ~2x slower than the benchmark times so deep spots hit the 60s backstop
  sooner; a max-size tree (~200 MB peak) sits under the 512 MB ceiling. Fine for
  light traffic.
- `standard` (1 vCPU / 2 GB, ~$25/mo) — upgrade for the benchmarked solve times
  and more headroom. Switch by editing `plan:` in `render.yaml` or in the Render
  dashboard (no code change).

Paid Render tiers are **always-on** (no scale-to-zero), so there is no cold
start and no prewarm needed.

## Benchmarked solve cost (single core = 1 vCPU)

| Spot | Nodes | 2000 it | per-iter |
|---|---|---|---|
| SPR 4, 3 sizes (default preset) | 454 | 40 s | 20 ms |
| SPR 4, 5 sizes | 574 | 51 s | 25 ms |
| SPR 8, 5 sizes | 1574 | 138 s | 69 ms |
| SPR 13, 5 sizes (near node cap) | 3734 | 343 s | 171 ms |

The UI default is 250 iters (~5 s for a typical spot). Deep spots are bounded by
the 60s wall-clock backstop and the 5000-node cap; on `starter`, halve the vCPU
(double the times).

## Tunable limits (env vars, see `config.py`)

All are set in `render.yaml` or overridable in the Render dashboard without a
code change: `SOLVER_MAX_CONCURRENT`, `SOLVER_MAX_ITERS`, `SOLVER_TIMEOUT_S`,
`SOLVER_MAX_TREE_NODES`, `SOLVER_RATE_LIMIT`, `SOLVER_MAX_POT/STACK`,
`SOLVER_MAX_FRACTIONS`, `SOLVER_MAX_RANGE_ENTRIES`, `SOLVER_MAX_JOBS`,
`SOLVER_JOB_TTL_S`.

**`SOLVER_MAX_CONCURRENT` must be pinned** — Render containers can report more
cores than the vCPU you're allocated, so the cpu-count default would
oversubscribe. One BLAS-capped solve = one core, so use 1 per vCPU.

## Local smoke test of the production image

```
docker build -t river-solver .
docker run --rm -p 8000:8000 -e SOLVER_MAX_CONCURRENT=1 river-solver
curl localhost:8000/health          # -> {"status":"ok","jobs":0}
```

To reproduce the Q6 tier benchmark exactly, run the test suite under a CPU cap:

```
docker run --rm --cpus=1.0 river-solver python test_lp.py
docker run --rm --cpus=0.5 river-solver python test_lp.py
```

FROM python:3.12-slim

# BLAS single-threaded per solve (see app.py for the why). Set here too so it
# holds regardless of how the container is launched.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# ONE worker: job state (async solves) lives in-process, so a second worker
# wouldn't see a job started in the first. --threads lets that single worker
# serve progress polls while a solve runs in its background thread. --timeout
# comfortably exceeds the in-app solve backstop (SOLVER_TIMEOUT_S, default 60s).
CMD ["sh", "-c", "gunicorn --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:${PORT} app:app"]

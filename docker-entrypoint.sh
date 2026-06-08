#!/usr/bin/env sh
set -eu

: "${CHATFLEET_API_WORKERS:=1}"
: "${TOKENIZERS_PARALLELISM:=false}"
: "${OMP_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"

export TOKENIZERS_PARALLELISM
export OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS
export MKL_NUM_THREADS

case "$CHATFLEET_API_WORKERS" in
  ''|*[!0-9]*)
    echo "[chatfleet-api][warn] invalid CHATFLEET_API_WORKERS='$CHATFLEET_API_WORKERS'; using 1" >&2
    CHATFLEET_API_WORKERS=1
    ;;
esac

if [ "$CHATFLEET_API_WORKERS" -le 1 ]; then
  exec uvicorn main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips '*'
fi

echo "[chatfleet-api][warn] starting $CHATFLEET_API_WORKERS workers. Local embeddings are loaded per worker." >&2
exec uvicorn main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips '*' --workers "$CHATFLEET_API_WORKERS"

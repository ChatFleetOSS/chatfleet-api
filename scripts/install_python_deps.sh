#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PIP_BIN="${PIP_BIN:-pip}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REQ_FILE="${REQ_FILE:-$REPO_ROOT/requirements.txt}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-$REPO_ROOT/constraints-linux-cpu.txt}"

install_args=(install --no-cache-dir)

if [[ "$(uname -s)" == "Linux" ]]; then
  install_args+=(
    --extra-index-url
    https://download.pytorch.org/whl/cpu
    -c
    "$CONSTRAINTS_FILE"
  )
fi

"$PIP_BIN" "${install_args[@]}" -r "$REQ_FILE"

"$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import platform
import sys

installed = {dist.metadata["Name"].lower(): dist.version for dist in md.distributions()}
torch_version = installed.get("torch")
print(f"[deps] torch={torch_version or 'not-installed'}")

if platform.system() == "Linux":
    gpu_runtime_packages = sorted(name for name in installed if name.startswith("nvidia-"))
    if gpu_runtime_packages:
        print(
            "[deps][error] unexpected GPU runtime packages installed: "
            + ", ".join(gpu_runtime_packages),
            file=sys.stderr,
        )
        raise SystemExit(1)
PY


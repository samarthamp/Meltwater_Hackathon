#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Provision the `hangman` runtime for the Meltwater Hangman Hackathon.
#
# Idempotent: safe to re-run.  Installs Miniforge3 (aarch64) if no conda is
# present, creates the `hangman` env, installs pinned requirements from PyPI
# plus NVIDIA's cu130 wheel index, registers a Jupyter kernel, and prints a
# hardware verification block.
#
# Usage:  bash scripts/00_setup_env.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="hangman"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# --- 1. conda -------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
    log "Using existing conda at ${CONDA_BIN}"
elif [[ -x "${CONDA_ROOT}/bin/conda" ]]; then
    CONDA_BIN="${CONDA_ROOT}/bin/conda"
    log "Using existing Miniforge at ${CONDA_ROOT}"
else
    log "Installing Miniforge3 (aarch64) into ${CONDA_ROOT}"
    INSTALLER="$(mktemp -t miniforge-XXXXXX.sh)"
    curl -fsSL -o "${INSTALLER}" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
    bash "${INSTALLER}" -b -p "${CONDA_ROOT}"
    rm -f "${INSTALLER}"
    CONDA_BIN="${CONDA_ROOT}/bin/conda"
fi

CONDA_BASE="$("${CONDA_BIN}" info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# --- 2. environment -------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    log "conda env '${ENV_NAME}' already exists - reusing"
else
    log "Creating conda env '${ENV_NAME}' from environment.yml"
    conda env create -f "${REPO_ROOT}/environment.yml"
fi
conda activate "${ENV_NAME}"

# --- 3. python packages ---------------------------------------------------
log "Installing Python requirements (torch from ${TORCH_INDEX})"
python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements.txt" --extra-index-url "${TORCH_INDEX}"

# --- 4. project package + jupyter kernel ----------------------------------
log "Installing project package (editable) and Jupyter kernel"
python -m pip install -e "${REPO_ROOT}"
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (hangman)"

# --- 5. verification ------------------------------------------------------
log "Verification"
python - <<'PYCHECK'
import os, platform, torch

print(f"python           : {platform.python_version()}  ({platform.machine()})")
print(f"torch            : {torch.__version__}")
print(f"torch CUDA build : {torch.version.cuda}")
print(f"cuda available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"device           : {props.name}")
    print(f"capability       : sm_{props.major}{props.minor}")
    print(f"unified memory   : {props.total_memory / 1e9:.1f} GB")
    print(f"multiprocessors  : {props.multi_processor_count}")
    print(f"bf16 supported   : {torch.cuda.is_bf16_supported()}")
    print(f"compiled archs   : {torch.cuda.get_arch_list()}")
print(f"cpu cores        : {os.cpu_count()}")

import hangman
print(f"hangman package  : {hangman.__version__} @ {os.path.dirname(hangman.__file__)}")
PYCHECK

log "Done.  Activate with:  conda activate ${ENV_NAME}"

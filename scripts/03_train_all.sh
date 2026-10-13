#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Train the ensemble members one at a time on a single GPU.
#
# Runs sequentially: one job already saturates the device, so concurrent jobs
# only add contention. Logs go to artifacts/logs/train_<name>.log.
#
# Usage:  bash scripts/03_train_all.sh
# Env:    PY  Python interpreter (default: the `hangman` conda env)
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/miniforge3/envs/hangman/bin/python}"
LOGS="${REPO_ROOT}/artifacts/logs"
mkdir -p "${LOGS}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

run() {
    local name="$1"; shift
    echo "=== ${name} start $(date +%H:%M:%S) ==="
    "${PY}" "${REPO_ROOT}/scripts/02_train.py" --run "${name}" "$@" \
        > "${LOGS}/train_${name}.log" 2>&1
    echo "=== ${name} done  $(date +%H:%M:%S) (exit $?) ==="
}

# Primary member: Transformer.
run former_a --arch transformer --seed 0 --d-model 512 --layers 8 --batch-size 4096 \
    --pretrain-steps 7000 --dagger-steps 4000 --dagger-rounds 3

# Recurrent member: bidirectional GRU.
run bigru_c --arch bigru --seed 2 --d-model 640 --rnn-layers 4 --batch-size 8192 \
    --pretrain-steps 7000 --dagger-steps 4000 --dagger-rounds 3

# Convolutional member: local receptive field suits affixes and digraphs.
# Fewer steps, as it is the most expensive per step.
run cnn_d --arch dilated_cnn --seed 3 --d-model 448 --cnn-layers 8 --batch-size 4096 \
    --pretrain-steps 5000 --dagger-steps 3000 --dagger-rounds 2

echo "all members finished $(date +%H:%M:%S)"

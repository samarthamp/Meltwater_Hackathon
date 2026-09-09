#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Train the ensemble members on the single GB10, one at a time.
#
# Sequential, and deliberately so.  Measured on this machine:
#
#   solo          transformer 3.34 it/s  |  bi-GRU 6.25 it/s
#   two at once   transformer 1.28 it/s  |  bi-GRU 1.93 it/s
#   four at once  ~15k states/s aggregate, against 13.7k for one job alone
#
# A single job already saturates the device, so sharing it only adds
# contention: finishing the transformer and the bi-GRU takes ~146 min
# sequentially against ~247 min concurrently.  Four at once also pushed unified
# memory to 104 of 119 GB, because GPU memory here is the same physical pool as
# system RAM.
#
# Usage:  bash scripts/03_train_all.sh
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

# Primary member: the architecture the report is written around.
run former_a --arch transformer --seed 0 --d-model 512 --layers 8 --batch-size 4096 \
    --pretrain-steps 7000 --dagger-steps 4000 --dagger-rounds 3

# Recurrent member: cheap, and expected strongest of the three on short words.
run bigru_c --arch bigru --seed 2 --d-model 640 --rnn-layers 4 --batch-size 8192 \
    --pretrain-steps 7000 --dagger-steps 4000 --dagger-rounds 3

# Convolutional member: local bias, suits affixes and digraphs.  Trained
# shorter because it is the most expensive per step of the three.
run cnn_d --arch dilated_cnn --seed 3 --d-model 448 --cnn-layers 8 --batch-size 4096 \
    --pretrain-steps 5000 --dagger-steps 3000 --dagger-rounds 2

echo "all members finished $(date +%H:%M:%S)"

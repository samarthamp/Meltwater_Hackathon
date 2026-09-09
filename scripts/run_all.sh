#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The whole pipeline, in order, from a clean checkout.
#
#   bash scripts/run_all.sh              # everything, including the LLM stage
#   SKIP_LLM=1 bash scripts/run_all.sh   # train.txt only, no pretrained weights
#
# Every stage is resumable: re-running skips work whose artifacts already
# exist, so an interrupted run can be restarted with the same command.
#
# Wall clock on a DGX Spark (GB10): roughly 8 hours end to end, of which the
# three ensemble members are ~4 h and the Qwen3-4B stage ~2 h.  Stages run
# sequentially by design - see reports/MODEL_REPORT.md section 8 for the
# measurements showing that sharing this GPU between jobs is slower, not faster.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/miniforge3/envs/hangman/bin/python}"
LOGS="${REPO_ROOT}/artifacts/logs"
CKPT="${REPO_ROOT}/artifacts/checkpoints"
SKIP_LLM="${SKIP_LLM:-0}"
mkdir -p "${LOGS}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

stage() { printf '\n\033[1;36m=== %s  (%s) ===\033[0m\n' "$1" "$(date +%H:%M:%S)"; }

stage "1/8  dictionaries, teacher tables, baselines, warm-start pool"
"${PY}" "${REPO_ROOT}/scripts/01_build_data.py" --states 20000000 2>&1 | tee "${LOGS}/01_build_data.log"

stage "2/8  ensemble members"
bash "${REPO_ROOT}/scripts/03_train_all.sh" 2>&1 | tee "${LOGS}/03_train_all.log"

if [[ "${SKIP_LLM}" != "1" ]]; then
    stage "3/8  Qwen3-4B teacher"
    "${PY}" "${REPO_ROOT}/scripts/04_finetune_llm.py" \
        --keep-layers 18 --mntp-steps 800 --sft-steps 3200 --batch-size 192 \
        2>&1 | tee "${LOGS}/04_finetune_llm.log"

    stage "4/8  selective distillation"
    "${PY}" "${REPO_ROOT}/scripts/05_distill.py" \
        --adapters "${CKPT}/qwen_teacher_adapters.pt" --keep-layers 18 \
        --alpha 0.5 --max-states 1200000 2>&1 | tee "${LOGS}/05_distill.log"

    stage "5/8  fold the distilled targets into each student"
    for ck in "${CKPT}"/*_final.pt; do
        "${PY}" "${REPO_ROOT}/scripts/06_distill_finetune.py" --ckpt "${ck}" --steps 3000 \
            2>&1 | tee -a "${LOGS}/06_distill_finetune.log"
    done
else
    stage "3-5/8  LLM stages skipped (SKIP_LLM=1)"
fi

stage "6/8  policy-gradient refinement of the primary member"
"${PY}" "${REPO_ROOT}/scripts/07_rl_refine.py" --ckpt "${CKPT}/former_a_final.pt" --steps 600 \
    2>&1 | tee "${LOGS}/07_rl_refine.log"

stage "7/8  calibration, ensembling, ablation table"
BEST=()
for name in former_a bigru_c cnn_d former_b; do
    for suffix in _distilled _rl _final; do
        if [[ -f "${CKPT}/${name}${suffix}.pt" ]]; then BEST+=("${CKPT}/${name}${suffix}.pt"); break; fi
    done
done
printf 'ensemble members: %s\n' "${BEST[*]}"
"${PY}" "${REPO_ROOT}/scripts/08_ensemble_eval.py" --ckpts "${BEST[@]}" \
    2>&1 | tee "${LOGS}/08_ensemble_eval.log"

stage "8/8  simulate the test set and write submission.csv"
"${PY}" "${REPO_ROOT}/scripts/09_make_submission.py" --ckpts "${BEST[@]}" \
    2>&1 | tee "${LOGS}/09_make_submission.log"

stage "reports"
"${PY}" "${REPO_ROOT}/tools/make_results.py"
"${PY}" "${REPO_ROOT}/tools/make_figures.py"
"${PY}" "${REPO_ROOT}/tools/model_stats.py" > "${REPO_ROOT}/reports/_model_stats.md"

printf '\n\033[1;32mdone (%s).  submission.csv written.\033[0m\n' "$(date +%H:%M:%S)"

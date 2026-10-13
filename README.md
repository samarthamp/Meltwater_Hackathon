# Neural Hangman

A bidirectional character Transformer that plays Hangman, trained from scratch on the provided
`train.txt`. It is supervised by an exact dictionary-posterior oracle, corrected for covariate
shift with DAgger self-play, and shipped as a uniform two-member ensemble played greedily.

## The two measurements that shaped the design

| Measurement | Value | Consequence |
|---|---|---|
| `train.txt` ∩ `test.txt` | **0 words** | The dictionaries are disjoint, so the model has to generalise to unseen words. |
| On-policy states with **no** consistent training-dictionary word | **45.1%** | Candidate filtering alone has no signal on nearly half the positions a real game reaches. |

## Results

Measured on 17,530 held-out words, disjoint from both training and the 5,000 words used for
fitting hyperparameters:

| Method | Win rate |
|---|---:|
| Static letter frequency (what `sample_submission.csv` does) | 14.18% |
| Interpolated character n-gram, noisy-OR pooled | 60.09% |
| Exact Bayes posterior + n-gram backoff (**the teacher / label source**) | 60.97% |
| Single model, `former_v2` | 67.41% |
| Single model, `former_a` | 68.98% |
| **Submitted: uniform 2-member ensemble** | **69.61%** |

**On the 250,000-word test set the submitted ensemble scores 72.955%** (182,388 words solved;
competition score 72.946978, 10.20 guesses per word).

Full results and the ablation table: [`reports/MODEL_REPORT.md`](reports/MODEL_REPORT.md).

## Quick start

```bash
bash scripts/00_setup_env.sh     # Miniforge + conda env `hangman` + Jupyter kernel
conda activate hangman

bash scripts/run_all.sh          # the whole pipeline, ~8 h on a DGX Spark
#   SKIP_LLM=1 bash scripts/run_all.sh   -> train.txt only, no pretrained weights

python -m pytest tests/ -q       # rules, teacher equivalence, policy identities
```

Re-runs reuse the cached n-gram tables and record shards; models are retrained. To drive the
stages individually:

```bash
python scripts/01_build_data.py --states 20000000    # teacher, baselines, warm-start pool (~7 min)
bash   scripts/03_train_all.sh                       # ensemble members, trained sequentially
python scripts/04_finetune_llm.py --keep-layers 18   # Qwen3-4B teacher            (optional)
python scripts/05_distill.py --adapters artifacts/checkpoints/qwen_teacher_adapters.pt
python scripts/06_distill_finetune.py --ckpt artifacts/checkpoints/former_a_final.pt
python scripts/07_rl_refine.py --ckpt artifacts/checkpoints/former_a_final.pt      # (optional)
python scripts/08_ensemble_eval.py --ckpts artifacts/checkpoints/*_final.pt
python scripts/09_make_submission.py --ckpts artifacts/checkpoints/*_final.pt
```

## Notebooks

| Notebook | Purpose |
|---|---|
| [`notebooks/hangman_end_to_end.ipynb`](notebooks/hangman_end_to_end.ipynb) | the full pipeline — data, teacher, training, DAgger, the LLM stage, calibration, ablations. Runs on the `Python (hangman)` kernel and calls the same library code as the scripts. |
| [`notebooks/kaggle_submission.ipynb`](notebooks/kaggle_submission.ipynb) | the Kaggle inference notebook. The model and game engine are vendored inline so it needs no repository; it loads the trained weights, plays all 250,000 test words, writes `submission.csv` and independently re-validates it. |

End-to-end notebook flags:

| Flag | Effect |
|---|---|
| `TRAIN_FROM_SCRATCH` | full training vs. loading checkpoints |
| `RUN_LLM_TEACHER` | include the Qwen3-4B stage |
| `RUN_RL_REFINEMENT` | include policy-gradient refinement |
| `SMOKE` | run every cell at reduced scale for a quick end-to-end check |

Regenerate them with `python tools/make_notebook.py` and
`python tools/make_kaggle_notebook.py`. The submission notebook vendors `src/hangman/` verbatim,
so regenerating it after a code change keeps the two in step.

## Reports

| Document | Contents |
|---|---|
| [`reports/EXECUTIVE_SUMMARY.md`](reports/EXECUTIVE_SUMMARY.md) | two-page overview of the approach and results |
| [`reports/MODEL_REPORT.md`](reports/MODEL_REPORT.md) | architecture, layer-by-layer parameter and FLOP tables, design decisions with their ablation numbers, results, reproduction commands |
| [`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md) | literature survey and what was borrowed from where |
| [`reports/FINETUNING_REPORT.md`](reports/FINETUNING_REPORT.md) | the Qwen3-4B stage: bidirectionalisation, character tokenisation, head surgery, LoRA, layer truncation, throughput accounting |
| [`reports/COMPLIANCE.md`](reports/COMPLIANCE.md) | how the rules are implemented and how the data is kept separate, with where to check each point in the code |

Diagrams: [`reports/figures/architecture.svg`](reports/figures/architecture.svg),
[`reports/figures/pipeline.svg`](reports/figures/pipeline.svg) (regenerate with
`python tools/make_diagram.py`).

## Key design choices

1. **Both guessed-letter sets are first-class inputs.** Knowing `q`, `x`, `z` are *not* in the
   word is strong evidence about what is. The present and absent sets are injected into every
   block by FiLM modulation rather than left to be routed through attention.
2. **Training states come from real play.** Randomly masked words are an unreachable
   distribution — a live game carries at most five absent letters, and they are mostly frequent
   ones. DAgger self-play trains on the states the policy actually visits.
3. **Noisy-OR pooling.** Per-position distributions are pooled into letter-presence
   probabilities with `1 − Π(1 − pᵢ)` instead of a max, worth **+8.0 points** on the n-gram
   baseline alone.
4. **The decision rule is selected, not assumed.** Value-guided lookahead and exact
   information-set search are both implemented and measured against greedy on held-out words
   (see below).

## A measured negative result

The planned inference policy — value-guided lookahead plus exact information-set search — turned
out to be **13.9 points worse** than plain greedy on the final model:

* The **value head was nearly constant** (mean 0.5155, std 0.0085): it is trained on Monte-Carlo
  returns and the off-policy warm-start records carry none, so one-ply lookahead was effectively
  choosing at random among the top five. `ValueGuidedPolicy` now refuses to run when the head
  shows no spread.
* Every **exact-search** variant sat within noise of greedy once a `miss_penalty` term was
  removed; that term alone cost 1.0 point. Candidate sets drawn from a *disjoint* dictionary are
  weak evidence, and overruling a well-trained network on them loses games.

So a search policy has to beat greedy by more than two standard errors on held-out words before
it ships.

## Hardware notes (NVIDIA DGX Spark, GB10, `sm_121`)

* The device has ~95 TFLOPS of bf16 matmul but only 273 GB/s of bandwidth, so a small model is
  **bandwidth-bound**. Profiling showed elementwise ops at 30% of step time against 13% for
  matmuls. Length-bucketed batches plus `torch.compile` fusion gave a **7.6× speedup**
  (1,807 → 13,680 states/s) and cut peak memory from 21.7 GB to 13.7 GB.
* **Concurrency does not help here.** One training job already saturates the GPU: running two
  members together took 247 minutes against 146 minutes sequentially, so training runs
  sequentially.

## Layout

```
Data/            train.txt, test.txt, sample_submission.csv (unmodified)
src/hangman/     the library — every script and notebook cell is a thin call into this
scripts/         00..09, one per pipeline stage, plus run_all.sh
tools/           analysis, model selection, and notebook/diagram/figure generators
tests/           rules, teacher matmul equivalence, policy identities
notebooks/       end-to-end pipeline and Kaggle inference notebooks
reports/         the reports plus figures
kaggle_upload/   the two shipped checkpoints (fp16) packaged as a Kaggle dataset
artifacts/       checkpoints, datasets, logs, metrics (generated, not tracked)
```

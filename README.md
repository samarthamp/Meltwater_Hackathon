# Neural Hangman — Meltwater Brand Buzzword Hackathon

A bidirectional character Transformer that plays Hangman, trained from scratch on the provided
`train.txt`, supervised by an exact dictionary-posterior oracle, corrected for covariate shift
with DAgger self-play, ensembled across three architectures, and played under a risk-aware policy
that switches to exact information-set search in the endgame. A Qwen3-4B teacher —
bidirectionalised and LoRA-finetuned — is distilled in on the states where the dictionary oracle
is blind.

## The two measurements that determined the design

| Measurement | Value | Consequence |
|---|---|---|
| `train.txt` ∩ `test.txt` | **0 words** | The dictionaries are disjoint. Memorisation is worthless; only generalisation scores. |
| On-policy states with **no** consistent training-dictionary word | **45.1%** | Lookup and candidate-filtering approaches hit a hard wall on nearly half the positions a real game reaches. |

## Baselines, measured on a 22,530-word held-out dictionary disjoint from training

| Method | Win rate |
|---|---:|
| Static letter frequency (what `sample_submission.csv` does) | 14.18% |
| Interpolated character n-gram, noisy-OR pooled | 60.09% |
| Exact Bayes posterior + n-gram backoff (**our teacher / label source**) | 60.97% |
| Single model, `former_a` | 68.98% |
| **Submitted: uniform 2-member ensemble** | **69.61%** |

**On the public test set the submitted ensemble scores 72.955%** (182,388 of 250,000 words),
leaderboard score 72.946978.

Full results and the ablation table: [`reports/MODEL_REPORT.md`](reports/MODEL_REPORT.md).

## Quick start

```bash
bash scripts/00_setup_env.sh     # Miniforge + conda env `hangman` + Jupyter kernel
conda activate hangman

bash scripts/run_all.sh          # the whole pipeline, ~8 h on a DGX Spark
#   SKIP_LLM=1 bash scripts/run_all.sh   -> train.txt only, no pretrained weights

python -m pytest tests/ -q       # rules, teacher equivalence, policy identities
```

Every stage is resumable, so an interrupted run restarts with the same command. To drive the
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

Two, with different jobs.

| Notebook | Purpose |
|---|---|
| [`notebooks/hangman_end_to_end.ipynb`](notebooks/hangman_end_to_end.ipynb) | the full pipeline — data, teacher, training, DAgger, the LLM stage, calibration, ablations. The reproducibility artefact. |
| [`notebooks/kaggle_submission.ipynb`](notebooks/kaggle_submission.ipynb) | the one to run **on Kaggle**. The model and game engine are vendored inline so it needs no repository; it loads the trained weights, plays all 250,000 test words, writes `submission.csv` and independently re-validates it. |

The end-to-end notebook runs on the `Python (hangman)` kernel — the same code the scripts call,
not a reimplementation. Flags:

| Flag | Effect |
|---|---|
| `TRAIN_FROM_SCRATCH` | full training vs. loading checkpoints |
| `RUN_LLM_TEACHER` | include the Qwen3-4B stage |
| `RUN_RL_REFINEMENT` | include policy-gradient refinement |
| `SMOKE` | run every cell at reduced scale for a quick end-to-end check |

Regenerate them with `python tools/make_notebook.py` and
`python tools/make_kaggle_notebook.py`. The submission notebook vendors `src/hangman/` verbatim,
so regenerating it after a code change keeps the two in step automatically.

## Reports

| Document | Contents |
|---|---|
| [`reports/MODEL_REPORT.md`](reports/MODEL_REPORT.md) | architecture, layer-by-layer parameter and FLOP tables, design decisions with their ablation numbers, results, reproduction commands |
| [`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md) | literature survey, what was borrowed from where, gap analysis against published hangman solutions |
| [`reports/FINETUNING_REPORT.md`](reports/FINETUNING_REPORT.md) | the Qwen3-4B stage: bidirectionalisation, character tokenisation, head surgery, LoRA, layer truncation, throughput accounting |
| [`reports/COMPLIANCE.md`](reports/COMPLIANCE.md) | point-by-point rule adherence, and where to check each claim in the code |

Diagrams: [`reports/figures/architecture.svg`](reports/figures/architecture.svg),
[`reports/figures/pipeline.svg`](reports/figures/pipeline.svg) (regenerate with
`python tools/make_diagram.py`).

## What is different from published hangman solutions

Public solutions to this problem (surveyed in the research report) cluster at 42–68%, and share
three omissions that this design targets:

1. **They discard the absent-letter set.** Knowing `q`, `x`, `z` are *not* in the word is strong
   evidence about what is. Here both guessed sets are first-class inputs, injected into every
   block by FiLM modulation rather than left to be routed through attention.
2. **They train on randomly masked words.** That distribution is unreachable — a live game carries
   at most five absent letters — and its absent sets are dominated by rare letters, where a real
   game's are dominated by frequent ones. DAgger self-play fixes the mismatch.
3. **They play `argmax P(letter present)` without ever checking whether it is the right rule.**
   Here the decision rule is *selected* on held-out words: value-guided lookahead and exact
   information-set search are both implemented, both measured, and both reported — including the
   finding that on this data they do **not** beat greedy, and why (see below).

Plus one correction that turned out to matter more than any of them: pooling per-position
distributions with **noisy-OR** instead of the max-pooling used in prior work, worth **+8.0
points** on the n-gram baseline alone.

## On the public leaderboard

The public leaderboard shows 94.52. We measured what produces that rather than guessing: building
the candidate set from `test.txt` itself — so the answer is guaranteed to be in it — scores
**94.26%**, and **2.73%** on words that are not in that set. The private re-run is exactly the
second condition. A perfect Bayes posterior over `train.txt` scores 60.77% on unseen words, and
there is no route from there to 94.5 with training data alone. Details and the measurement script
are in [`reports/COMPLIANCE.md`](reports/COMPLIANCE.md) §8.

## A measured negative result

The planned inference policy — value-guided lookahead plus exact information-set search — turned
out to be **13.5 points worse** than plain greedy, and isolating it was instructive:

* The **value head was constant** (mean 0.5155, std 0.0085) because it is trained on Monte-Carlo
  returns and off-policy warm-start records carry none, so one-ply lookahead was choosing at
  random among the top-5. `ValueGuidedPolicy` now refuses to run when the head shows no spread.
* Every **exact-search** variant sat within noise of greedy once a `miss_penalty` term was removed;
  that term alone cost 1.0 point. Its removal is the same lesson as the 45% statistic from the
  other side — candidate sets drawn from a *disjoint* dictionary are weak evidence, and overruling
  a well-trained network on them loses games.

So the policy is chosen on the validation dictionary rather than assumed, and a search policy has
to beat greedy by more than two standard errors before it ships.

## Hardware notes (NVIDIA DGX Spark, GB10, `sm_121`)

Two findings worth reusing:

* The device has ~95 TFLOPS of bf16 matmul but only 273 GB/s of bandwidth, so a small model is
  **bandwidth-bound**. Profiling showed elementwise ops at 30% of step time against 13% for
  matmuls. Length-bucketed batches plus `torch.compile` fusion gave a **7.6× speedup**
  (1,807 → 13,680 states/s) and cut peak memory from 21.7 GB to 13.7 GB.
* **Concurrency is counter-productive here.** One training job already saturates the GPU: running
  two members together took 247 minutes against 146 minutes sequentially. Training runs
  sequentially by design.

## Layout

```
Data/            train.txt, test.txt, sample_submission.csv (unmodified)
src/hangman/     the library — every script and notebook cell is a thin call into this
scripts/         00..08, one per pipeline stage
tools/           notebook, diagram and stats generators
tests/           rules, teacher matmul equivalence, policy identities
notebooks/       hangman_end_to_end.ipynb
reports/         the four reports plus figures
artifacts/       checkpoints, datasets, logs, metrics (generated)
```

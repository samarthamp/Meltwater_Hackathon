#!/usr/bin/env python
"""Generate notebooks/hangman_end_to_end.ipynb.

The notebook is generated rather than hand-edited so that it stays in sync with
``src/hangman`` and so the cell structure is reviewable in a diff.

    python tools/make_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "hangman_end_to_end.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# Neural Hangman

End-to-end solution: a **bidirectional character Transformer** trained from scratch on
`train.txt`, supervised by an exact dictionary-posterior oracle and corrected for covariate shift
with **DAgger self-play**. The notebook also covers the recurrent and convolutional baselines, a
bidirectionalised, LoRA-finetuned **Qwen3-4B** teacher, and the search-based inference policies
that were measured against greedy play.

## The two measurements that shaped the design

| Measurement | Value | Why it matters |
|---|---|---|
| `train.txt` ∩ `test.txt` | **0 words** | The dictionaries are disjoint; only generalisation scores. |
| On-policy states with **no** consistent training-dictionary word | **45%** | Dictionary filtering is silent on nearly half the states a real game visits. |

## Baselines to beat (all on a 22,530-word held-out dictionary disjoint from training)

| Method | Win rate |
|---|---|
| Static letter frequency (this is what `sample_submission.csv` does) | 13.99% |
| Interpolated character n-gram with noisy-OR pooling | 60.66% |
| Exact Bayes dictionary posterior + n-gram backoff (**our teacher**) | 61.13% |

## Notebook layout

1. Environment and configuration
2. Data and exploratory analysis
3. The teacher: a three-tier Bayes posterior oracle
4. State encoding and the batched game engine (with rule tests)
5. Model architectures and parameter budgets
6. Stage 0 — reachable-state warm start
7. Stage 1 — DAgger self-play rounds
8. Stage 2 — Qwen3-4B teacher (optional)
9. Stage 3 — selective distillation
10. Stage 4 — policy-gradient refinement
11. Stage 5 — calibration, ensembling, ablation table
12. Submission and validation

> **Reproducibility.** Every stage is a thin call into `src/hangman/`, which is what the
> `scripts/0*.py` CLI drivers use too — the notebook and the command line run identical code.
> `test.txt` is used **only** to simulate games; it never trains or tunes anything.
""")

md("## 1. Environment and configuration")

code(r"""
import os, sys, json, time, math
from pathlib import Path

# The package lives in src/; add it when running from a clone or a Kaggle input.
for candidate in (Path.cwd(), Path.cwd().parent, Path("/kaggle/input/hangman-src")):
    src = candidate / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
        break

import numpy as np
import torch

import hangman
print(f"hangman        : {hangman.__version__}")
print(f"python         : {sys.version.split()[0]}")
print(f"torch          : {torch.__version__}  (CUDA {torch.version.cuda})")
print(f"cuda available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"device         : {props.name}  sm_{props.major}{props.minor}")
    print(f"memory         : {props.total_memory / 1e9:.1f} GB   SMs: {props.multi_processor_count}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
print(f"cpu cores      : {os.cpu_count()}")
""")

code(r"""
# --------------------------------------------------------------------------
# Run switches.  Defaults are set for a full local training run on the DGX
# Spark.  On Kaggle, set TRAIN_FROM_SCRATCH = False and attach the checkpoints
# as a dataset; the training code below still runs unchanged if you flip it on.
# --------------------------------------------------------------------------
TRAIN_FROM_SCRATCH = True     # False -> load checkpoints from CHECKPOINT_DIR
RUN_LLM_TEACHER    = True     # Qwen3-4B stage; needs the local weights
RUN_RL_REFINEMENT  = True
RUN_ABLATIONS      = True

# Reduced-scale switch, for a quick end-to-end pass that exercises every cell.
SMOKE = False

PRETRAIN_STEPS = 500 if SMOKE else 7000
DAGGER_STEPS   = 300 if SMOKE else 4000
DAGGER_ROUNDS  = 1   if SMOKE else 3
OFFPOLICY_STATES = 1_000_000 if SMOKE else 20_000_000
DAGGER_STATES    = 300_000   if SMOKE else 6_000_000
EVAL_WORDS       = 2000 if SMOKE else 0     # 0 = the whole validation dictionary
""")

code(r"""
from hangman.config import ExperimentConfig, ensure_dirs, CHECKPOINT_DIR, DATASET_DIR, METRIC_DIR
from hangman.pipeline import set_seed

ensure_dirs()
set_seed(0)

cfg = ExperimentConfig(run_name="notebook_former")
cfg.optim.batch_size = 4096
cfg.optim.compile = not SMOKE          # compilation is not worth its cost at smoke scale
cfg.collect.offpolicy_states = OFFPOLICY_STATES
cfg.collect.onpolicy_states_per_round = DAGGER_STATES

print(json.dumps({
    "model": cfg.model.__dict__,
    "teacher": cfg.teacher.__dict__,
    "optim": cfg.optim.__dict__,
}, indent=2, default=str))
""")

md(r"""
## 2. Data and exploratory analysis

The single most important property of this dataset is that the train and test word lists share
**no words at all**. Everything downstream — the leave-one-out teacher, the validation protocol,
the decision to bring in an external language prior — follows from that.
""")

code(r"""
from hangman.data import (
    build_dictionaries, load_test_words, length_histogram, unique_letter_frequency,
)

dicts = build_dictionaries(cfg.data)
test_words = load_test_words(cfg.data)

train_all = sorted(set(dicts.fit) | set(dicts.val))
print(f"train.txt : {len(train_all):,} words")
print(f"test.txt  : {len(test_words):,} words")
print(f"overlap   : {len(set(train_all) & set(test_words)):,} words   <-- disjoint dictionaries")
print()
print(dicts.summary())
""")

code(r"""
import matplotlib
import matplotlib.pyplot as plt
from hangman.config import FIGURE_DIR, REPO_ROOT

train_hist = length_histogram(train_all)
test_hist = length_histogram(test_words)
train_freq = unique_letter_frequency(train_all)
test_freq = unique_letter_frequency(test_words)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
lengths = np.arange(1, 30)
axes[0].bar(lengths - 0.2, 100 * train_hist[1:30] / train_hist.sum(), width=0.4, label="train")
axes[0].bar(lengths + 0.2, 100 * test_hist[1:30] / test_hist.sum(), width=0.4, label="test")
axes[0].set_xlabel("word length"); axes[0].set_ylabel("% of words")
axes[0].set_title("Length distribution: identical shape, disjoint words"); axes[0].legend()

order = np.argsort(-train_freq)
axes[1].bar(np.arange(26) - 0.2, train_freq[order], width=0.4, label="train")
axes[1].bar(np.arange(26) + 0.2, test_freq[order], width=0.4, label="test")
axes[1].set_xticks(np.arange(26))
axes[1].set_xticklabels([hangman.ALPHABET[i] for i in order])
axes[1].set_ylabel("fraction of words containing the letter")
axes[1].set_title("Per-letter word coverage"); axes[1].legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "eda_distributions.png", dpi=130)
plt.show()

print("Mean length  train %.2f   test %.2f" % (
    sum(len(w) for w in train_all) / len(train_all),
    sum(len(w) for w in test_words) / len(test_words),
))
""")

md(r"""
### What the length breakdown implies

Almost all of the losses live in **short words**. Measured with the n-gram baseline, win rate
runs from ~11% at length 4 to ~94% at length 15. Long words reveal enough structure that almost
any competent model finishes them; short words are where capacity, loss weighting and lookahead
have to be aimed. This is why the sampler oversamples lengths 4–9 and why the distillation stage
targets them specifically.
""")

md(r"""
## 3. The teacher: a three-tier Bayes posterior oracle

Training on hard 0/1 "is this letter in the word" labels throws away nearly all of the available
signal. Given a board, the *exact* posterior probability that each letter occurs can be computed
by enumerating the dictionary words consistent with that board, and that posterior is a far
richer target — it carries the inter-letter structure hard labels destroy, the "dark knowledge"
argument of Hinton, Vinyals & Dean (2015). We use it as a soft target: we **distil an exact
solver into a network**.

Three details make it work:

**Leave-one-out.** The played word is removed from its own candidate set. Without this the
teacher can identify the answer outright and would teach the student to expect information it
will never have — train and test dictionaries are disjoint.

**A relaxation ladder.** 45% of on-policy states have no consistent dictionary word at all, so a
single exact posterior is silent on nearly half of them. We fall back through progressively
weaker constraint sets, then to a smoothed character n-gram which is defined everywhere.

**Noisy-OR pooling.** The n-gram produces a distribution per hidden position; those are pooled
into presence marginals with `1 − Π(1 − pᵢ)`. This is the correct probabilistic aggregation and
it matters a lot: switching from max-pooling to noisy-OR moved the n-gram baseline from 52.7% to 60.7%.
""")

code(r"""
from hangman.pipeline import build_teacher, evaluate_frequency, evaluate_ngram, evaluate_teacher

dicts, ngram, teacher = build_teacher(cfg, dicts=dicts)
print(f"n-gram contexts: {ngram.keys.size:,}   table {ngram.probs.nbytes / 1e6:.0f} MB")
""")

code(r"""
# Baselines on the held-out dictionary.  These are the numbers the neural model
# has to beat, and the teacher's row is the quality of our own labels.
val_words = dicts.val if not EVAL_WORDS else dicts.val[:EVAL_WORDS]

from hangman.evaluate import AblationTable
table = AblationTable()
table.add(evaluate_frequency(cfg, dicts, val_words))
table.add(evaluate_ngram(cfg, teacher, val_words))
baseline_teacher = table.add(evaluate_teacher(cfg, teacher, val_words))
print()
print(baseline_teacher.length_table())
""")

md(r"""
## 4. State encoding and the batched game engine

A hangman state is `(board, present letters, absent letters, lives)`. Two design decisions:

**The absent set is evidence, not bookkeeping.** Knowing `q`, `x`, `z` are *not* in the word
sharply reshapes the posterior over what is. Both guessed sets are first-class inputs, injected
through **FiLM** modulation so they reach every block rather than having to be routed through
attention.

**The guessed sets are sets.** They carry no order, so they are encoded as permutation-invariant
indicator vectors (Deep Sets, Zaheer et al. 2017).

The engine advances thousands of games in lock-step so one forward pass serves the whole batch,
and it implements the published rules exactly — including that a *repeated* letter is a strike,
and that the **sixth** wrong guess is a loss (five are survivable).
""")

code(r"""
from hangman.simulator import BatchGame, reference_play, play_batch
from hangman.state import encode_batch

demo = BatchGame(["hangman", "zyzzyva", "ox"], device=cfg.device)
demo.apply(torch.tensor([0, 24, 16], device=cfg.device))   # 'a' hit, 'y' hit, 'q' miss
batch = encode_batch(demo.board(), demo.lengths, demo.present, demo.absent)

def render(row: int) -> str:
    board = demo.board()[row].tolist()
    return "".join("_" if c == 27 else (hangman.ALPHABET[c] if c >= 0 else " ") for c in board)

for i, word in enumerate(demo.words):
    print(f"{word:<10} board={render(i)!r:<12} lives={int(demo.lives()[i])} "
          f"absent={[hangman.ALPHABET[j] for j in demo.absent[i].nonzero().flatten().tolist()]}")
print()
print("tokens  :", tuple(batch["tokens"].shape), "(CLS + board, padded to this batch's width)")
print("globals :", tuple(batch["globals"].shape))
""")

code(r"""
# --- The rules, asserted rather than assumed -------------------------------
game = BatchGame(["cat"], device=cfg.device)
for letter in "bdfgh":                              # five misses
    game.apply(torch.tensor([ord(letter) - 97], device=cfg.device))
assert int(game.wrong.item()) == 5 and bool(game.active().item()), "five misses must be survivable"
game.apply(torch.tensor([ord("j") - 97], device=cfg.device))
assert not bool(game.active().item()), "the sixth wrong guess must end the game"

game = BatchGame(["cat"], device=cfg.device)
game.apply(torch.tensor([2], device=cfg.device))    # 'c' hits
game.apply(torch.tensor([2], device=cfg.device))    # 'c' again reveals nothing new
assert int(game.wrong.item()) == 1, "a repeated letter must cost a life"

# The batched engine must agree with a naive per-game scorer.
rng = np.random.default_rng(0)
words = ["".join(rng.choice(list(hangman.ALPHABET), size=int(rng.integers(2, 20)))) for _ in range(2000)]
seq = "etaonrishdlfcmugypwbvkxjqz"
def fixed(sequence):
    step = {"i": 0}
    def fn(board, lengths, present, absent, lives, active):
        letter = ord(sequence[min(step["i"], len(sequence) - 1)]) - 97
        step["i"] += 1
        return torch.full((board.shape[0],), letter, dtype=torch.long, device=board.device)
    return fn
result = play_batch(words, fixed(seq), device=cfg.device)
won, wrong, strings = result.won.cpu().numpy(), result.wrong.cpu().numpy(), result.guess_strings()
for i, word in enumerate(words):
    ref_won, ref_wrong, ref_n = reference_play(word, seq)
    assert bool(won[i]) == ref_won and int(wrong[i]) == ref_wrong and strings[i] == seq[:ref_n]
print(f"batched engine agrees with the reference scorer on all {len(words):,} random games")
""")

md(r"""
## 5. Model architectures and parameter budgets

Four heads share one trunk:

| Head | Output | Role |
|---|---|---|
| `set_head` | 26 sigmoid logits | **P(letter ∈ word)** — what the game actually asks |
| `pos_head` | per-position 26-way softmax | dense masked-LM supervision; this is what teaches orthography |
| `count_head` | 26 × {0,1,2,≥3} | how *much* a hit reveals, which decides whether a guess is worth a life |
| `value_head` | scalar | P(win from here), the baseline for RL and the evaluator for lookahead |

Three architectures are trained. Architectural spread contributes more to an ensemble than seed
spread alone (Lakshminarayanan et al. 2017), and the recurrent member in particular makes
different mistakes on short words — where the losses are.
""")

code(r"""
from hangman.models import build_model

def parameter_table(model, name: str) -> None:
    print(f"\n{name}  ({model.__class__.__name__})")
    print(f"  {'module':<16} {'parameters':>13}  kind")
    print(f"  {'-' * 16} {'-' * 13}  {'-' * 20}")
    for module, count, kind in model.parameter_report():
        print(f"  {module:<16} {count:>13,}  {kind}")

specs = {
    "HangmanFormer-M": dict(arch="transformer", d_model=512, n_layers=8, n_heads=8, d_ff=1408),
    "HangmanBiGRU":    dict(arch="bigru", d_model=640, rnn_layers=4),
    "HangmanCNN":      dict(arch="dilated_cnn", d_model=512, cnn_layers=10),
}
for name, spec in specs.items():
    probe = ExperimentConfig().model
    for key, value in spec.items():
        setattr(probe, key, value)
    parameter_table(build_model(probe), name)
""")

code(r"""
# Forward-pass shapes and an estimate of inference cost for the whole test set.
model = build_model(cfg.model).to(cfg.device)
with torch.autocast(torch.device(cfg.device).type, dtype=torch.bfloat16):
    out = model(**batch)
print("set_logits  ", tuple(out.set_logits.shape))
print("pos_logits  ", tuple(out.pos_logits.shape))
print("count_logits", tuple(out.count_logits.shape))
print("value       ", tuple(out.value.shape))

n_params = sum(p.numel() for p in model.parameters())
mean_guesses = 10.6          # measured on the baseline
states = len(test_words) * mean_guesses
print(f"\n{n_params / 1e6:.2f}M parameters")
print(f"~{states / 1e6:.1f}M state evaluations to play all {len(test_words):,} test words")
print(f"~{2 * n_params * 11 * states / 1e12:.0f} TFLOPs total  (seconds on this GPU, minutes on a Kaggle T4)")
""")

md(r"""
## 6. Stage 0 — reachable-state warm start

The usual approach trains on *randomly masked* words. That distribution is not the one a game
visits, in two ways:

* **Reachability.** A live state carries at most five absent letters, because the sixth ends the
  game. Uniform masking happily produces states with fifteen known-absent letters, which no game
  reaches, and starves the model of the tight five-strike endgames that decide win rate.
* **Correlation.** A real absent set is biased toward *frequent* letters, because that is what a
  competent policy guesses first. Random absent sets are dominated by `j`, `q`, `x`, `z` — the
  letters a real game rarely wastes a guess on.

So the warm-start sampler draws only reachable states, with absent letters sampled in proportion
to corpus frequency. A state is fully determined by `(word, guessed-letter set)`, so records are
66 bytes each — 20M states fit in 1.3 GB and are memory-mapped during training.
""")

code(r"""
from hangman.dataset import shard_paths
from hangman.pipeline import stage_offpolicy_dataset, dataset_for

if TRAIN_FROM_SCRATCH:
    offpolicy_shards = stage_offpolicy_dataset(cfg, dicts, teacher, n_states=OFFPOLICY_STATES)
else:
    offpolicy_shards = shard_paths(DATASET_DIR, "offpolicy")

pool = dataset_for(cfg, dicts, offpolicy_shards, onpolicy_mix=None)
stats = pool.stats()
print(f"{int(stats['n_records']):,} states")
print(f"  Tier-1 empty (no consistent dictionary word): {100 * stats['empty_tier1_frac']:.1f}%")
for tier in range(4):
    label = ["exact", "relaxed (drop present rule)", "positional only", "n-gram backoff"][tier]
    print(f"  tier {tier} ({label:<28}): {100 * stats[f'tier{tier}_frac']:5.1f}%")
""")

code(r"""
from hangman.train import train_steps, save_checkpoint, clone_for_eval, load_checkpoint
from hangman.pipeline import checkpoint_path, evaluate_model

from hangman.train import TrainState

state, ema, optimizer = TrainState(), None, None
if TRAIN_FROM_SCRATCH:
    state, ema, optimizer = train_steps(model, pool, cfg, PRETRAIN_STEPS, state=state, log_every=200)
    save_checkpoint(checkpoint_path(cfg.run_name, "pretrain"), model, cfg, state, ema)
else:
    model, cfg = load_checkpoint(checkpoint_path(cfg.run_name, "pretrain"))

warm = evaluate_model(cfg, clone_for_eval(model, ema), val_words, tag="stage0/off-policy only")
table.add(warm)
""")

md(r"""
## 7. Stage 1 — DAgger self-play rounds

Behavioural cloning on the wrong state distribution is exactly the covariate shift DAgger was
introduced to fix (Ross, Gordon & Bagnell, 2011). Each round rolls the *current* model out over
the training dictionary, records every state it actually visits, labels those with the teacher,
and **aggregates** them into the pool rather than replacing it. Each state also carries the
Monte-Carlo return of its game, which is what trains the value head.
""")

code(r"""
from hangman.pipeline import stage_onpolicy_dataset

if TRAIN_FROM_SCRATCH:
    dagger_pool = list(offpolicy_shards)
    for rnd in range(1, DAGGER_ROUNDS + 1):
        frozen = clone_for_eval(model, ema)
        dagger_pool.extend(
            stage_onpolicy_dataset(cfg, dicts, teacher, frozen, round_idx=rnd, n_states=DAGGER_STATES)
        )
        mixed = dataset_for(cfg, dicts, dagger_pool, onpolicy_mix=0.70)
        print(f"round {rnd} pool: {int(mixed.stats()['n_records']):,} states, "
              f"{100 * mixed.stats()['onpolicy_frac']:.1f}% on-policy")
        state, ema, optimizer = train_steps(
            model, mixed, cfg, DAGGER_STEPS, state=state, ema=ema, optimizer=optimizer, log_every=200
        )
        save_checkpoint(checkpoint_path(cfg.run_name, f"dagger{rnd}"), model, cfg, state, ema)
        table.add(evaluate_model(cfg, clone_for_eval(model, ema), val_words, tag=f"stage1/DAgger round {rnd}"))
    save_checkpoint(checkpoint_path(cfg.run_name, "final"), model, cfg, state, ema)
""")

md(r"""
## 8. Stage 2 — Qwen3-4B as a posterior teacher  *(optional)*

`train.txt` holds ~203k words after the validation split, and the test dictionary shares none of
them. A model fitted only on that list knows English orthography only as far as those words
expose it. Qwen3-4B has read orders of magnitude more English, so it carries a prior the training
dictionary cannot supply — and the 45% of states where the dictionary oracle is blind are exactly
where an outside prior can still say something.

Four adaptations, each with a reason:

1. **Bidirectional attention.** Reading `_ o _ _ e r` needs the suffix as much as the prefix, and
   a causal decoder cannot see it. We replace the causal mask with a padding-only mask — step one
   of LLM2Vec (BehnamGhader et al. 2024). In `transformers ≥ 4.5x` this is done cleanly by
   passing `attention_mask` as a `{"full_attention": mask}` dict.
2. **Exact character tokenisation.** Board characters map straight to the tokenizer's
   single-character token ids rather than through BPE, so one board position is exactly one token
   and per-position supervision lines up.
3. **Head surgery.** `lm_head` is dropped; a presence head and a per-position head are attached.
   One forward pass, no generation.
4. **Layer truncation + LoRA.** Only the first 18 of 36 blocks are kept and only adapters and
   heads are trained. Orthography is a lower-layer phenomenon and the teacher has to score
   millions of states.

> **Compliance.** The submitted model never runs Qwen at inference — Qwen only shapes the
> student's *targets*, and it is only ever asked about boards built from `train.txt` words, so it
> cannot import test vocabulary. A no-LLM student is trained alongside and both scores are
> reported. See `reports/COMPLIANCE.md`.
""")

code(r"""
if RUN_LLM_TEACHER:
    from hangman.models.llm_teacher import QwenHangmanTeacher, PromptLayout

    llm = QwenHangmanTeacher(cfg.llm).to(cfg.device)
    trainable, total = llm.trainable_parameters()
    print(f"Qwen3-4B: kept {len(llm.backbone.layers)} of 36 blocks -> {total / 1e9:.2f}B params "
          f"({trainable / 1e6:.1f}M would be trainable under LoRA)")

    ids, pad = llm.codec.encode(
        demo.board(), demo.lengths, demo.present, demo.absent, PromptLayout.for_length(7)
    )
    print(f"prompt for a 7-letter board: {ids.shape[1]} tokens")
    print("decoded:", repr(llm.tokenizer.decode(ids[0][pad[0]])))

    # Bidirectionality, verified rather than asserted: changing a LATER board
    # character must change an EARLIER position's output, which is impossible
    # under a causal mask.
    b1 = demo.board().clone(); b2 = b1.clone(); b2[0, 6] = 25
    with torch.no_grad():
        h1 = llm(b1, demo.lengths, demo.present, demo.absent)["pos_logits"][0, 0]
        h2 = llm(b2, demo.lengths, demo.present, demo.absent)["pos_logits"][0, 0]
    delta = float((h1 - h2).abs().max())
    print(f"later character changes position 0 by {delta:.3f} -> attention is bidirectional: {delta > 1e-3}")
    del llm; torch.cuda.empty_cache()
""")

md(r"""
### Running the finetune and the distillation

Both stages are long enough that they belong on the command line rather than in a cell; the
notebook calls the same functions. See `reports/FINETUNING_REPORT.md` for the layer-truncation
study and the measured teacher quality.

```bash
python scripts/04_finetune_llm.py --keep-layers 18 --mntp-steps 1500 --sft-steps 6000
python scripts/05_distill.py --adapters artifacts/checkpoints/qwen_teacher_adapters.pt --alpha 0.5
```
""")

md(r"""
## 9. Stage 3 — selective distillation

Scoring states with a 4B model costs roughly four orders of magnitude more than scoring them
with the dictionary oracle, so we spend it only where the oracle is blind: states with an empty
candidate set, and short words. Everywhere else the exact posterior is already the right answer
and an LLM opinion could only add noise.

The distillation shards carry the blend `α·bayes + (1−α)·qwen` as their target, so fine-tuning on
them *is* the distillation objective — no new loss plumbing, and the blend plays the role of the
usual hard/soft mixture.
""")

code(r"""
distill_shards = shard_paths(DATASET_DIR, "distill")
if distill_shards and TRAIN_FROM_SCRATCH:
    combined = list(shard_paths(DATASET_DIR, "offpolicy"))
    for rnd in range(1, DAGGER_ROUNDS + 1):
        combined.extend(shard_paths(DATASET_DIR, f"onpolicy_{cfg.run_name}_r{rnd}"))
    combined.extend(distill_shards)
    distilled = dataset_for(cfg, dicts, combined, onpolicy_mix=0.70)
    state, ema, optimizer = train_steps(
        model, distilled, cfg, DAGGER_STEPS, state=state, ema=ema, optimizer=optimizer, log_every=200
    )
    save_checkpoint(checkpoint_path(cfg.run_name, "distilled"), model, cfg, state, ema)
    table.add(evaluate_model(cfg, clone_for_eval(model, ema), val_words, tag="stage3/+ Qwen distillation"))
else:
    print("no distillation shards found - run scripts/05_distill.py first (this stage is optional)")
""")

md(r"""
## 9b. Generation two — self-distillation

Once generation one is trained, measure its labels against its own output. On 200,000 sampled
states, top-1 hit rate:

| Target source | Top-1 |
|---|---:|
| Bayes posterior (what generation one learned from) | 0.6813 |
| **generation one's own predictions** | **0.7568** |

The oracle is 7.5 points worse than the model it trained, so for generation two the model is the
better teacher — the setting of Born-Again Networks (Furlanello et al., 2018). The gate between
the two sources is *chosen by measurement*, not assumed; `tools/blend_sweep.py` finds that no
mixture beats the student alone, because the student was trained on that posterior and has
already absorbed it.

```bash
python tools/blend_sweep.py --ckpt artifacts/checkpoints/former_a_final.pt
python tools/relabel_with_student.py --ckpt artifacts/checkpoints/former_a_final.pt --kappa inf
python scripts/02_train.py --run former_v2 --pool-prefix sd \
    --self-distill-ckpt artifacts/checkpoints/former_a_final.pt --kappa-sd inf \
    --w-pos 1.0 --pretrain-steps 18000 --dagger-steps 6500 --dagger-rounds 3
```

Generation two also raises `w_pos` from 0.40 to 1.0 — section 11 shows the per-position head is
the *better* presence estimator, yet it trained at less than half the weight — and roughly doubles
the step count, since generation one's loss was still falling when it stopped.
""")

md(r"""
## 10. Stage 4 — policy-gradient refinement

Supervised training optimises agreement with a posterior. The objective rewards *finishing
words within six misses*, and those are not the same objective: the supervised loss is
indifferent to the order in which lives are spent, and the game is not.

We induce an action distribution from the presence head, `π(ℓ|s) = softmax(set_logits/τ)` over
unguessed letters — greedy play is the `τ → 0` limit, so the supervised model is already a
sensible starting policy — and optimise `win − 0.001·wrong_guesses`, matching the published
tiebreak `Score − E/10⁸`. The value head supplies the baseline (Williams 1992) and a KL penalty
back to the frozen supervised policy keeps the refinement local (Schulman et al. 2017); without
it the policy sharpens into a near-deterministic argmax within a few hundred updates and the
probabilities stop being usable by the lookahead policy.
""")

code(r"""
if RUN_RL_REFINEMENT and TRAIN_FROM_SCRATCH:
    from hangman.rl import refine
    import copy

    before = evaluate_model(cfg, clone_for_eval(model, ema), val_words, tag="stage4/before RL")
    print(before.summary())
    candidate = clone_for_eval(model, ema)
    _, rl_stats = refine(
        candidate, dicts, cfg.rl,
        steps=100 if SMOKE else 600, games_per_step=2048, device=cfg.device, log_every=50,
    )
    after = evaluate_model(cfg, candidate, val_words, tag="stage4/after RL")
    table.add(after)
    if after.win_rate >= before.win_rate:
        save_checkpoint(checkpoint_path(cfg.run_name, "rl"), candidate, cfg, state, ema=None)
        print(f"kept the refined policy (+{after.win_rate - before.win_rate:.2f} points)")
    else:
        print(f"RL regressed ({before.win_rate:.2f}% -> {after.win_rate:.2f}%); keeping supervised weights")
""")

md(r"""
## 11. Stage 5 — calibration, ensembling and the inference policy

`argmax P(letter ∈ word)` maximises hit rate, not win rate, and games here end at an average of
~4.0 wrong guesses out of six — so most finish near the wall, where the two objectives diverge.
That is a good reason to expect a search policy to help, and two are implemented: value-guided
one-ply lookahead using the value and count heads, and exact information-set search over small
candidate sets (Meng & Lucas 2024; Reams 2009).

**Both were measured, and neither shipped.** The cells below reproduce that result rather than
asserting it:

* the value head is trained on Monte-Carlo returns, which off-policy warm-start records do not
  carry, so before the first DAgger round it is *constant* and one-ply lookahead degenerates into
  a random choice among the top-5;
* every exact-search variant sits within noise of greedy once a penalty for letters absent from
  the candidate set is removed — that penalty alone cost 1.0 point, because candidate sets drawn
  from a dictionary disjoint from the evaluation words are weak evidence.

So the decision rule is **selected** on the validation dictionary, not assumed: a search policy
has to beat greedy by more than two standard errors before it is used. The same applies to the
noisy-OR blend weight and to the ensemble's weights and temperatures.
""")

code(r"""
from hangman.calibrate import collect_calibration_states, fit_calibration
from hangman.policy import EnsembleSource, GreedyPolicy, ModelSource, build_policy
from hangman.evaluate import evaluate_words

member_paths = sorted(CHECKPOINT_DIR.glob("*_final.pt"))
print("ensemble members:", [p.stem for p in member_paths] or "(none - using the notebook model)")

if member_paths:
    members = [load_checkpoint(p, use_ema=True)[0] for p in member_paths]
else:
    members = [clone_for_eval(model, ema)]
sources = [ModelSource(m, noisy_or_weight=0.25) for m in members]

for path, source in zip(member_paths or ["notebook"], sources):
    name = getattr(path, "stem", path)
    table.add(evaluate_words(val_words, lambda L, s=source: GreedyPolicy(s),
                             device=cfg.device, tag=f"single/{name}"))

if len(sources) > 1:
    uniform = EnsembleSource(sources)
    table.add(evaluate_words(val_words, lambda L: GreedyPolicy(uniform),
                             device=cfg.device, tag="ensemble/uniform"))
    logits, labels = collect_calibration_states(
        dicts.val[-4000:], sources, lambda L: GreedyPolicy(uniform), device=cfg.device
    )
    fit = fit_calibration(logits, labels)
    print("calibration:", fit.summary())
    calibrated = EnsembleSource(sources, weights=fit.weights.to(cfg.device),
                                temperatures=fit.temperatures.to(cfg.device))
    table.add(evaluate_words(val_words, lambda L: GreedyPolicy(calibrated),
                             device=cfg.device, tag="ensemble/calibrated"))
else:
    calibrated = sources[0]
""")

code(r"""
if RUN_ABLATIONS:
    primary = sources[0]
    for mode in ("greedy", "value", "exact", "auto"):
        pcfg = ExperimentConfig().policy
        pcfg.mode = mode
        table.add(evaluate_words(
            val_words,
            lambda L, m=mode, p=pcfg: build_policy(
                primary, p, teacher=teacher if m in ("exact", "auto") else None, length=L
            ),
            device=cfg.device, tag=f"policy/{mode}",
        ))

print()
print(table.to_markdown())
table.save(METRIC_DIR / "notebook_ablation.json")
""")

md(r"""
## 12. Submission

For each test word we *simulate* the game under our policy and record the letters actually
guessed. The same simulation self-scores the submission, and the validator below re-reads the
CSV and re-scores it from scratch with a naive per-game scorer.
""")

code(r"""
from hangman.submit import generate_submission, validate_submission, save_stats

cfg.policy.mode = "auto"

# The auxiliary heads the lookahead needs (count, value) belong to individual
# models, so the search policies read them from the primary member.
def submission_policy(length: int):
    return build_policy(sources[0], cfg.policy, teacher=teacher, length=length)

out_path, gen_stats = generate_submission(
    test_words, submission_policy, REPO_ROOT / "submission.csv",
    device=cfg.device, verbose=False,
)
print("generated:", gen_stats.summary())

verified = validate_submission(out_path, test_words)
print("validated:", verified.summary())
assert abs(verified.win_rate - gen_stats.win_rate) < 1e-9, "generator and validator disagree"
save_stats(verified, METRIC_DIR / "submission.json")

import pandas as pd
print()
print(pd.read_csv(out_path).head(8).to_string(index=False))
""")

md(r"""
## Summary

| Component | What it contributes |
|---|---|
| Exact leave-one-out Bayes posterior as a soft target | calibrated supervision instead of 0/1 labels |
| Relaxation ladder + noisy-OR n-gram backoff | covers the 45% of states where the oracle is blind |
| Absent-set conditioning through FiLM | uses the absent-letter evidence a board-only encoder ignores |
| Noisy-OR pooling of the per-position head | +8.0 points on the n-gram baseline, +3.2 on the model |
| DAgger self-play rounds | removes the covariate shift random masking bakes in |
| Heterogeneous ensemble + fitted calibration | independent errors, correctly weighted |
| Qwen3-4B distillation on oracle-blind states | an English prior the training dictionary cannot supply |
| Value-guided lookahead, exact endgame search | *measured and not shipped* — see section 11 |

Full write-ups: `reports/MODEL_REPORT.md` (architecture, layer tables, ablations),
`reports/RESEARCH_REPORT.md` (literature and what was borrowed),
`reports/FINETUNING_REPORT.md` (the Qwen stage), `reports/COMPLIANCE.md` (rule adherence).
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {
    "display_name": "Python (hangman)",
    "language": "python",
    "name": "hangman",
}
nb.metadata.language_info = {"name": "python", "version": "3.12"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(cells)} cells)")

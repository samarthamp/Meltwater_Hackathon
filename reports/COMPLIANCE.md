# Compliance Notes

**Meltwater Brand Buzzword Hackathon**

The competition states that hardcoded solutions, lookup leaks and reverse-engineering result in
disqualification, and that results must be reproducible by training on the provided train
dataset. This document states, point by point, what this solution does and how each claim can be
checked in the code.

---

## 1. `test.txt` is never used to fit or tune anything

The public test words are used in exactly one way: to **simulate games** so that the submission
strings can be generated and self-scored. They never enter a training set, a teacher's candidate
dictionary, or a hyperparameter search.

**Where to check:**

* `src/hangman/data.py` — `build_dictionaries()` reads **only** `cfg.data.train_path`.
  `load_test_words()` is a separate function, and the only callers are
  `scripts/09_make_submission.py` and the submission cell of the notebook.
* `src/hangman/pipeline.py` — `build_teacher()` builds the n-gram tables and the candidate
  buckets from `dicts.fit`, which is derived from `train.txt` alone.
* `grep -rn "load_test_words\|test_path" src/ scripts/` returns exactly four lines: the
  declaration in `DataConfig`, the definition and body of `load_test_words` in `data.py`, and its
  single call site in `scripts/09_make_submission.py`. There is no other reader of `test.txt` in
  the codebase.
* `grep -c llm_teacher scripts/09_make_submission.py src/hangman/policy.py src/hangman/submit.py`
  returns `0` for all three: the inference path has no reference to the pretrained model.

Model selection and every hyperparameter (teacher `κ`, n-gram order, noisy-OR blend, policy
choice, ensemble weights and temperatures) were chosen on a **held-out slice of `train.txt`**,
described in §3. That slice is further split so that nothing is fitted and reported on the same
words: the last 5,000 validation words are the fitting set, and every number in the reports comes
from the 17,530 that remain.

## 2. No lookup tables, no memorisation of answers

The shipped model is a neural network whose only inputs are the current board, the two
guessed-letter sets and the remaining lives. It has no access to a word list at inference.

One component *does* consult the training dictionary at inference: the optional exact
information-set search (`ExactSearchPolicy` in `src/hangman/policy.py`), which enumerates
**training-dictionary** words consistent with the board when that set is small. This is
deliberate and it is not a leak:

* The candidate set is built from `train.txt` only, which shares **zero** words with `test.txt`
  (verified in the notebook's EDA cell and in `artifacts/metrics/baselines.json`).
* It therefore cannot contain the answer. It is used as a *prior over plausible letter patterns*,
  exactly as the n-gram tables are — a smoothing structure, not an answer key.
* Measured: **45.1%** of on-policy states have an empty consistent set, so the mechanism is
  structurally incapable of acting as a lookup.

The primary configuration and the reported headline number can be produced without this component
at all (`--policy greedy`), and both are in the ablation table.

## 3. Reproducible from `train.txt`

The full pipeline is:

```bash
bash scripts/00_setup_env.sh
python scripts/01_build_data.py --states 20000000
bash scripts/03_train_all.sh
python scripts/08_ensemble_eval.py --ckpts artifacts/checkpoints/*_final.pt
python scripts/09_make_submission.py --ckpts artifacts/checkpoints/*_final.pt
```

Every stage reads `Data/train.txt` and nothing else outside the repository, apart from the
optional Qwen stage discussed in §4. Seeds are fixed in `ExperimentConfig`; the dictionary split
is deterministic given `DataConfig.split_seed`, and its hash is recorded in every checkpoint
(`Dictionaries.digest`) so a rerun can be confirmed to have used the same split.

The notebook contains the same code and runs it on the `hangman` kernel; `TRAIN_FROM_SCRATCH`
switches between full training and loading the checkpoints that training produced.

## 4. The pretrained LLM: what it contributes, and the self-contained fallback

A local **Qwen3-4B** checkpoint is used as a *teacher* in an optional stage. Three properties
matter for the audit:

1. **It is never run at inference.** The submitted model is the compact character transformer
   ensemble. Qwen influences it only through training targets. `scripts/09_make_submission.py`
   never imports `hangman.models.llm_teacher`.
2. **It is never asked to produce vocabulary.** It is only ever queried about boards constructed
   from `train.txt` words, and only for a 26-dimensional per-letter posterior. There is no
   generation, no sampling, and no path by which a test word could be produced. See
   `scripts/05_distill.py`: inputs come from record shards whose `word_id` indexes `dicts.fit`.
3. **A no-LLM model is trained and reported alongside.** The `*_final.pt` checkpoints are
   trained on `train.txt` alone; the distillation stage produces separate `*_distilled.pt`
   checkpoints. Both appear in the ablation table in `reports/MODEL_REPORT.md`, so the
   contribution of external knowledge is quantified rather than hidden, and the submission can be
   switched to the fully self-contained model by pointing `--ckpts` at the `_final.pt` files.

If the reviewing team prefers a solution with no external pretrained weights in its provenance,
the `_final.pt` ensemble is that solution and its score is reported.

## 5. No external API calls

Nothing in `src/` or `scripts/` makes a network request. The only downloads happen in
`scripts/00_setup_env.sh`, which installs the conda environment and Python packages. The Qwen
weights are read from a local path (`LLMConfig.model_path`). `requirements-kaggle.txt` lists the
inference-path dependencies, all of which are preinstalled in the Kaggle image.

## 6. The competition rules are implemented as specified, strictly

`src/hangman/simulator.py` implements:

* all occurrences of a correct letter revealed at once;
* **any** character that fails to reveal something new counts as exactly one wrong guess,
  including a repeat of an already-guessed letter;
* the game terminates on the **sixth** wrong guess (five are survivable) or on completion.

The problem statement's own example loop uses `while wrong_guesses <= 6`, which would permit a
seventh guess. That contradicts its normative text ("the moment your `guessed_letters_string`
triggers a 6th wrong guess"), so the strict rule is implemented. This is the conservative choice:
a policy tuned for six survivable misses would lose games under a strict scorer, but never the
reverse.

`tests/test_simulator.py` asserts each of these and additionally checks the batched engine against
a naive single-game reference scorer on 2,000 random words, comparing win flag, strike count and
the exact guess string produced.

## 7. Submission integrity

`src/hangman/submit.py` writes the CSV and then **re-reads it and re-scores it from scratch** with
the naive reference scorer, checking:

* exactly 250,000 rows plus a header, columns `word_id,guessed_letters_string`;
* `word_id` equal to `0..249,999` in order, matching `test.txt` line order;
* every guess string lowercase `a-z` with no repeated letters;
* no game abandoned before either a win or six strikes.

`scripts/09_make_submission.py` fails loudly if the generator's win rate and the validator's
disagree by any amount, so the reported score cannot drift from what the file actually says.

## 8. On the public leaderboard, and why we did not chase it

At the time of writing the public leaderboard shows **94.52**. We investigated what produces a
score of that shape rather than guessing, because the answer determines whether it is worth
pursuing. `tools/leak_ceiling.py` measures it directly:

| Approach | Win rate |
|---|---:|
| Filter `test.txt` itself, guess the most common letter (the answer **is** in the candidate set) | **94.26%** |
| Same, with a survival tie-break | **94.42%** |
| *Reported leaderboard* | *94.52* |
| **The same solver on words not in its candidate set** | **2.73%** |

A **91.5-point collapse**. Building the candidate set from the evaluation words is what the rules
call a lookup leak, and the private re-run is precisely the condition that exposes it: the code
still filters `test.txt`, which does not contain the private words.

For scale, the legitimate ceiling: an exact Bayes posterior over `train.txt`, evaluated on words
absent from it, scores **60.77%**. There is no path from 60.77 to 94.5 using the training data —
reaching it would require roughly 90% on obscure six-letter words with six lives, and that
information is not present in `train.txt`. Our model reaches 68.48% on held-out words and 72.10%
on the public test set.

The measurement is recorded in `artifacts/metrics/leak_investigation.json`. Nothing in
`tools/leak_ceiling.py` or `tools/failure_analysis.py` is imported by the submission path; both
carry a header saying so.

## 9. Honest reporting

* The validation dictionary holds 22,530 words, so the standard error of a win rate near 60% is
  **±0.32 percentage points**. Differences smaller than roughly 0.7 points are not separable and
  are not claimed as improvements.
* Ablation rows that required their own training run are marked as such; rows that vary only the
  inference path are marked as such.
* Where a stage failed to help, that is stated with its number rather than omitted — for example,
  `scripts/07_rl_refine.py` keeps the supervised checkpoint and prints the regression if
  policy-gradient refinement does not improve the held-out win rate.

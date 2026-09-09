# Submission checklist

Everything the Kaggle submission needs, in the order you need it. The competition is
code-based — you submit a **notebook**, not a predictions file — and it is judged by an
internal re-run against a firewalled private dictionary, so what matters is that the
notebook runs and that the method generalises.

---

## Before anything else — the two administrative requirements

From the problem statement, both are prerequisites for being graded at all:

1. **Add the hosts as collaborators.** Open the submission notebook in the Kaggle editor →
   **Share** → add `chetanatanmayee` and `atharva212` → permission level **Viewer** → **Save**.
2. **Rename your Kaggle Team Name.** The problem statement says this must be done immediately,
   for grading and tracking.

Neither is something I can do from here.

---

## What to upload

### 1. The weights, as a Kaggle Dataset

```bash
conda activate hangman
python tools/export_for_kaggle.py --half \
    --ckpts artifacts/checkpoints/former_a_final.pt artifacts/checkpoints/former_v2_final.pt
```

The submitted model is the **uniform two-member ensemble** of `former_a` and `former_v2`, so both
checkpoints are needed. In the notebook, keep `--uniform` behaviour: do **not** apply the fitted
calibration, which measured 1.46 points worse than a plain average on these two members
(`reports/MODEL_REPORT.md` §10a).

This writes `kaggle_upload/` containing a slimmed checkpoint (EMA weights folded in, training
history dropped, fp16) plus the calibration file. A 229 MB training checkpoint becomes ~57 MB.

Upload that directory as a **private Kaggle Dataset** named `hangman-weights`.

### 2. The notebook

Upload [`notebooks/kaggle_submission.ipynb`](notebooks/kaggle_submission.ipynb). It is
**self-contained** — the model definition and the game engine are vendored inline, so it does not
need this repository. Attach the `hangman-weights` dataset to it. It:

* loads the checkpoint(s) and the fitted calibration;
* plays all 250,000 test words in length-bucketed lock-step batches (~3 min on a T4);
* writes `submission.csv`;
* **re-reads that CSV and re-scores it from scratch** with a naive per-game scorer, asserting
  250,000 rows, `word_id` 0…249,999 in order, lowercase `a–z` only, no repeated letters, and
  that the generator and validator agree exactly.

If `WEIGHTS_DIR` does not resolve on Kaggle, the notebook stops with a clear message rather than
producing a silently wrong file.

### 3. Optional but worth attaching

[`notebooks/hangman_end_to_end.ipynb`](notebooks/hangman_end_to_end.ipynb) is the reproducibility
artefact: the complete pipeline from `train.txt` to a submission, the same code the scripts call.
Setting `TRAIN = True` in the submission notebook, or running this one, reproduces the model from
the provided training data alone — which is what the rules require.

---

## Sanity checks before you submit

```bash
# The file is what we think it is.
python - <<'PY'
import pandas as pd
df = pd.read_csv("submission.csv", dtype={"word_id": int, "guessed_letters_string": str})
df["guessed_letters_string"] = df["guessed_letters_string"].fillna("")
assert list(df.columns) == ["word_id", "guessed_letters_string"]
assert len(df) == 250_000, len(df)
assert df["word_id"].tolist() == list(range(len(df)))
s = df["guessed_letters_string"]
assert s.map(lambda x: set(x) <= set("abcdefghijklmnopqrstuvwxyz")).all()
assert s.map(lambda x: len(set(x)) == len(x)).all()
print(f"OK — {len(df):,} rows, mean {s.str.len().mean():.2f} guesses per word")
PY

# The tests still pass.
python -m pytest tests/ -q
```

---

## What to point the reviewers at

The competition weights "Algorithmic Sophistication", "Generalization Integrity" and "Code
Cleanliness" above the leaderboard. The four reports are written for that audience:

| Document | What it answers |
|---|---|
| [`reports/MODEL_REPORT.md`](reports/MODEL_REPORT.md) | architecture, layer tables, every design decision with its ablation number, results, failure analysis |
| [`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md) | the literature, what was borrowed from where, and the three things published hangman solutions omit |
| [`reports/FINETUNING_REPORT.md`](reports/FINETUNING_REPORT.md) | the Qwen3-4B teacher, and the measurement showing it did not help |
| [`reports/COMPLIANCE.md`](reports/COMPLIANCE.md) | rule-by-rule adherence, with the grep commands that verify each claim — including §8 on the leaderboard |

**One thing worth saying explicitly in your submission description:** the public leaderboard's
94.52 is reproducible in one line by building the candidate set from `test.txt` itself, which
scores 94.26% on `test.txt` and **2.73%** on words it has not been given. Our 72.955% is lower and
is the same number on both. `reports/COMPLIANCE.md` §8 has the measurement.

**Final numbers for the description:** 72.955% on the public test set (182,388 / 250,000),
leaderboard 72.946978, E = 822,235, 10.20 guesses per word. Held-out validation: 69.61% against a
60.97% exact-Bayes teacher and a 14.18% frequency baseline.

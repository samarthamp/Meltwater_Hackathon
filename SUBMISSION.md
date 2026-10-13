# Running the Kaggle submission

The competition is code-based: a **notebook** is submitted rather than a predictions file, and it
is re-run against a private word list. This is how the submitted notebook is packaged and run.

## 1. Export the weights as a Kaggle Dataset

```bash
conda activate hangman
python tools/export_for_kaggle.py --half \
    --ckpts artifacts/checkpoints/former_a_final.pt artifacts/checkpoints/former_v2_final.pt
```

The submitted model is the **uniform two-member ensemble** of `former_a` and `former_v2`, so both
checkpoints are needed. The export folds in the EMA weights, drops the training history and casts
to fp16, turning a 229 MB training checkpoint into ~57 MB. The result is written to
`kaggle_upload/` (already committed in this repository) along with the calibration file and the
dataset metadata.

Upload that directory as a private Kaggle Dataset named `hangman-weights`.

## 2. Run the notebook

Upload [`notebooks/kaggle_submission.ipynb`](notebooks/kaggle_submission.ipynb) and attach the
`hangman-weights` dataset and the competition data. The model definition and game engine are
vendored inline, so the notebook does not need this repository. It:

* discovers the test words and checkpoints under `/kaggle/input`, stopping with a clear message if
  either is missing;
* loads both checkpoints and averages them with equal weight (`USE_FITTED_CALIBRATION = False`;
  the fitted weights measured 1.46 points worse than a plain average — see
  `reports/MODEL_REPORT.md` §10a);
* plays all 250,000 test words in length-bucketed lock-step batches;
* writes `submission.csv`;
* **re-reads that CSV and re-scores it from scratch** with a naive per-game scorer, asserting
  250,000 rows, `word_id` 0…249,999 in order, lowercase `a–z` only, no repeated letters, and that
  the generator and validator agree exactly.

Setting `TRAIN = True` in the notebook, or running
[`notebooks/hangman_end_to_end.ipynb`](notebooks/hangman_end_to_end.ipynb), reproduces the
model from `train.txt` alone.

## 3. Sanity-check the output

```bash
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

python -m pytest tests/ -q
```

## Shipped files

| File | What it is |
|---|---|
| `submission_ensemble.csv` | the submitted output of the uniform ensemble: 72.955% (182,388 / 250,000), 10.20 guesses per word |
| `submission_former_a_72.101.csv` | the single-model (`former_a`) output, 72.101%, kept for comparison |

# Research Report — Neural Hangman

**Meltwater Brand Buzzword Hackathon**
Author: m.samartha@students.iiit.ac.in

This report surveys what has been published on machine-learned Hangman and on the techniques
this solution borrows, states what was taken from where, and identifies the gaps in prior work
that the design targets. Numbers quoted as "measured" were produced by the code in this
repository on the provided data; see `reports/MODEL_REPORT.md` for the full result tables.

---

## 1. The problem, stated precisely

Hangman with six lives is a **sequential decision problem under partial observability**. The
state is `(revealed pattern, letters guessed and present, letters guessed and absent, lives
remaining)`; the action is a letter; the reward is 1 if the word is fully revealed before the
sixth wrong guess. The hidden word is drawn from a distribution the agent can only learn from a
training dictionary.

The property that shapes everything is this one, measured on the provided files:

> `train.txt` (225,300 words) and `test.txt` (250,000 words) share **exactly zero words**.

So the task is not "learn this dictionary". It is "learn the generative structure of the
language this dictionary was drawn from, well enough to play words you have never seen". Two
further measurements make the point concrete:

| Measurement | Value |
|---|---|
| Length distributions, train vs test | near-identical (mean 9.35 vs 9.43) |
| On-policy game states with **no** consistent `train.txt` word | **45.1%** |
| Win rate of a static letter-frequency policy | 13.99% |

The 45.1% figure is the important one. It says that for nearly half of the positions a real game
reaches, the exact Bayesian answer *conditioned on the training dictionary* is the empty set —
there is simply nothing to look up. Any approach built on candidate-set filtering hits a wall
there, and the wall is not a small one.

---

## 2. Prior work on learned Hangman

### 2.1 What the public solutions do

The best-documented line of work is the set of public solutions to the **Trexquant** Hangman
interview challenge, which poses essentially this problem — a large obscure dictionary, six
lives, disjoint evaluation words. Reported win rates cluster between **42% and 68%**, with most
credible implementations in the 50–60% band:

* **Transformer encoder, masked-letter training** — [ryanyuan42/transformer-hangman-bot][thb].
  A BERT-style encoder over characters; 40% of letters masked during training; the model outputs
  a 26-way distribution per position. Three versions are documented: V1 uses per-position
  cross-entropy, V2 adds **max-pooling** across positions into a single 26-way distribution
  trained with KL divergence, V3 suppresses already-guessed letters by setting their scores to
  `-inf`. Reported: 42% (small model), 60% (larger).
* **BiLSTM + n-gram hybrid** — several implementations combine an n-gram model for the opening
  guesses with a BiLSTM for the rest, reporting ~62% over 1,000 games and ~68% over a 100-game
  sample (the latter has a standard error of roughly ±4.7 points, so it is not distinguishable
  from ~60%).
* **LSTM over n-gram features** — [aghalandar/Hangman_solution][ag] and similar, reporting
  51–55%.
* **BERT backbone inside a DQN** — [ZosoV/pretrained_rl_hangman][zos] proposes a pretrained BERT
  as the feature extractor for a Q-network, on the argument that BERT already encodes character
  co-occurrence structure and the RL layer only has to learn the decision rule.
* **Recurrent PPO** — [sagarnildas][sag] reports >60% with a recurrent policy trained by PPO,
  motivated by the need to carry memory of past guesses through a partially observed episode.

### 2.2 Three gaps, and what this solution does about them

Reading across those implementations, three things are consistently absent. Each is a design
axis in this solution.

**(a) The absent-letter set is discarded.** Every surveyed implementation encodes the *board* and,
at best, suppresses already-guessed letters at the output. The `transformer-hangman-bot` README
is explicit that its V3 "doesn't account for letters guessed but absent during gameplay" and
notes this would need "bigger dataset or alternative architectures". But knowing that `q`, `x`
and `z` are **not** in the word is strong evidence about what is — and it is evidence the model
gets for free. Here the guessed-present and guessed-absent sets are first-class inputs, encoded
as permutation-invariant indicator vectors (§3.4) and injected into every transformer block by
FiLM modulation (§3.3) rather than left to be routed through attention.

**(b) Training uses the wrong state distribution.** All of them train on *randomly masked* words.
That distribution is not the one a game visits, in two ways that matter:

* *Reachability.* A live state carries at most five absent letters, because the sixth ends the
  game. Uniform masking produces states with fifteen known-absent letters, which no game ever
  reaches, and correspondingly under-samples the tight five-strike endgames that decide win rate.
* *Correlation.* A real absent set is biased toward **frequent** letters, because that is what a
  competent policy guesses first. Random absent sets are dominated by `j`, `q`, `x`, `z` — the
  letters a real game rarely spends a guess on.

This is textbook covariate shift in imitation learning, and the standard remedy is DAgger (§3.5).
Measured here it is worth **+1.50 points** in a single round on the primary transformer — the
largest training-side gain in this project — and it saturates immediately afterwards. On the
weaker Bi-GRU member the same procedure is worth **+5.84 points**, which is the expected
direction: a model that makes more mistakes visits states further from the randomly masked
distribution, so correcting the mismatch buys more.

**(c) The decision rule is `argmax P(letter present)`.** That maximises hit rate, not win rate.
Measured here, games end at an average of ~4.0 wrong guesses out of six, so most finish near the
wall, and near the wall the two objectives come apart — a slightly less likely letter that
reveals three positions can be worth more than a marginally likelier one that reveals one.
Separately, once the consistent candidate set is small the problem stops being statistical and
becomes a solved search problem (§3.7).

---

## 3. Techniques borrowed, and why

### 3.1 Bidirectional character-level modelling

**Borrowed from:** Vaswani et al. (2017) for the encoder; Devlin et al. (2019) for the masked
bidirectional objective; Al-Rfou et al. (2018), Clark et al. (2021, CANINE), Xue et al. (2021,
ByT5) and Tay et al. (2022, Charformer) for the case that character granularity is the right
level for spelling-sensitive tasks.

Reading `_ o _ _ e r` requires the suffix as much as the prefix, so the encoder is bidirectional
with no causal mask. ByT5 in particular reports that byte/character-level models outperform
subword models "on tasks sensitive to spelling", which is precisely this task.

### 3.2 Distilling an exact solver — soft targets instead of hard labels

**Borrowed from:** Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network*.

Training on hard "is this letter in the word" labels discards nearly all of the available signal.
Given a board, the exact posterior `P(letter ∈ word | board, guesses)` is computable by
enumerating consistent dictionary words, and that posterior is a far richer target: it is the
*expected* label over all words consistent with the board rather than one sample from it, so it
carries the inter-letter structure hard labels destroy and it sharply reduces gradient variance.
The teacher here plays the role of Hinton's cumbersome model, except that it is exact rather than
learned.

Two adaptations were necessary:

* **Leave-one-out.** The played word is removed from its own candidate set. Without this the
  teacher can identify the answer outright and would teach the student to expect information it
  will never have at test time — the dictionaries are disjoint.
* **A relaxation ladder with smoothed backoff.** Because 45.1% of on-policy states have an empty
  candidate set, the exact posterior alone is silent on nearly half of them. We fall through
  progressively weaker constraint sets and finally onto an interpolated character n-gram, mixed
  by candidate count with a Jelinek-Mercer style weight `λ = n/(n+κ)` — the interpolation idea is
  standard language-model smoothing (Chen & Goodman, 1999).

### 3.3 Conditioning on global constraints — FiLM

**Borrowed from:** Perez et al. (2018), *FiLM: Visual Reasoning with a General Conditioning
Layer*; the zero-initialised adaLN form is as used in diffusion transformers.

The guessed sets and remaining lives are *global* constraints on the whole board, not tokens in a
sequence. Appending them to the sequence forces the network to route them through attention;
FiLM instead lets them scale and shift the activations of every block directly. Generators are
zero-initialised, so training starts from an unconditioned transformer and learns to use the
conditioning.

### 3.4 Permutation-invariant set encoding

**Borrowed from:** Zaheer et al. (2017), *Deep Sets*.

The guessed letters carry no order, so they are encoded as 26-dimensional indicator vectors
rather than as a sequence. This is the trivial case of the Deep Sets result but it is the reason
the encoding is an indicator vector and not, say, a guess-history sequence.

### 3.5 On-policy data aggregation — DAgger

**Borrowed from:** Ross, Gordon & Bagnell (2011), *A Reduction of Imitation Learning and
Structured Prediction to No-Regret Online Learning*.

Supervised learning from an expert on a distribution the learner does not induce is subject to
compounding error. DAgger's fix is to iterate: roll the current policy out, query the expert on
the states it actually visits, **aggregate** those into the dataset, retrain. Here the expert is
the Bayes teacher (§3.2), which is free to query, so DAgger applies almost without modification —
we keep all rounds' data and mix 70% on-policy / 30% off-policy per batch. This directly targets
gap (b) above and is the single largest departure from published Hangman work.

### 3.6 Heterogeneous ensembling and calibration

**Borrowed from:** Lakshminarayanan, Pritzel & Blundell (2017) for the ensemble; Guo et al. (2017)
for temperature scaling.

Three architectures are trained — a transformer, a bidirectional GRU, and a dilated gated CNN
(the last following the receptive-field argument of van den Oord et al., 2016, *WaveNet*, and
Kalchbrenner et al., 2016). Architectural spread produces more independent errors than seed
spread. Members are combined in **logit space**, which for a decision ensemble keeps a confident
correct member from being pulled toward the mean by an uncertain one, with per-source
temperatures and non-negative weights fitted on the held-out validation dictionary.

### 3.7 Playing for win rate, not hit rate

**Borrowed from:** Reams (2009), *Playing Perfect Hangman*; Barbay & Subercaseaux (2020) on the
complexity of the adversarial variant; Meng & Lucas (2024), *Deduction Game Framework and
Information Set Entropy Search*.

Reams gives a brute-force optimal strategy plus two near-optimal approximations, establishing
that exact search is tractable once the candidate set is small. Barbay & Subercaseaux show that
in *Evil Hangman* a greedy strategy can be arbitrarily far from optimal and that finding the
optimal one is coNP-hard — a useful warning that greedy marginals are a heuristic, not a
solution. Meng & Lucas formalise single-player deduction games in terms of Shannon entropy over
the information set and give a forward search (ISES) that beats information-set MCTS under a
time budget.

This solution takes the practical consequence: when the consistent candidate set is non-empty and
small (≤ 200 words), enumerate it and search exactly; otherwise use the network's marginals, with
a value-head-guided one-ply lookahead in low-life states where the decision is tight.

### 3.8 Policy-gradient refinement

**Borrowed from:** Williams (1992) for REINFORCE with a learned baseline; Schulman et al. (2017)
for the KL-regularised trust region.

Supervised training optimises agreement with a posterior; the leaderboard rewards finishing words.
An action distribution is induced from the presence head, `π(ℓ|s) = softmax(logits/τ)` over
unguessed letters — greedy play is the `τ → 0` limit, so the supervised model is already a valid
starting policy — and refined against `win − 0.001·wrong_guesses`, which mirrors the published
tiebreak `Score − E/10⁸`. The value head supplies the baseline; a KL penalty to the frozen
supervised policy keeps the update local, without which the policy collapses to a
near-deterministic argmax within a few hundred updates and its probabilities stop being usable by
the lookahead policy.

### 3.9 Transferring an external English prior — LLM2Vec + LoRA

**Borrowed from:** BehnamGhader et al. (2024), *LLM2Vec: Large Language Models Are Secretly
Powerful Text Encoders*; Hu et al. (2021), *LoRA*; Dettmers et al. (2023), *QLoRA*.

After the validation split, the training dictionary holds ~203k words, and the test dictionary
shares none of them. A model fitted only on that list knows English orthography only as far as
those words expose it. Qwen3-4B has read orders of magnitude more English text, so it carries a
prior the dictionary cannot supply — and the 45.1% of states where the dictionary oracle is blind
are exactly where an outside prior can still contribute.

LLM2Vec's recipe for turning a decoder into an encoder is (i) enable bidirectional attention,
(ii) adapt with masked next-token prediction, (iii) contrastive training. Steps (i) and (ii) are
directly applicable and are used; step (iii) is not, because we need a per-letter posterior rather
than a sentence embedding, so it is replaced by supervised fitting to the Bayes teacher's
posteriors. LoRA keeps the base weights frozen; QLoRA is implemented but off by default because
`bitsandbytes` is unreliable on this aarch64 / `sm_121` platform and 128 GB of unified memory
makes 4-bit quantisation unnecessary.

Full details, including the layer-truncation study and the throughput accounting that forced
distillation to be *selective*, are in `reports/FINETUNING_REPORT.md`.

### 3.10 Smaller borrowings

| Technique | Source | Where used |
|---|---|---|
| Pre-norm transformer, RMSNorm | Zhang & Sennrich (2019) | every block |
| SwiGLU feed-forward | Shazeer (2020) | every block |
| Rotary position embedding | Su et al. (2021) | character axis, alongside a learned absolute embedding |
| Stochastic depth | Huang et al. (2016) | linearly ramped across blocks |
| AdamW + cosine schedule with warmup | Loshchilov & Hutter (2019) | all training |
| Weight EMA | standard practice | evaluated against raw weights at every checkpoint |
| Label smoothing | Müller et al. (2019) | small, on the presence head |

---

## 4. What is *not* borrowed — contributions of this solution

1. **Noisy-OR pooling instead of max-pooling.** Prior work pools per-position distributions into a
   presence score with a max. The probabilistically correct aggregation is
   `P(present) = 1 − Π_i (1 − p_i(ℓ))`, which max-pooling systematically under-states for letters
   likely to occur in several places. **Measured effect: the n-gram baseline moved from 52.7% to
   60.7% on the same held-out dictionary from this change alone.**

   The same pooling applied to the network's per-position head turned out to matter just as much.
   Sweeping the blend between the dedicated presence head and the pooled per-position head moved
   the model from 61.0% (presence head alone) to **64.3%** at a blend of 0.6 — **+3.2 points**. The
   per-position head is simply better supervised: it receives one gradient per hidden character
   against one per state for the presence head. This suggests that the natural reading of prior
   work — that the per-position output is an auxiliary and the pooled score a convenience — has it
   backwards.

2. **A leave-one-out Bayes teacher with a constraint-relaxation ladder.** Prior work either uses
   hard labels or dictionary filtering as the *policy*. Using the exact posterior as a *soft
   target*, with the played word removed and with graceful degradation through weaker constraint
   sets into a smoothed n-gram, appears to be new to this problem.

3. **Reachability-aware off-policy sampling.** Sampling states subject to the "at most five absent
   letters" constraint, with absent letters drawn in proportion to corpus frequency, produces a
   warm-start distribution far closer to the game's own than uniform masking does.

4. **Candidate filtering as matrix multiplication.** The consistency predicate is normally written
   as a loop over positions. It can instead be expressed as two matrix products — one counting
   matched revealed positions against a one-hot word matrix, one bounding surplus occurrences of
   guessed letters against a count matrix. **Measured: bit-exact with the elementwise predicate
   and ~3.4× faster**, which is what makes labelling tens of millions of states practical. The
   equivalence is asserted as a unit test in `tests/test_teacher.py`.

5. **Length-bucketed batching as a throughput lever.** On this hardware (273 GB/s against ~95
   TFLOPS bf16) the model is bandwidth-bound, so cost scales with padded sequence width. Drawing
   each batch from a single word length keeps sequences at `L+1` rather than the global 32.
   **Measured: 3.3× faster.** Combined with `torch.compile` fusion of the elementwise work that
   profiling showed to be 30% of step time, total speedup was **7.6×** (1,807 → 13,680 states/s).

---

## 5. Methodological notes

**Validation.** `train.txt` is split 90/10 *by word* into a fit dictionary (202,770) and a
validation dictionary (22,530). The teacher's candidate set is the fit dictionary only, so the
validation words are unseen by both the model and its labels — reproducing the real disjointness.
All hyperparameters and model selection use the validation dictionary. At 22,530 words the
standard error of a win rate near 60% is ±0.32 points, so differences below ~0.7 points are not
separable and are not claimed as such.

**`test.txt` is never trained or tuned on.** The public test words are known, so games can be
simulated against them for an exact public-leaderboard estimate, and that is the only use they
get. See `reports/COMPLIANCE.md`.

**Reported numbers are measured, not estimated.** Every figure in the reports comes from a run
recorded under `artifacts/metrics/`; the ablation table rows are separate training runs, not
ablations of a single run's inference path unless the row says so.

---

## 6. References

1. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
2. Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
3. Al-Rfou, R. et al. (2018). *Character-Level Language Modeling with Deeper Self-Attention.* arXiv:1808.04444.
4. Clark, J. H. et al. (2021). *CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation.* TACL.
5. Xue, L. et al. (2021). *ByT5: Towards a token-free future with pre-trained byte-to-byte models.* arXiv:2105.13626.
6. Tay, Y. et al. (2022). *Charformer: Fast Character Transformers via Gradient-based Subword Tokenization.* ICLR.
7. Hinton, G., Vinyals, O. & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531.
8. Ross, S., Gordon, G. & Bagnell, J. A. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS.
9. Perez, E. et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI.
10. Zaheer, M. et al. (2017). *Deep Sets.* NeurIPS.
11. Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS.
12. Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
13. Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning.* Machine Learning 8.
14. Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
15. Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
16. Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS.
17. BehnamGhader, P. et al. (2024). *LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders.* arXiv:2404.05961.
18. Chen, S. F. & Goodman, J. (1999). *An empirical study of smoothing techniques for language modeling.* Computer Speech & Language 13(4).
19. Reams, C. (2009). *Playing Perfect Hangman.* ICGA Journal 32(3).
20. Barbay, J. & Subercaseaux, B. (2020). *The Computational Complexity of Evil Hangman.* arXiv:2003.10000.
21. Meng, F. & Lucas, S. (2024). *Deduction Game Framework and Information Set Entropy Search.* IEEE Conference on Games. arXiv:2407.21178.
22. Zhang, B. & Sennrich, R. (2019). *Root Mean Square Layer Normalization.* NeurIPS.
23. Shazeer, N. (2020). *GLU Variants Improve Transformer.* arXiv:2002.05202.
24. Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864.
25. Huang, G. et al. (2016). *Deep Networks with Stochastic Depth.* ECCV.
26. Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR.
27. Müller, R., Kornblith, S. & Hinton, G. (2019). *When Does Label Smoothing Help?* NeurIPS.
28. van den Oord, A. et al. (2016). *WaveNet: A Generative Model for Raw Audio.* arXiv:1609.03499.
29. Kalchbrenner, N. et al. (2016). *Neural Machine Translation in Linear Time.* arXiv:1610.10099.

### Public implementations surveyed

| Implementation | Approach | Reported |
|---|---|---|
| [ryanyuan42/transformer-hangman-bot][thb] | transformer encoder, 40% masked-letter training, max-pooled presence score | 42% small / 60% large |
| [aghalandar/Hangman_solution][ag] | LSTM over n-gram features | ~51–55% |
| [ZosoV/pretrained_rl_hangman][zos] | pretrained BERT backbone inside a DQN | — |
| [sagarnildas.com][sag] | recurrent PPO | >60% |
| Various BiLSTM + n-gram hybrids | n-gram for the opening, BiLSTM thereafter | ~62% over 1,000 games |

All figures are as reported by their authors, on their own evaluation protocols, and are not
directly comparable with the numbers in this repository — which are measured on a 22,530-word
dictionary held out of `train.txt` and disjoint from it. They are quoted to establish roughly
where the published state of the art sits, not as a leaderboard.

[thb]: https://github.com/ryanyuan42/transformer-hangman-bot
[ag]: https://github.com/aghalandar/Hangman_solution
[zos]: https://github.com/ZosoV/pretrained_rl_hangman
[sag]: https://www.sagarnildas.com/blogs/playing-hangman-with-reinforcement-learning

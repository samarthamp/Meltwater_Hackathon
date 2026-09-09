# Finetuning Report — Qwen3-4B as a Hangman Posterior Teacher

**Meltwater Brand Buzzword Hackathon**

This document covers the optional LLM stage: why an external model is brought in at all, the four
surgical changes made to Qwen3-4B, the throughput accounting that forced distillation to be
*selective* rather than blanket, and what the stage did and did not buy.

Numbers marked **measured** were produced on the DGX Spark (GB10, `sm_121`, 128 GB unified
memory, 273 GB/s effective elementwise bandwidth, ~95 TFLOPS bf16 matmul peak).

---

## 1. Why an external model at all

After the 90/10 validation split, the training dictionary holds **202,770 words**, and the test
dictionary shares **none** of them. A model fitted only on that list knows English orthography
only as far as those words expose it.

The measurement that makes the case concrete:

> **45.1%** of the game states a real policy visits have **no** consistent word in the training
> dictionary at all.

For those states the exact dictionary posterior is the empty set. The interpolated n-gram covers
them, but only with statistics estimated from the same 203k words. Qwen3-4B has read orders of
magnitude more English, so it carries a prior over letter sequences that the dictionary cannot
supply — and the blind states are exactly where an outside prior can still contribute something.

This is transfer learning, not leakage. Section 6 states the compliance position precisely, and
`reports/COMPLIANCE.md` §4 covers it in full.

---

## 2. The four adaptations

### 2.1 Bidirectional attention

**Problem.** Reading `_ o _ _ e r` requires the suffix as much as the prefix. A causal decoder
cannot see it: at the first `_`, everything to the right is masked out.

**Solution.** Replace the causal mask with a padding-only mask — step one of LLM2Vec
(BehnamGhader et al., 2024). In `transformers` 4.57 the `Qwen3Model.forward` contains

```python
if not isinstance(causal_mask_mapping := attention_mask, dict):
    ...  # build a causal mask
```

so passing `attention_mask` as `{"full_attention": mask}` makes the model use our mask verbatim
and skip causal-mask construction entirely. The mask dtype expected by the configured attention
implementation is probed once at load time via `transformers.masking_utils.create_causal_mask`
rather than hard-coded, so this survives an implementation switch.

**Verified, not assumed.** `QwenHangmanTeacher` is checked by changing a board character at
position 6 and confirming the output at position 0 moves. Under a causal mask that delta is
exactly zero. **Measured delta: 3.79** — the attention is genuinely bidirectional. The same
assertion runs as a cell in the notebook.

### 2.2 Exact character tokenisation

**Problem.** BPE would merge board characters unpredictably, so a board position would not
correspond to a token and per-position supervision could not be aligned.

**Solution.** Skip the tokenizer's BPE entirely and map each character straight to its
single-character token id via `convert_tokens_to_ids`. Verified at load time that all 26 letters
plus `_`, `|` and the digits exist as single tokens in Qwen's vocabulary (they do — `a` is 64,
`z` is 89, `_` is 62) and that `decode(encode(board)) == board`.

Character granularity is the right level for a spelling task, which is the argument behind ByT5
(Xue et al., 2021) and CANINE (Clark et al., 2021); here it is also a hard requirement for
alignment.

**Prompt layout**, sized to the batch's actual word length:

```
<board chars> | <absent letters> | <present letters> | <lives digit>
```

Example, for the board `_a___a_` with `q` absent and `a` present, 5 lives left:

```
_a___a_|q|a|5      ->  23 tokens
```

Two facts bound the letter fields exactly, so nothing is lost by shrinking them: a live game
carries at most `MAX_WRONG_GUESSES - 1 = 5` absent letters, and a word of length `L` has at most
`L` distinct letters. **This mattered enormously** — see §3.

### 2.3 Head surgery

`lm_head` is deleted. Two heads are attached to the final hidden states:

| Head | Input | Output |
|---|---|---|
| `set_head` | concat(mean-pool over all tokens, mean-pool over board tokens) | 26 presence logits |
| `pos_head` | per-token hidden state over the board columns | 26 per-position logits |

One forward pass, no generation, no sampling. The heads run in fp32 over a bf16 backbone.

### 2.4 Layer truncation and LoRA

Only the first `keep_layers` of 36 blocks are kept (`backbone.layers = backbone.layers[:k]` plus
`config.num_hidden_layers = k`, which the forward loop honours). Orthographic and morphological
structure is a lower-layer phenomenon, and the teacher has to score millions of states.

LoRA (Hu et al., 2021) at `r = 32`, `alpha = 64`, dropout 0.05, on
`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. Base weights frozen in bf16;
adapters and both heads trained.

QLoRA (Dettmers et al., 2023) is implemented behind `LLMConfig.load_in_4bit` but **off by
default**: `bitsandbytes` is unreliable on aarch64 + `sm_121`, and with 128 GB of unified memory
an 18-layer bf16 backbone (~4.5 GB resident) leaves no reason to quantise.

---

## 3. Throughput accounting — why distillation is selective

This is the part of the design that measurement changed most.

The first working version used a fixed 85-token prompt (board padded to 29, both letter fields
padded to 26 slots each). **Measured: 88 states/s at batch 256** — 0.32M states/hour. Distilling
onto tens of millions of states was simply not on the table, and even the finetune would have
taken most of a day.

Making the layout dynamic (§2.2) took the prompt for a 9-letter board from 85 tokens to 27:

| Configuration | Batch | **Measured** states/s | States/hour |
|---|---:|---:|---:|
| 85-token fixed layout, 18 layers | 256 | 88 | 0.32M |
| Dynamic layout, 18 layers | 256 | 388 | 1.40M |
| Dynamic layout, 18 layers | 1024 | 404 | 1.46M |
| Dynamic layout, 12 layers | 256 | 384 | 1.38M |
| Dynamic layout, 12 layers | 1024 | **558** | **2.01M** |

A 4.6–6.3× improvement from prompt geometry alone, with no change to the model. Note that 12
layers beats 18 only at the larger batch — at batch 256 the two are indistinguishable, which says
the small-batch regime is launch-bound rather than compute-bound.

Forward-only throughput of ~1.5M states/hour sets the inference budget. The *training* budget had
to be measured separately, and the first estimate was wrong: scaling the forward number by the
usual 3× for a backward pass predicted ~490 states/s, but a real training run at 18 layers and
batch 128 came in far below that — under 47 states/s while sharing the device, and an estimated
~130 states/s alone. Backward through a bidirectional 2.25B-parameter stack with no gradient
checkpointing is simply not 3× the forward cost at this sequence length, where the fixed per-kernel
overheads dominate.

The measured training throughput, with the GPU to itself, at 18 layers and batch 192:

| Steps | Instantaneous states/s |
|---|---:|
| 50 → 100 (sharing the GPU with another job) | 37 |
| **100 → 150 (GPU to itself)** | **73** |

73 states/s against a predicted 490 — a **6.7× gap**. Forward-only at the same depth measures 388
states/s, so the backward pass is costing roughly 4× the forward rather than the customary 2×, and
the model is nowhere near any roofline: at 73 states/s the arithmetic works out to ~11 TFLOPS
against ~95 available, and the activation traffic is a small fraction of the 360 GB/s bus. What
is left is fixed per-kernel overhead — 18 blocks × 7 LoRA-adapted projections is 126 adapters,
each adding two small matmuls to the forward and more to the backward, on sequences of only ~27
tokens. There is simply not enough work per kernel to amortise the launches.

**The corrected budget, and what it changed:**

* **Finetune**: ~73 states/s → ~263k samples/hour.
* **Teacher inference** (forward only): ~1.5M states/hour.

At that rate the originally planned 3,300 steps would have taken 145 minutes, so the finetune was
re-sized to 350 MNTP + 1,600 SFT steps under a hard 62-minute wall-clock budget, and the
distillation pass was cut from 2M states to 700k. Two changes were made to the script at the same
time, both of which should have been there from the start: adapters are now checkpointed every
200 steps, and `--max-minutes` stops training cleanly at a deadline and keeps what it has. A
long-running finetune that saves only at the end is one interruption away from being a total loss.

This is recorded in full because "multiply the forward pass by three" is a rule of thumb that
fails badly on short sequences with many small adapters, and because the honest version of this
report is the one that says the teacher was trained on a *reduced* budget and lets the reader
weigh the distillation result accordingly.

Against a training pool of ~38M states, blanket distillation would need ~25 GPU-hours. That is
what forced the design decision:

> **Distil selectively.** Score only the states where the dictionary oracle is blind
> (`n_cand == 0`, 45.1% of the pool) or the word is short (lengths 4–9, where the measured
> baseline loses most of its games). Everywhere else the exact posterior is already the right
> answer, and an LLM opinion could only add noise.

This is not a compromise forced on the science by the hardware — it is also the better design.
The LLM's comparative advantage is precisely on the states where the dictionary has nothing to
say; spending it where the oracle is already exact would be wasted at best.

---

## 4. Training recipe

Two phases, following LLM2Vec's structure adapted to this task.

### Phase 1 — MNTP (masked-character adaptation)

The backbone has only ever run causally. Enabling bidirectional attention changes what every
position sees, so its representations need re-grounding before the presence head means anything.
We train the per-position head alone to predict the true letter at hidden board positions —
ordinary masked language modelling, on characters, over the reachable-state pool.

```
loss = CE(pos_logits, true letter at hidden positions)
```

### Phase 2 — SFT against the Bayes posterior

The presence head is then fitted to the teacher's soft posteriors, with the per-position loss
retained at a lower weight to keep the trunk grounded:

```
loss = BCE(set_logits, bayes_posterior) + 0.3 * CE(pos_logits, true letters)
```

Both phases use AdamW, `lr = 1e-4`, linear warmup, gradient clipping at 1.0, bf16 autocast.

```bash
python scripts/04_finetune_llm.py --keep-layers 18 --mntp-steps 350 --sft-steps 1600 \
    --batch-size 192 --save-every 200 --max-minutes 62
```

### What the run actually did

| Quantity | Value |
|---|---:|
| Wall clock | 62.7 min (stopped by the budget, not by step count) |
| MNTP steps | 350 / 350 |
| SFT steps | 1,183 / 1,600 (budget reached; adapters kept) |
| Samples seen | 294,336 |
| Steady-state throughput | 79 states/s |
| MNTP per-position loss | 3.327 → 2.473 |
| SFT total loss | 1.181 → 1.013 |
| SFT presence-head loss | 0.460 → **0.326** |

The presence-head loss of 0.326 is worth comparing against the compact students, which converge
around 0.33–0.34 on the same objective. On that metric alone the LLM teacher is competitive with a
28.6M-parameter model trained for 19,000 steps, after 1,183 steps and one hour — which is the
transfer working. Whether that translates into *useful* targets for distillation is a separate
question, and the answer is the top-1 hit rate measured in the next stage, reported in section 10
rather than assumed here.

The MNTP phase behaved as the LLM2Vec recipe predicts: loss falls steeply as the backbone adapts
to seeing both directions, then flattens, at which point the presence head has something stable to
sit on.

### Phase 3 — distillation into the students

```bash
python scripts/05_distill.py --adapters artifacts/checkpoints/qwen_teacher_adapters.pt --alpha 0.5
```

The output is a set of record shards whose target is

```
target = alpha * bayes_posterior + (1 - alpha) * qwen_posterior
```

so fine-tuning a student on them *is* the distillation objective (Hinton et al., 2015), with the
blend playing the role of the usual hard/soft mixture. No new loss plumbing is required, which
keeps the student's training path identical across the ablation.

---

## 5. Parameter budget

| Component | Parameters |
|---|---:|
| Qwen3-4B, full | 4.02 B (36 blocks) |
| Backbone kept (18 blocks) + embeddings | ~2.22 B |
| LoRA adapters (`r=32`, 7 projections × 18 blocks) | ~30 M |
| `set_head` (2·2560 → 2560 → 26) | ~13.2 M |
| `pos_head` (2560 → 26) | ~0.07 M |
| **Trained** | **~43 M (1.9% of the resident model)** |

Only the adapters and heads are serialised — a few tens of MB, not 8 GB.

---

## 5b. Did it work? No — and here is the measurement

The distillation stage reports the teacher's top-1 hit rate, but a bare number is not
interpretable: 0.53 is excellent if the alternative is 0.40 and useless if the alternative is
0.66. `tools/teacher_comparison.py` therefore measures every source on **exactly the same
60,000 states** — the oracle-blind and short-word states that distillation selected, i.e. the ones
chosen specifically to favour an outside prior:

| Source | Top-1 hit rate |
|---|---:|
| Bayes teacher (exact dictionary posterior + backoff) | **0.6557** |
| **Qwen3-4B teacher** | **0.5290** |
| Blended target, α = 0.5 | 0.6301 |
| Student `former_a` (28.6 M, trained from scratch) | **0.7436** |

Three things follow, and none of them are what the design predicted.

**The LLM teacher is worse than the dictionary oracle it was brought in to rescue** — 0.529
against 0.656 — on the very states where the oracle was supposed to be blind. Blending the two at
α = 0.5 produces a target *worse than pure Bayes* (0.630 < 0.656), so the distillation objective
is actively degraded relative to doing nothing.

**The student already beats every teacher available to it.** At 0.744 it is 0.09 above the Bayes
oracle that trained it and 0.21 above Qwen. This is the same phenomenon as the win-rate result
(65.6% student against 61.1% teacher) seen at the level of individual decisions rather than whole
games: a network trained on soft posteriors over millions of states generalises better than the
posterior itself, because the posterior is computed from a finite dictionary and the network is
not.

**So distillation should hurt the strong model and may still help a weak one.** That is exactly
what happened. `scripts/06_distill_finetune.py` keeps its output only if it beats the input
checkpoint on held-out words, and the two members went opposite ways:

| Member | Before | After 2,000 distillation steps | Outcome |
|---|---:|---:|---|
| `former_a` (28.6 M) | 65.76 ±0.53 | 65.60 ±0.53 | **regressed — rejected** |
| `bigru_c` (9.5 M) | 52.21 ±0.56 | 53.44 ±0.56 | **+1.23 — kept** |

The split follows directly from the table above. The blended target scores 0.630, which is *below*
`former_a`'s own 0.744 and above whatever `bigru_c` manages — so training the strong model toward
that target drags it down, and training the weak one toward it pulls it up. A teacher is only
worth distilling from if it is better than the student, and "better" is a per-student question,
not a property of the teacher.

**A confound, and the control for it.** `bigru_c` was the less converged member (13,000 training
steps against `former_a`'s 19,000), and its distillation pass also gave it 2,000 *additional*
steps on a pool that is 98.6% non-distillation records. Extra training alone could account for
+1.23. The control is the identical run with the distillation shards removed
(`--no-distill --tag control`); its result is reported in `reports/MODEL_REPORT.md` section 10.
Until that number is in, the +1.23 should be read as an upper bound on what distillation
contributed, not as its measured effect.

### Why the teacher underperformed

Not, as far as the evidence goes, because the idea is wrong — because the budget was.

| | Qwen3-4B teacher | Student `former_a` |
|---|---:|---:|
| Task-specific samples seen | 294,336 | ~78,000,000 |
| Depth used | 18 of 36 blocks | 8 of 8 |
| Trainable parameters | 46.3 M (2.05%) | 28.6 M (100%) |
| Wall clock | 62.7 min | 132.8 min |

The student saw **265× more task-specific data**. Whatever prior over English orthography Qwen
carries, one hour of adaptation on 294k states was not enough to convert it into per-letter
posteriors competitive with either an exact combinatorial oracle or a small network that has
effectively read the training dictionary hundreds of times. The throughput analysis in section 3
explains why that hour bought so few samples: 73 states/s, 6.7× below the naive estimate, because
126 LoRA adapters on 27-token sequences leave nothing to amortise kernel launches against.

The honest reading is that this is a **compute-bound negative result, not a refutation of transfer
learning for hangman**. What would be needed to test the idea properly:

* an order of magnitude more finetuning — 3–5 M samples rather than 294 k, which at the measured
  throughput is 11–19 GPU-hours rather than one;
* gradient checkpointing and adapter fusion, or simply LoRA on fewer modules, to raise the
  step rate;
* batching states of *mixed* lengths into longer packed sequences, so each kernel does enough
  work to matter.

None of those fit the window. The finding is reported as it stands.

## 6. Compliance position

Three properties, each checkable in the code:

1. **The submitted model never runs Qwen.** `scripts/09_make_submission.py` does not import
   `hangman.models.llm_teacher`. Inference is the compact character-transformer ensemble alone.
2. **Qwen never proposes vocabulary.** It is only queried about boards constructed from
   `train.txt` words, and only for a 26-dimensional per-letter posterior. There is no generation
   and no sampling, so there is no mechanism by which a test word could be introduced.
3. **A no-LLM model is trained and reported alongside.** The `*_final.pt` checkpoints are trained
   on `train.txt` alone; distillation produces separate `*_distilled.pt` checkpoints. Both appear
   in the ablation table, so the contribution of external knowledge is quantified rather than
   assumed.

   As section 5b shows, that contribution measured **negative**, so the shipped model is the
   `train.txt`-only one. The submission therefore has no pretrained weights anywhere in its
   provenance — not by policy, but because the measurement said so.

---

## 7. What went wrong, and what it cost

Recorded because it shaped the design:

* **The 85-token prompt.** The first layout padded every field to its global maximum, which put
  two thirds of the compute into padding and made the whole stage look infeasible at 88 states/s.
  Sizing the layout to the batch fixed it without touching the model. The lesson generalises: on
  a bandwidth-bound accelerator, sequence geometry is a first-order cost, not a detail.
* **`torch.bool` vs float masks.** The mask dtype expected by `scaled_dot_product_attention`
  differs by attention implementation. Rather than hard-code it, the loader probes
  `create_causal_mask` once and matches whatever it returns.
* **Tied embeddings.** Qwen3-4B declares `tie_word_embeddings: true` but also ships a separate
  `lm_head.weight`, so the on-disk size double-counts the embedding matrix. Deleting `lm_head`
  after loading is safe and reclaims the duplicate.
* **Running it alongside student training.** The finetune is kernel-launch-bound at small batch
  (388 states/s at batch 256 against 404 at batch 1024), which made it look like a good candidate
  to overlap with the bandwidth-bound character transformer. It was not: the transformer dropped
  from 3.35 to 2.19 it/s and the finetune failed to complete 100 steps in four and a half minutes.
  On a machine whose bottleneck is one shared memory bus, spare memory is not spare throughput.
  Both jobs went back on the sequential schedule.

---

## 8. References

* BehnamGhader, P. et al. (2024). *LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders.* arXiv:2404.05961.
* Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
* Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS.
* Hinton, G., Vinyals, O. & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531.
* Xue, L. et al. (2021). *ByT5: Towards a token-free future with pre-trained byte-to-byte models.* arXiv:2105.13626.
* Clark, J. H. et al. (2021). *CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation.* TACL.

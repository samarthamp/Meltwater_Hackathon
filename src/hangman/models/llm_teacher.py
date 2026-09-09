"""Qwen3-4B adapted into a bidirectional hangman posterior teacher.

Why bring in a pretrained LLM at all
------------------------------------
``train.txt`` holds 202,770 words after the validation split, and the test
dictionary shares none of them.  A model fitted only on that list can learn
English orthography only as far as those words expose it.  Qwen3-4B has read
orders of magnitude more English, so it carries a prior over letter sequences
that the training dictionary cannot supply.  The measurement that motivates
this: **44% of the game states a real policy visits have no consistent word in
the training dictionary at all**, and those are exactly the states where an
outside prior can still say something useful.

What is done to it
------------------
1. **Bidirectional attention.**  Reading ``_ o _ _ e r`` needs the suffix as
   much as the prefix, and a causal decoder cannot see it.  We replace the
   causal mask with a padding-only mask, which is step one of LLM2Vec
   (BehnamGhader et al., 2024, *LLM2Vec: Large Language Models Are Secretly
   Powerful Text Encoders*).  In transformers >= 4.5x this is done cleanly by
   passing ``attention_mask`` as a ``{"full_attention": mask}`` dict, which the
   model uses verbatim instead of building a causal one.
2. **Exact character tokenisation.**  Board characters are mapped straight to
   the tokenizer's single-character token ids rather than run through BPE, so
   one board position is exactly one token and per-position supervision lines
   up.  Character granularity is the right level for a spelling task - the
   argument behind ByT5 (Xue et al., 2021) and CANINE (Clark et al., 2021).
3. **Head surgery.**  ``lm_head`` is dropped; a presence head and a
   per-position head are attached to the final hidden states.  One forward
   pass, no generation, no sampling.
4. **Layer truncation.**  Only the first ``keep_layers`` blocks are kept.
   Orthographic and morphological structure is a lower-layer phenomenon, and
   the teacher has to score millions of states, so depth is bought only where
   it pays.  The trade-off is measured in reports/FINETUNING_REPORT.md.
5. **LoRA.**  Low-rank adapters on attention and MLP projections
   (Hu et al., 2021), heads trained in full, bf16 throughout.  QLoRA
   (Dettmers et al., 2023) is available but off by default: ``bitsandbytes`` is
   unreliable on aarch64 + sm_121 and 128 GB of unified memory makes it
   unnecessary.

Compliance note
---------------
The submitted model never runs Qwen at inference.  Qwen only shapes the
student's training targets, and it is only ever asked about boards derived from
``train.txt`` words - it is never used to propose vocabulary, so it cannot
import test-set words.  A ``--no-llm`` student is trained alongside and both
scores are reported; see reports/COMPLIANCE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ..config import LLMConfig, MAX_WRONG_GUESSES
from ..ngram import HOLE
from ..simulator import PAD_POS

ALPHABET = "abcdefghijklmnopqrstuvwxyz"

#: Prompt layout, fixed so that board positions are at known token offsets:
#:   [board chars] | [absent letters] | [present letters] | [lives digit]
SEP = "|"
MASK_CHAR = "_"


@dataclass
class PromptLayout:
    """Token offsets of each field for a given board width.

    The layout is sized to the batch's actual word length rather than to the
    global maximum.  That matters: attention and MLP cost are linear in
    sequence length, and padding a 9-character board out to 29 characters plus
    two 26-slot letter fields wastes roughly two thirds of every forward pass.

    Two facts bound the letter fields exactly, so nothing is lost by shrinking
    them:  a live game carries at most ``MAX_WRONG_GUESSES - 1`` absent letters
    (the sixth ends it), and a word of length ``L`` has at most ``L`` distinct
    letters.
    """

    board_width: int

    @property
    def absent_slots(self) -> int:
        return MAX_WRONG_GUESSES - 1

    @property
    def present_slots(self) -> int:
        return min(self.board_width, 26)

    @property
    def total_len(self) -> int:
        return self.board_width + 1 + self.absent_slots + 1 + self.present_slots + 1 + 1

    @staticmethod
    def for_length(length: int) -> "PromptLayout":
        return PromptLayout(board_width=length)


class CharCodec:
    """Maps hangman characters to single-token ids in the LLM's vocabulary."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.letter_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids(c) for c in ALPHABET], dtype=torch.long
        )
        self.mask_id = tokenizer.convert_tokens_to_ids(MASK_CHAR)
        self.sep_id = tokenizer.convert_tokens_to_ids(SEP)
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.digit_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)], dtype=torch.long
        )
        bad = [
            name
            for name, value in (
                ("mask", self.mask_id), ("sep", self.sep_id), ("pad", self.pad_id)
            )
            if value is None
        ]
        if bad or bool((self.letter_ids < 0).any()):
            raise RuntimeError(f"tokenizer lacks required single-character tokens: {bad}")

    def _pack(
        self, flags: torch.Tensor, n_slots: int, letters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pack the set letters of ``flags`` into ``n_slots`` columns."""
        batch = flags.shape[0]
        device = flags.device
        slot = flags.cumsum(dim=1) - 1
        valid = flags & (slot < n_slots)
        # One extra column absorbs everything that does not fit, then is dropped.
        scratch = torch.full((batch, n_slots + 1), self.pad_id, dtype=torch.long, device=device)
        index = torch.where(valid, slot, torch.full_like(slot, n_slots))
        source = torch.where(
            valid, letters.unsqueeze(0).expand(batch, 26), torch.full_like(slot, self.pad_id)
        )
        scratch.scatter_(1, index, source)
        filled = valid.sum(dim=1, keepdim=True)
        occupied = torch.arange(n_slots, device=device).unsqueeze(0) < filled
        return scratch[:, :n_slots], occupied

    def encode(
        self,
        board: torch.Tensor,
        lengths: torch.Tensor,
        present: torch.Tensor,
        absent: torch.Tensor,
        layout: PromptLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(input_ids, attention_mask)`` for a batch of boards.

        Columns ``0..board_width-1`` are board characters, one token each, so
        the per-position head reads off exactly those columns.
        """
        device = board.device
        batch = board.shape[0]
        letters = self.letter_ids.to(device)
        digits = self.digit_ids.to(device)
        width = layout.board_width

        ids = torch.full((batch, layout.total_len), self.pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((batch, layout.total_len), dtype=torch.bool, device=device)

        chunk = board[:, :width]
        if chunk.shape[1] < width:
            pad = torch.full((batch, width - chunk.shape[1]), PAD_POS, dtype=chunk.dtype, device=device)
            chunk = torch.cat([chunk, pad], dim=1)
        board_ids = torch.where(
            chunk == HOLE, torch.full_like(chunk, self.mask_id), letters[chunk.clamp(0, 25)]
        )
        real = chunk != PAD_POS
        ids[:, :width] = torch.where(real, board_ids, torch.full_like(board_ids, self.pad_id))
        attn[:, :width] = real

        cursor = width
        ids[:, cursor] = self.sep_id
        attn[:, cursor] = True
        cursor += 1

        for field, n_slots in ((absent, layout.absent_slots), (present, layout.present_slots)):
            packed, occupied = self._pack(field, n_slots, letters)
            ids[:, cursor : cursor + n_slots] = packed
            attn[:, cursor : cursor + n_slots] = occupied
            cursor += n_slots
            ids[:, cursor] = self.sep_id
            attn[:, cursor] = True
            cursor += 1

        lives = (MAX_WRONG_GUESSES - absent.sum(dim=1)).clamp(0, 9)
        ids[:, cursor] = digits[lives]
        attn[:, cursor] = True
        return ids, attn


class QwenHangmanTeacher(nn.Module):
    """Bidirectional, layer-truncated Qwen3 with hangman heads."""

    def __init__(self, cfg: LLMConfig, max_len: int = 29) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        self.codec = CharCodec(self.tokenizer)

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        backbone = base.model
        if cfg.keep_layers and cfg.keep_layers < len(backbone.layers):
            backbone.layers = backbone.layers[: cfg.keep_layers]
            backbone.config.num_hidden_layers = cfg.keep_layers
        del base.lm_head
        self.backbone = backbone
        self.hidden = backbone.config.hidden_size
        self._mask_dtype: torch.dtype | None = None

        self.set_head = nn.Sequential(
            nn.Linear(2 * self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, 26),
        ).to(torch.float32)
        self.pos_head = nn.Linear(self.hidden, 26).to(torch.float32)

    # -- bidirectionalisation -------------------------------------------------

    def _attention_mask(self, pad_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """A padding-only mask, replacing the causal one (LLM2Vec step 1)."""
        batch, seq = pad_mask.shape
        if self._mask_dtype is None:
            self._mask_dtype = self._probe_mask_dtype(pad_mask)
        keep = pad_mask[:, None, None, :].expand(batch, 1, seq, seq)
        if self._mask_dtype == torch.bool:
            return {"full_attention": keep}
        dtype = self._mask_dtype
        mask = torch.zeros((batch, 1, seq, seq), dtype=dtype, device=pad_mask.device)
        return {"full_attention": mask.masked_fill(~keep, torch.finfo(dtype).min)}

    def _probe_mask_dtype(self, pad_mask: torch.Tensor) -> torch.dtype:
        """Ask transformers what mask type this attention implementation wants."""
        from transformers.masking_utils import create_causal_mask

        dummy = torch.zeros(
            (pad_mask.shape[0], pad_mask.shape[1], self.hidden),
            dtype=torch.bfloat16,
            device=pad_mask.device,
        )
        built = create_causal_mask(
            config=self.backbone.config,
            input_embeds=dummy,
            attention_mask=pad_mask.to(torch.long),
            cache_position=torch.arange(pad_mask.shape[1], device=pad_mask.device),
            past_key_values=None,
            position_ids=None,
        )
        return torch.bool if built is None else built.dtype

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        board: torch.Tensor,
        lengths: torch.Tensor,
        present: torch.Tensor,
        absent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        layout = PromptLayout.for_length(int(lengths.max().item()))
        ids, pad_mask = self.codec.encode(board, lengths, present, absent, layout)
        hidden = self.backbone(
            input_ids=ids,
            attention_mask=self._attention_mask(pad_mask),
            use_cache=False,
        ).last_hidden_state

        weights = pad_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        board_slice = hidden[:, : layout.board_width]
        board_weights = pad_mask[:, : layout.board_width].unsqueeze(-1).to(hidden.dtype)
        board_pool = (board_slice * board_weights).sum(dim=1) / board_weights.sum(dim=1).clamp_min(1.0)

        summary = torch.cat([pooled, board_pool], dim=-1).float()
        return {
            "set_logits": self.set_head(summary),
            "pos_logits": self.pos_head(board_slice.float()),
        }

    @torch.no_grad()
    def presence(self, board, lengths, present, absent) -> torch.Tensor:
        """``(B, 26)`` presence probabilities, guessed letters zeroed."""
        out = self(board, lengths, present, absent)
        probs = torch.sigmoid(out["set_logits"])
        return probs.masked_fill(present | absent, 0.0)

    # -- parameter-efficient finetuning ----------------------------------------

    def apply_lora(self) -> None:
        """Wrap the backbone in LoRA adapters and freeze the base weights."""
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=list(self.cfg.lora_targets),
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.backbone = get_peft_model(self.backbone, lora)

    def trainable_parameters(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def save_adapters(self, path: str | Path) -> Path:
        """Persist only the adapters and heads - a few tens of MB, not 8 GB."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "set_head": self.set_head.state_dict(),
            "pos_head": self.pos_head.state_dict(),
            "cfg": self.cfg,
        }
        if hasattr(self.backbone, "peft_config"):
            from peft import get_peft_model_state_dict

            state["lora"] = get_peft_model_state_dict(self.backbone)
        torch.save(state, path)
        return path

    def load_adapters(self, path: str | Path) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.set_head.load_state_dict(blob["set_head"])
        self.pos_head.load_state_dict(blob["pos_head"])
        if "lora" in blob:
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(self.backbone, blob["lora"])

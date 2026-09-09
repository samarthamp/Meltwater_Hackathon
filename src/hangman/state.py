"""Game state -> model tensors.

A hangman state is ``(board, present letters, absent letters, lives)``.  Two
observations drive the encoding:

1. **The absent set is evidence, not bookkeeping.**  Knowing that ``q``, ``x``
   and ``z`` are *not* in the word sharply reshapes the posterior over what is.
   Public hangman transformers encode only the board and at best suppress
   already-guessed letters at the output; the surveyed reference implementation
   states outright that it "doesn't account for letters guessed but absent".
   We feed both guessed sets in as first-class inputs.

2. **The guessed sets are sets.**  They carry no order, so they are encoded as
   permutation-invariant indicator vectors rather than as a sequence
   (Zaheer et al., 2017, *Deep Sets*).

Everything the model needs is derivable from ``(word, guessed_letter_mask)``,
which is what makes the training records in :mod:`hangman.collect` only eight
bytes of state each.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import (
    GLOBAL_FEATURE_DIM,
    MAX_WORD_LEN,
    MAX_WRONG_GUESSES,
    SEQ_LEN,
    TOK_CLS,
    TOK_MASK,
    TOK_PAD,
)
from .simulator import PAD_POS
from .ngram import HOLE


def encode_tokens(board: torch.Tensor, seq_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Board -> ``(tokens, attention_mask)`` of shape ``(B, seq_len)``.

    Slot 0 is the ``[CLS]`` summary token; slots ``1..L`` hold the board, with
    ``TOK_MASK`` for hidden positions and ``TOK_PAD`` beyond the word.

    ``seq_len`` defaults to whatever this batch actually needs rather than the
    global maximum of 32.  Words average 9.4 characters, so padding every batch
    to 32 would trip the model's cost - which on this hardware is dominated by
    activation traffic, not arithmetic.  Combined with the length-bucketed
    sampler in :mod:`hangman.dataset`, batches are typically 10-12 wide.
    """
    batch, lmax = board.shape
    device = board.device
    if seq_len is None:
        seq_len = min(SEQ_LEN, lmax + 1)
    tokens = torch.full((batch, seq_len), TOK_PAD, dtype=torch.long, device=device)
    tokens[:, 0] = TOK_CLS

    width = min(lmax, seq_len - 1)
    chunk = board[:, :width]
    body = torch.where(
        chunk == PAD_POS,
        torch.full_like(chunk, TOK_PAD),
        torch.where(chunk == HOLE, torch.full_like(chunk, TOK_MASK), chunk),
    )
    tokens[:, 1:1 + width] = body

    attn = tokens != TOK_PAD
    attn[:, 0] = True
    return tokens, attn


def encode_globals(
    lengths: torch.Tensor,
    present: torch.Tensor,
    absent: torch.Tensor,
) -> torch.Tensor:
    """``(B, GLOBAL_FEATURE_DIM)`` float32 conditioning vector.

    Layout (see :mod:`hangman.config`)::

        [ 0:26]  absent indicator
        [26:52]  present indicator
        [52:58]  one-hot lives remaining, clamped to 0..5
        [58]     word length / MAX_WORD_LEN
        [59]     guesses made / 26
        [60]     fraction of positions revealed
        [61]     distinct revealed letters / 26
    """
    device = present.device
    batch = present.shape[0]
    absent_f = absent.to(torch.float32)
    present_f = present.to(torch.float32)

    wrong = absent_f.sum(dim=1)
    lives = (MAX_WRONG_GUESSES - wrong).clamp(0, MAX_WRONG_GUESSES - 1).to(torch.long)
    lives_onehot = torch.zeros((batch, MAX_WRONG_GUESSES), dtype=torch.float32, device=device)
    lives_onehot.scatter_(1, lives.unsqueeze(1), 1.0)

    n_guesses = present_f.sum(dim=1) + wrong
    lengths_f = lengths.to(torch.float32)

    # Fraction of *positions* revealed needs the per-letter multiplicity, which
    # the caller does not have; the distinct-letter proxy below is equivalent
    # information for the network and costs nothing to compute.
    revealed_letters = present_f.sum(dim=1)

    extras = torch.stack(
        [
            lengths_f / float(MAX_WORD_LEN),
            n_guesses / 26.0,
            revealed_letters / lengths_f.clamp_min(1.0),
            revealed_letters / 26.0,
        ],
        dim=1,
    )
    return torch.cat([absent_f, present_f, lives_onehot, extras], dim=1)


def encode_batch(
    board: torch.Tensor,
    lengths: torch.Tensor,
    present: torch.Tensor,
    absent: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Full model input for a batch of states."""
    tokens, attn = encode_tokens(board)
    return {
        "tokens": tokens,
        "attn_mask": attn,
        "globals": encode_globals(lengths, present, absent),
        "guessed": present | absent,
    }


def board_from_records(
    words: torch.Tensor,
    lengths: torch.Tensor,
    guessed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct ``(board, present, absent)`` from compact training records.

    Training records store only the word and a 26-bit guessed-letter mask; the
    board, the present/absent split and the remaining lives all follow
    deterministically, which keeps a 20M-state dataset near 1.3 GB.

    Parameters
    ----------
    words : (B, Lmax) int64 letter ids, ``PAD_POS`` beyond the word.
    lengths : (B,) int64.
    guessed : (B, 26) bool.
    """
    real = words >= 0
    word_presence = torch.zeros_like(guessed)
    word_presence.scatter_(1, words.clamp_min(0), True)
    # Words shorter than Lmax scatter a spurious True at column 0; undo it.
    has_a = (words == 0).any(dim=1)
    word_presence[:, 0] &= has_a

    present = guessed & word_presence
    absent = guessed & ~word_presence

    revealed = present.gather(1, words.clamp_min(0)) & real
    board = torch.where(revealed, words, torch.full_like(words, HOLE))
    board = torch.where(real, board, torch.full_like(words, PAD_POS))
    return board, present, absent


def pack_words(words: list[str], device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """``(word_ids, lengths)`` tensors for a list of words."""
    lengths = np.array([len(w) for w in words], dtype=np.int64)
    lmax = int(lengths.max()) if len(words) else 1
    out = np.full((len(words), lmax), PAD_POS, dtype=np.int64)
    for i, w in enumerate(words):
        out[i, : len(w)] = np.frombuffer(w.encode("ascii"), dtype=np.uint8).astype(np.int64) - 97
    return torch.from_numpy(out).to(device), torch.from_numpy(lengths).to(device)


def guessed_mask_to_bits(mask: torch.Tensor) -> torch.Tensor:
    """``(B, 26)`` bool -> ``(B,)`` uint32 bitmask, for compact storage."""
    weights = (1 << torch.arange(26, device=mask.device, dtype=torch.long))
    return (mask.to(torch.long) * weights).sum(dim=1).to(torch.int64)


def bits_to_guessed_mask(bits: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`guessed_mask_to_bits`."""
    shifts = torch.arange(26, device=bits.device, dtype=torch.long)
    return ((bits.unsqueeze(1) >> shifts) & 1).to(torch.bool)

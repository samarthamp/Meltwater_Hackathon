"""Neural Hangman solver.

Modules:

    config       - dataclass configuration for every experiment
    data         - dictionary loading and the disjoint fit/validation split
    ngram        - interpolated character n-gram model (teacher backoff)
    teacher      - Bayes posterior oracle that labels training states
    state        - game state -> model tensors
    simulator    - batched lock-step game engine implementing the rules
    collect      - off-policy and DAgger on-policy state collection
    dataset      - record shards -> model batches
    models       - HangmanFormer, Bi-GRU, dilated CNN, Qwen3-4B teacher
    losses       - multi-task objective
    train        - supervised trainer
    selfdistill  - student-as-teacher target blending
    rl           - self-play policy refinement
    policy       - guess-selection policies (greedy, value-guided, exact search)
    calibrate    - probability calibration and ensemble weighting
    evaluate     - win-rate / efficiency metrics
    pipeline     - stage functions for the end-to-end pipeline
    submit       - test.txt -> submission.csv with validators
"""

__version__ = "1.0.0"

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
N_LETTERS = 26
MAX_WRONG_GUESSES = 6

__all__ = ["ALPHABET", "N_LETTERS", "MAX_WRONG_GUESSES", "__version__"]

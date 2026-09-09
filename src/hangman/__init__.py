"""Neural Hangman solver for the Meltwater Brand Buzzword Hackathon.

The package is organised so that the submission notebook is a thin driver over
importable, unit-testable modules:

    config      - dataclass configuration, the single source of truth
    data        - dictionary loading and the disjoint fit/validation split
    teacher     - three-tier Bayes/n-gram posterior oracle (label generator)
    state       - game state -> model tensors
    simulator   - batched lock-step game engine implementing competition rules
    collect     - off-policy and DAgger on-policy state collection
    models      - HangmanFormer, Bi-GRU, dilated CNN, Qwen3-4B teacher
    losses      - multi-task objective
    policy      - guess-selection policies (greedy, value-guided, exact search)
    ensemble    - calibrated multi-model combination
    evaluate    - win-rate / efficiency metrics and ablation harness
    submit      - test.txt -> submission.csv with validators
"""

__version__ = "1.0.0"

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
N_LETTERS = 26
MAX_WRONG_GUESSES = 6

__all__ = ["ALPHABET", "N_LETTERS", "MAX_WRONG_GUESSES", "__version__"]

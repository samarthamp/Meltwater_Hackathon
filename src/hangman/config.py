"""Configuration objects.

Every tunable lives in a JSON-serialisable dataclass, so an experiment is
described by a single object that is saved next to each checkpoint.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Data"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
DATASET_DIR = ARTIFACT_DIR / "datasets"
LOG_DIR = ARTIFACT_DIR / "logs"
METRIC_DIR = ARTIFACT_DIR / "metrics"
REPORT_DIR = REPO_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TRAIN_TXT = DATA_DIR / "train.txt"
TEST_TXT = DATA_DIR / "test.txt"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

#: Six wrong guesses is a loss; five is survivable (see simulator.py).
MAX_WRONG_GUESSES = 6

# Longest word in the provided files is 29 characters; SEQ_LEN adds a [CLS]
# slot and rounds up to 32.
MAX_WORD_LEN = 29
SEQ_LEN = 32

# Token vocabulary for the character axis.
TOK_PAD, TOK_MASK, TOK_CLS = 26, 27, 28
VOCAB_SIZE = 29

# Global feature vector layout (see state.encode_states).
#   [0:26]   absent-letter indicator
#   [26:52]  present-letter indicator
#   [52:58]  one-hot lives remaining (0..5)
#   [58]     word length / MAX_WORD_LEN
#   [59]     number of guesses so far / 26
#   [60]     distinct present letters / word length (revealed-fraction proxy)
#   [61]     number of distinct revealed letters / 26
GLOBAL_FEATURE_DIM = 62


@dataclass
class DataConfig:
    """Dictionary loading and the disjoint fit/validation split."""

    train_path: Path = TRAIN_TXT
    test_path: Path = TEST_TXT
    #: Fraction of train.txt held out as a disjoint validation dictionary,
    #: mirroring the zero word overlap between train and test.
    val_fraction: float = 0.10
    split_seed: int = 20260903
    #: When True the fit dictionary is all of train.txt (no validation split).
    #: Used only for the final submission model, never during model selection.
    use_full_dictionary: bool = False


@dataclass
class TeacherConfig:
    """Three-tier Bayes/n-gram posterior oracle."""

    #: Highest n-gram order (inclusive).  Orders 2..max are interpolated.
    max_ngram: int = 7
    #: Weight of order-n evidence in the interpolation, w(n) = n ** ngram_power.
    ngram_power: float = 3.0
    #: Jelinek-Mercer style mixing between the exact candidate posterior and the
    #: smoothed n-gram posterior: lambda = n_candidates / (n_candidates + kappa).
    backoff_kappa: float = 300.0
    #: How many rungs of the constraint-relaxation ladder to use (1..3).
    n_exact_tiers: int = 3
    #: 'linear' mixes presence marginals arithmetically; 'loglinear' pools them
    #: geometrically, which is sharper when the sources agree.
    blend: str = "linear"
    #: Laplace smoothing added to candidate presence counts.
    presence_prior: float = 0.5
    #: Leave-one-out: remove the played word from its own candidate set so the
    #: teacher faces the same unseen-word condition as test time.
    leave_one_out: bool = True


@dataclass
class ModelConfig:
    """HangmanFormer architecture."""

    name: str = "hangmanformer_m"
    arch: str = "transformer"          # transformer | bigru | dilated_cnn
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1408                   # SwiGLU: ~2.75 x d_model
    dropout: float = 0.10
    stochastic_depth: float = 0.05
    use_rope: bool = True
    use_abs_pos: bool = True
    use_film: bool = True
    #: Auxiliary heads.
    use_pos_head: bool = True
    use_count_head: bool = True
    use_value_head: bool = True
    count_buckets: int = 4             # counts {0, 1, 2, >=3}
    #: Bi-GRU / CNN specific.
    rnn_layers: int = 3
    cnn_layers: int = 8
    cnn_kernel: int = 3


@dataclass
class LossConfig:
    """Multi-task objective weights."""

    w_set_soft: float = 1.0            # BCE against the Bayes posterior
    w_set_hard: float = 0.25           # BCE against the ground-truth 0/1 label
    w_pos: float = 0.40                # per-position KL
    w_count: float = 0.15              # count-bucket CE
    w_value: float = 0.10              # win-probability BCE
    w_llm: float = 0.0                 # KL to the Qwen teacher (set in distill)
    distill_temperature: float = 2.0
    label_smoothing: float = 0.01


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    batch_size: int = 4096
    epochs: int = 1
    #: Exponential moving average of weights; 0 disables.
    ema_decay: float = 0.999
    amp_dtype: str = "bfloat16"
    #: torch.compile fuses the elementwise work (RMSNorm, FiLM, SwiGLU gating,
    #: residuals) that otherwise dominates step time.
    compile: bool = True


@dataclass
class CollectConfig:
    """State collection for the off-policy warm start and DAgger rounds."""

    #: Number of (word, state) samples drawn in the off-policy stage.
    offpolicy_states: int = 20_000_000
    #: On-policy states collected per DAgger round.
    onpolicy_states_per_round: int = 8_000_000
    dagger_rounds: int = 4
    #: Fraction of each training batch drawn from the off-policy pool.
    offpolicy_mix: float = 0.30
    #: Oversampling weight for words in ``short_word_range``, where most games
    #: are lost.
    short_word_boost: float = 1.6
    short_word_range: tuple[int, int] = (4, 9)
    n_workers: int = 18
    seed: int = 12345


@dataclass
class SelfDistillConfig:
    """Using a trained model as the teacher for the next generation.

    See :mod:`hangman.selfdistill`.
    """

    enabled: bool = False
    #: Gate on candidate count: ``lambda = n / (n + kappa)`` is the weight on the
    #: Bayes posterior.  0 = pure Bayes, inf = pure student.  Tuned with
    #: ``tools/blend_sweep.py``.
    kappa_sd: float = 50.0
    #: Checkpoint of the previous generation, used as the teacher.
    teacher_ckpt: str = ""
    #: Weight of the noisy-OR pooled positional head in the teacher's inference.
    teacher_noisy_or_weight: float = 0.4


@dataclass
class PolicyConfig:
    """Guess-selection at inference time."""

    mode: str = "auto"                 # greedy | value | exact | auto
    #: Value-guided one-ply lookahead is used when lives_left <= this.
    value_lookahead_max_lives: int = 2
    value_topk: int = 5
    #: Exact information-set search runs when the consistent candidate set is
    #: non-empty and no larger than this.
    exact_max_candidates: int = 120
    #: Logit-space ensemble/source blending, fitted on the validation dictionary.
    calibrate: bool = True


@dataclass
class LLMConfig:
    """Qwen3-4B teacher."""

    model_path: str = "/home/samarthamp/ai-models/qwen3-4b"
    #: Keep only the first N transformer layers (0 = keep all); see the
    #: layer-truncation study in reports/FINETUNING_REPORT.md.
    keep_layers: int = 18
    bidirectional: bool = True         # LLM2Vec-style causal-mask removal
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    load_in_4bit: bool = False         # QLoRA via bitsandbytes
    mntp_steps: int = 4000             # masked-character adaptation
    sft_steps: int = 12000             # fit to Bayes posteriors
    batch_size: int = 64
    lr: float = 1e-4
    max_seq_len: int = 96
    #: Cap on states scored by the teacher during distillation.
    max_distill_states: int = 3_000_000


@dataclass
class RLConfig:
    """Self-play policy refinement."""

    enabled: bool = True
    algo: str = "ppo"                  # ppo | reinforce
    steps: int = 4000
    games_per_step: int = 2048
    lr: float = 2e-5
    clip_eps: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    #: KL trust region back to the supervised policy, preventing collapse.
    kl_coef: float = 0.05
    #: Reward = win - efficiency_penalty * wrong_guesses, matching the
    #: scoring tiebreak (Score - E / 1e8).
    efficiency_penalty: float = 0.001


@dataclass
class ExperimentConfig:
    """Top-level configuration for one training run."""

    run_name: str = "hangmanformer_m_seed0"
    seed: int = 0
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    selfdistill: SelfDistillConfig = field(default_factory=SelfDistillConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @staticmethod
    def load(path: str | Path) -> "ExperimentConfig":
        return ExperimentConfig.load_from_dict(json.loads(Path(path).read_text()))

    @staticmethod
    def load_from_dict(raw: dict[str, Any]) -> "ExperimentConfig":
        return _from_dict(ExperimentConfig, raw)


def _from_dict(cls: type, raw: dict[str, Any]) -> Any:
    """Rebuild a nested dataclass from a plain dict.

    Field annotations are strings under ``from __future__ import annotations``,
    so nested config classes are recovered by calling each ``default_factory``.
    """
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        nested = None
        if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            candidate = f.default_factory()               # type: ignore[misc]
            if dataclasses.is_dataclass(candidate):
                nested = type(candidate)
        if nested is not None and isinstance(value, dict):
            kwargs[f.name] = _from_dict(nested, value)
        elif f.name.endswith("_path") or f.name in {"train_path", "test_path"}:
            kwargs[f.name] = Path(value)
        elif isinstance(value, list) and isinstance(
            f.default if f.default is not dataclasses.MISSING else None, tuple
        ):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def ensure_dirs() -> None:
    """Create every artifact directory the pipeline writes to."""
    for d in (ARTIFACT_DIR, CHECKPOINT_DIR, DATASET_DIR, LOG_DIR, METRIC_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)

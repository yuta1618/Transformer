from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 128
    vocab_size: int = 65
    dropout: float = 0.2
    learning_rate: float = 3e-4
    weight_decay: float = 0.1

    batch_size: int = 32
    max_steps: int = 400
    eval_interval: int = 50
    save_interval: int = 200
    train_dataset_size: int = 500_000
    eval_steps: int = 10
    grad_clip: float = 1.0
    warmup_steps: int = 50
    min_lr: float = 1e-5
    sample_length: int = 64
    sample_temperature: float = 1.0
    sample_top_k: int = 10
    checkpoint_dir: str = "checkpoints"
    resume_checkpoint: str | None = None
    use_fp16: bool = False
    device: str = "auto"
    seed: int = 42

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert 0 <= self.min_lr <= self.learning_rate, "min_lr must be between 0 and learning_rate"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.max_steps > 0, "max_steps must be positive"
        assert self.eval_interval > 0, "eval_interval must be positive"

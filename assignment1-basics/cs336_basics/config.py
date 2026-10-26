from dataclasses import dataclass
import torch

@dataclass
class Config:
    # hyperparameters for model
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000
    device: torch.device | None = "cuda:0"
    dtype: torch.dtype | None = None

    # for ablation experiments only
    use_rope: bool = True
    use_rms_norm: bool = True
    use_pre_norm: bool = True
    use_swiglu: bool = True

    # hyperparameters for training
    training_data_path: str = "cs336_basics/data/test_train.npy"
    valid_data_path: str = "cs336_basics/data/test_valid.npy"
    batch_size: int = 32
    eval_batches: int = 10
    max_iters: int = 500
    max_checkpoints: int = 5
    checkpoints_dir: str = "checkpoints"
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    lr_strategy: str | None = None

    logging_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 50
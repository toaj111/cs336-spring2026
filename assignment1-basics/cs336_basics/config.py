from dataclasses import dataclass, fields
import torch
import yaml

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
    device: str | torch.device | None = "cuda:0"
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


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("配置文件的顶层必须是 YAML mapping")

    # 检查未知字段，避免拼错参数后悄悄使用默认值
    valid_fields = {field.name for field in fields(Config)}
    unknown_fields = set(raw) - valid_fields
    if unknown_fields:
        raise ValueError(f"未知配置字段: {sorted(unknown_fields)}")

    if "device" in raw and raw["device"] is not None:
        raw["device"] = torch.device(raw["device"])

    if "dtype" in raw and raw["dtype"] is not None:
        dtype_names = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
        }
        dtype_name = raw["dtype"]
        if dtype_name not in dtype_names:
            raise ValueError(f"不支持的 dtype: {dtype_name}")
        raw["dtype"] = dtype_names[dtype_name]

    return Config(**raw)

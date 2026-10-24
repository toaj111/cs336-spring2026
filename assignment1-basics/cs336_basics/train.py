from model import TransformerLM
from optimizer import AdamW, SGD
from dataclasses import dataclass
import torch
from cs336_basics.data import data_loading
from cs336_basics.loss import cross_entropy
from tqdm import tqdm

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
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    # for ablation experiments only
    use_rope: bool = True
    use_rms_norm: bool = True
    use_pre_norm: bool = True
    use_swiglu: bool = True

    # hyperparameters for training
    # TODO: 完成data_path
    data_path: str = ""
    batch_size: int = 16
    max_iters: int = 1000
    checkpoints_path: str = "checkpoints/latest.pt"
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    lr_strategy: str | None = None

    logging_interval: int = 10
    eval_interval: int = 10

def train(config: Config):
    model = TransformerLM(config.vocab_size, config.context_length, config.d_model, config.num_layers, config.num_heads, config.d_ff, config.rope_theta, config.device, config.dtype)
    model.train()
    optimizer = AdamW(model.parameters, lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2), eps=config.eps) if config.optimizer == "adamw" else SGD(model.parameters, config.lr)

    pbar = tqdm(
        total=config.max_iters,
        desc="Model Training",
        unit="iter"
    )
    for step in range(config.max_iters):
        inputs, targets = data_loading(config.data_path, config.batch_size, config.context_length, config.device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        pbar.update(1)

    pbar.close()


        

if __name__ == "__main__":
    config = Config()
    train(config)







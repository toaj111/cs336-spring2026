from model import TransformerLM
from optimizer import AdamW, SGD, gradient_clipping, cosine_lr_schedule_with_warmup
from dataclasses import dataclass
import torch
from cs336_basics.data import data_loading
from cs336_basics.loss import cross_entropy
from cs336_basics.checkpointing import save_checkpoint
from pathlib import Path
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from cs336_basics.config import Config
from cs336_basics.config import load_config



def eval(model: TransformerLM, valid_data: npt.NDArray, context_length: int, batch_size: int, eval_batches: int, device: torch.device):
    model.eval()
    losses = []
    with torch.inference_mode():
        pbar = tqdm(
            total=eval_batches,
            desc="Evaluating",
            unit="iter"
        )

        for _ in range(eval_batches):
            inputs, targets = data_loading(valid_data, batch_size, context_length, device)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            losses.append(loss.item())
            pbar.update(1)
        pbar.close()
    model.train()
    return sum(losses) / len(losses)

def train(config: Config):
    model = TransformerLM(config.vocab_size, config.context_length, config.d_model, config.num_layers, config.num_heads, config.d_ff, config.rope_theta, config.device, config.dtype)
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2), eps=config.eps) if config.optimizer == "adamw" else SGD(model.parameters(), config.lr)
    training_data = np.load(config.training_data_path, mmap_mode="r")
    valid_data = np.load(config.valid_data_path, mmap_mode="r")
    pbar = tqdm(
        total=config.max_iters,
        desc="Model Training",
        unit="iter"
    )
    checkpoints_dir = Path(config.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_paths: list[Path] = []
    best_loss = float("inf")
    for iter in range(config.max_iters):
        inputs, targets = data_loading(training_data, config.batch_size, config.context_length, config.device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()

        gradient_clipping(model.parameters(), 1.0)
        lr = cosine_lr_schedule_with_warmup(iter + 1, config.lr, config.lr * 0.1, int(0.05 * config.max_iters), config.max_iters)
        for group in optimizer.param_groups:
            group["alpha"] = lr

        optimizer.step()
        pbar.update(1)
        if (iter + 1) % config.logging_interval == 0:
            print("iter:", iter + 1, "lr:", lr, "loss:", loss.item())

        if (iter + 1) % config.eval_interval == 0:
            valid_loss = eval(model, valid_data, config.context_length, config.batch_size, config.eval_batches, config.device)
            print("valid loss:", valid_loss)
            if valid_loss < best_loss:
                best_model_path = checkpoints_dir / f"best_model.pt"
                save_checkpoint(model, optimizer, iter + 1, best_model_path)
                best_loss = valid_loss

        if (iter + 1) % config.save_interval == 0:
            checkpoint_path = checkpoints_dir / f"checkpoint_iter_{iter + 1}.pt"
            if len(checkpoints_paths) == config.max_checkpoints:
                checkpoints_paths[0].unlink()
                checkpoints_paths.pop(0)
            checkpoints_paths.append(checkpoint_path)
            save_checkpoint(model, optimizer, iter + 1, checkpoint_path)

    pbar.close()


        

if __name__ == "__main__":
    config_path = "cs336_basics/configs/baseline.yaml"
    config = load_config(config_path)
    train(config)







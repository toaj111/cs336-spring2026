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
from cs336_basics.config import load_config, write_config_to_json
import csv
import time
from datetime import datetime
import json

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

    # 文件准备
    experiments_dir = Path(config.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiments_dir / config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir / config.checkpoints_dir
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = experiment_dir / "metrics.csv"
    summary_path = experiment_dir / "summary.json"
    config_path = experiment_dir / "config.json"
    write_config_to_json(config, config_path)

    checkpoints_paths: list[Path] = []
    best_valid_loss = float("inf")
    best_valid_step = None
    best_train_loss = float("inf")
    best_train_step = None
    final_valid_loss = None
    final_train_loss = None

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "iteration",
                "train_loss",
                "valid_loss",
                "learning_rate",
                "interval_time",
                "elapsed_time",
                "eval_seconds"
            ],
        )
        writer.writeheader()
        start_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        training_start = time.perf_counter()
        last_log_time = training_start
        for iter in range(config.max_iters):
            # 核心训练代码
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

            # 模型评测和log
            if (iter + 1) % config.logging_interval == 0:
                print("iter:", iter + 1, "lr:", lr, "loss:", loss.item())

            if (iter + 1) % config.eval_interval == 0:
                current_time = time.perf_counter()
                interval_time = current_time - last_log_time
                elapsed_time = current_time - training_start

                # 在验证集上进行测试
                valid_loss = eval(model, valid_data, config.context_length, config.batch_size, config.eval_batches, config.device)
                
                eval_end_time = time.perf_counter()
                eval_seconds = eval_end_time - current_time
                row = {
                    "timestamp": datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                    "iteration": iter + 1,
                    "train_loss": loss.item(),
                    "valid_loss": valid_loss,
                    "learning_rate": lr,
                    "interval_time": interval_time,
                    "elapsed_time": elapsed_time,
                    "eval_seconds": eval_seconds,
                }

                writer.writerow(row)
                f.flush()

                last_log_time = current_time
                
                print("valid loss:", valid_loss)
                if valid_loss < best_valid_loss:
                    best_model_path = checkpoints_dir / f"best_model.pt"
                    save_checkpoint(model, optimizer, iter + 1, best_model_path)
                    best_valid_loss = valid_loss
                    best_valid_step = iter + 1

                if loss.item() < best_train_loss:
                    best_train_loss = loss.item()
                    best_train_step = iter + 1

                if iter + 1 == config.max_iters:
                    final_train_loss = loss.item()
                    final_valid_loss = valid_loss

            if (iter + 1) % config.save_interval == 0:
                checkpoint_path = checkpoints_dir / f"checkpoint_iter_{iter + 1}.pt"
                if len(checkpoints_paths) == config.max_checkpoints:
                    checkpoints_paths[0].unlink()
                    checkpoints_paths.pop(0)
                checkpoints_paths.append(checkpoint_path)
                save_checkpoint(model, optimizer, iter + 1, checkpoint_path)

        
    end_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")   
    total_time = time.perf_counter() - training_start

    summary = {
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "total_time": total_time,
        "best_train_loss": best_train_loss,
        "best_train_step": best_train_step,
        "best_valid_loss": best_valid_loss,
        "best_valid_step": best_valid_step,
        "final_train_loss": final_train_loss,
        "final_valid_loss": final_valid_loss,
    }


    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    pbar.close()



        

if __name__ == "__main__":
    config_path = "cs336_basics/configs/baseline.yaml"
    config = load_config(config_path)
    train(config)







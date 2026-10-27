import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curves(csv_path: str | Path, output_path: str | Path):
    df = pd.read_csv(csv_path)

    required_columns = ["iteration", "train_loss", "valid_loss"]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV 缺少字段: {missing_columns}")

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_columns).sort_values("iteration")

    if df.empty:
        raise ValueError("CSV 中没有可用于绘图的有效数据")
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["iteration"],
        df["train_loss"],
        label="Train Loss",
        color="blue",
        marker="o",
        markersize=3,
    )
    ax.plot(
        df["iteration"],
        df["valid_loss"],
        label="Validation Loss",
        color="red",
        marker="s",
        markersize=3,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or args.csv.parent / "learning_curves.png"
    plot_learning_curves(args.csv, output_path)
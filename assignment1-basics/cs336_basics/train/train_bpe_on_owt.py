from cs336_basics.bpe import train_bpe
import json
from pathlib import Path

input_path = "data/owt_valid.txt"
special_tokens = ["<|endoftext|>"]
vocab_size = 10000

def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | Path,
    merges_path: str | Path,
) -> None:
    """把 vocab 和 merges 保存到两个文件。

    vocab  -> JSON: {"0": "\\x00", "1": "\\x01", ...}
    merges -> 每行 "token1 token2"，token 用 latin-1 解码
    """
    vocab_path = Path(vocab_path)
    merges_path = Path(merges_path)

    vocab_json = {
        str(token_id): token_bytes.decode("latin-1")
        for token_id, token_bytes in vocab.items()
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            line = f"{a.decode('latin-1')} {b.decode('latin-1')}"
            f.write(line + "\n")


def load_vocab_and_merges(
    vocab_path: str | Path,
    merges_path: str | Path,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """从两个文件读回 vocab 和 merges。"""
    vocab_path = Path(vocab_path)
    merges_path = Path(merges_path)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab = {
        int(token_id): token_str.encode("latin-1")
        for token_id, token_str in vocab_json.items()
    }

    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            a_str, b_str = line.split(" ", 1)   
            merges.append((a_str.encode("latin-1"), b_str.encode("latin-1")))

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    save_vocab_and_merges(vocab, merges, "cs336_basics/train/owt_vocab", "cs336_basics/train/owt_merges")
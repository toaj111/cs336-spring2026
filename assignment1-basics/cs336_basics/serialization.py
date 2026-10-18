import json
import numpy as np

def save_vocab(vocab: dict[int, bytes], file_path: str):
    # 经过合并后可能会出现不合法的utf-8序列，因此需要用其他方式编码解码
    serializable_vocab = {
        str(token_id) : token.decode("latin-1")
        for token_id, token in vocab.items()
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)

def load_vocab(file_path: str) -> dict[int, bytes]:
    with open(file_path, "r", encoding="utf-8") as f:
        serializable_vocab: dict[str, str] = json.load(f)

    vocab = {
        int(token_id): token.encode(encoding="latin-1")
        for token_id, token in serializable_vocab.items()
    }

    return vocab


def save_merges(merges: list[tuple[bytes, bytes]], file_path: str):
    serializable_merges = [[token1.decode("latin-1"), 
                            token2.decode("latin-1")] 
                            for token1, token2 in merges]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_merges, f, ensure_ascii=False, indent=2)

def load_merges(file_path: str) -> list[tuple[bytes, bytes]]:
    with open(file_path, "r", encoding="utf-8") as f:
        serializable_merges: list[list[str]] = json.load(f)

    merges = [(token1.encode("latin-1"), 
               token2.encode("latin-1")) 
               for token1, token2 in serializable_merges]

    return merges


# TODO: 完成tokenid的序列化和逆序列化
def save_token_IDs(token_ids: list[int]):
    pass

def load_token_IDs(file_path: str) -> list[np.uint16]:
    pass
    

from cs336_basics.bpe import train_bpe
import numpy as np
from pathlib import Path
from cs336_basics.serialization import save_vocab, save_merges
from cs336_basics.tokenizer import Tokenizer

# 完成训练数据的生成
input_path = "data/TinyStoriesV2-GPT4-valid.txt"
special_tokens = ["<|endoftext|>"]
vocab_size = 10000
output_dir = "cs336_basics/data"

if __name__ == "__main__":
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    save_vocab(vocab, "cs336_basics/data/tiny_story_vocab.json")
    save_merges(merges, "cs336_basics/data/tiny_story_merges.json")
    tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens)

    batch_size = 1000000
    batch = []
    batch_files = []
    # 文件可能很大，因此需要分批次读出
    with open(input_path, "r", encoding="utf-8") as f:
        token_iter = tokenizer.encode_iterable(f)

        for token_id in token_iter:
            batch.append(token_id)

            if len(batch) >= batch_size:
                arr = np.asarray(batch, dtype=np.uint16)
                path = output_dir / f"batch_{len(batch_files):05d}.npy"
                np.save(path, arr)
                batch_files.append(path)
                batch.clear()


    if batch:
        arr = np.asarray(batch, dtype=np.uint16)
        path = output_dir / f"batch_{len(batch_files):05d}.npy"
        np.save(path, arr)
        batch_files.append(path)
        batch.clear()

    total_tokens = sum(np.load(p, mmap_mode="r").size for p in batch_files)

    final_path = output_dir / "tiny_story_tokens.npy"
    merged = np.lib.format.open_memmap(
        final_path, 
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens, ),
    )

    offset = 0
    for path in batch_files:
        arr = np.load(path)
        merged[offset:offset+len(arr)] = arr
        offset += len(arr)

    merged.flush()
    del merged

    for path in batch_files:
        path.unlink()


    print(f"Token 数量: {total_tokens}")
    print(f"保存位置: {final_path}")


    

from cs336_basics.bpe import train_bpe
import json
from pathlib import Path
from cs336_basics.serialization import save_vocab, save_merges

input_path = "data/owt_valid.txt"
special_tokens = ["<|endoftext|>"]
vocab_size = 32000

if __name__ == "__main__":
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    save_vocab(vocab, "cs336_basics/train/owt_vocab.json")
    save_merges(merges, "cs336_basics/train/owt_merges.json")
    
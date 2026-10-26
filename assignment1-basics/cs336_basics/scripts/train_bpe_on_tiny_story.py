from cs336_basics.bpe import train_bpe
import json
from pathlib import Path
from cs336_basics.serialization import save_vocab, save_merges


input_path = "data/TinyStoriesV2-GPT4-train.txt"
special_tokens = ["<|endoftext|>"]
vocab_size = 10000

if __name__ == "__main__":
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    save_vocab(vocab, "cs336_basics/scripts/tiny_story_vocab.json")
    save_merges(merges, "cs336_basics/scripts/tiny_story_merges.json")
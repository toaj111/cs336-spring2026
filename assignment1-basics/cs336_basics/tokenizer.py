from typing import Iterable
from cs336_basics.serialization import load_merges, load_vocab

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens


    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab = load_vocab(vocab_filepath)
        merges = load_merges(merges_filepath)

        return Tokenizer(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        token_bytes = bytes()

        for token_id in ids:
            if token_id not in self.vocab:
                replacement_bytes = "\uFFFD".encode("utf-8")
                token_bytes += replacement_bytes
            else:
                token_bytes += self.vocab[id]

        return token_bytes.decode("utf-8", errors="replace")





from typing import Iterable, Iterator
from cs336_basics.serialization import load_merges, load_vocab
import regex as re


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        if special_tokens:
            self.special_tokens = sorted(special_tokens, reverse=True)
        else:
            self.special_tokens = []
        special_token_bytes = [token.encode() for token in self.special_tokens]
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

        for special_token in special_token_bytes:
            if special_token not in self.reversed_vocab:
                id = len(self.vocab)
                self.reversed_vocab[special_token] = id
                self.vocab[id] = special_token

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab = load_vocab(vocab_filepath)
        merges = load_merges(merges_filepath)

        return Tokenizer(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            PAT = "|".join(re.escape(token) for token in self.special_tokens)
            chunks = re.split(f"({PAT})", text)
        else:
            chunks = [text]

        pretokens = self._pretokenize(chunks)

        tokens = self._merge(pretokens)
        token_ids = [self.reversed_vocab[token] for token in tokens]

        return token_ids

    # TODO: 完成test_encode_iterable_memory_usage测试
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # 使用AI完成
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            safe_len = self._safe_split_index(buffer)
            if safe_len > 0:
                yield from self.encode(buffer[:safe_len])
                buffer = buffer[safe_len:]
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        token_bytes = bytes()

        for token_id in ids:
            if token_id not in self.vocab:
                replacement_bytes = "\uFFFD".encode("utf-8")
                token_bytes += replacement_bytes
            else:
                token_bytes += self.vocab[token_id]

        return token_bytes.decode("utf-8", errors="replace")



    def _pretokenize(self, chunks: list[str]) -> list[tuple[bytes, ...] | bytes]:
        special_tokens_set = set(self.special_tokens)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretokens: list[tuple[bytes, ...] | bytes] = []

        for chunk in chunks:
            if chunk in special_tokens_set:
                pretokens.append(chunk.encode())
            else:
                matches = re.finditer(PAT, chunk)
                for match in matches:
                    pretoken_str = match.group()
                    pretoken_bytes = tuple(bytes([byte]) for byte in pretoken_str.encode("utf-8"))
                    pretokens.append(pretoken_bytes)

        return pretokens


    def _merge(self, pretokens: list[tuple[bytes, ...] | bytes]) -> list[bytes]:
        tokens: list[bytes] = [] 
        for pretoken in pretokens:
            if isinstance(pretoken, bytes):
                tokens.append(pretoken)
                continue

            new_pretoken = list(pretoken)
            for merge in self.merges:
                old_pretoken = new_pretoken
                new_pretoken = []

                i = 0
                n = len(old_pretoken)
                new_token = merge[0] + merge[1]
                while (i < n):
                    if i + 1 < n and old_pretoken[i] == merge[0] and old_pretoken[i + 1] == merge[1]:
                        new_pretoken.append(new_token)
                        i += 2
                    else:
                        new_pretoken.append(old_pretoken[i])
                        i += 1

            tokens += new_pretoken
        return tokens

    

    def _safe_split_index(self, text: str) -> int:
        """
        返回 text 中最长的、可以安全地立即 tokenize 的前缀长度。
        余下的部分（可能被后续输入扩展）会被留到下一次。
        """
        # 使用ai完成

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        n = len(text)

        # (1) 如果结尾是某个特殊 token 的真前缀，回退这段
        back = 0
        for st in self.special_tokens:
            for k in range(len(st) - 1, 0, -1):  # 从长到短找最长的合法前缀
                if k > back and text.endswith(st[:k]):
                    back = k
                    break
        n -= back
        if n <= 0:
            return 0

        prefix = text[:n]

        # (2) 把 prefix 拆成 units（特殊 token + 预 token），丢掉最后一个 unit
        if self.special_tokens:
            pat = "|".join(re.escape(t) for t in self.special_tokens)
            parts = re.split(f"({pat})", prefix)
        else:
            parts = [prefix]

        special_set = set(self.special_tokens)
        pos = 0
        last_unit_start = 0
        for part in parts:
            if not part:
                continue
            if part in special_set:
                last_unit_start = pos           # 特殊 token 算一个 unit
                pos += len(part)
            else:
                for m in re.finditer(PAT, part):
                    last_unit_start = pos + m.start()  # 记录最后一个预 token 的起点
                pos += len(part)

        return last_unit_start
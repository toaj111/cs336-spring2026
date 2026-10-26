from typing import Iterable, Iterator
from cs336_basics.serialization import load_merges, load_vocab
import regex as re


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}
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

            pretoken = list(pretoken)
            while True:
                best_rank = None
                best_merge = None
                n = len(pretoken)
                for i in range(n - 1):
                    pair = (pretoken[i], pretoken[i + 1])
                    if pair in self.merges_rank and (best_rank is None or self.merges_rank[pair] < best_rank):
                        best_rank = self.merges_rank[pair]
                        best_merge = pair

                if best_merge is None:
                    break

                new_pretoken = []
                i = 0
                while i < n:
                    if i + 1 < n and pretoken[i] == best_merge[0] and pretoken[i + 1] == best_merge[1]:
                        new_pretoken.append(best_merge[0] + best_merge[1])
                        i += 2
                    else:
                        new_pretoken.append(pretoken[i])
                        i += 1

                pretoken = new_pretoken

            tokens += pretoken
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

# import heapq
# import os
# import time
# from collections import Counter, defaultdict
# from dataclasses import dataclass
# from multiprocessing import Pool, cpu_count
# from typing import BinaryIO

# import regex as re

# # GPT-2 style pre-tokenizer pattern
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# def find_chunk_boundaries(
#     file: BinaryIO,
#     desired_num_chunks: int,
#     split_special_token: bytes,
# ) -> list[int]:
#     """Chunk the file into independently countable parts (course example, inlined)."""
#     assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

#     file.seek(0, os.SEEK_END)
#     file_size = file.tell()
#     file.seek(0)

#     chunk_size = file_size // desired_num_chunks
#     chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
#     chunk_boundaries[-1] = file_size

#     mini_chunk_size = 4096
#     for bi in range(1, len(chunk_boundaries) - 1):
#         initial_position = chunk_boundaries[bi]
#         file.seek(initial_position)
#         while True:
#             mini_chunk = file.read(mini_chunk_size)
#             if mini_chunk == b"":
#                 chunk_boundaries[bi] = file_size
#                 break
#             found_at = mini_chunk.find(split_special_token)
#             if found_at != -1:
#                 chunk_boundaries[bi] = initial_position + found_at
#                 break
#             initial_position += mini_chunk_size

#     return sorted(set(chunk_boundaries))


# def _count_chunk(args: tuple) -> Counter:
#     """Worker: count pre-token byte-sequences in a file byte range.

#     Must be top-level for Windows spawn pickling.
#     """
#     input_path, start, end, special_tokens = args
#     with open(input_path, "rb") as f:
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")

#     pieces = [chunk]
#     for st in special_tokens:
#         new_pieces = []
#         for piece in pieces:
#             new_pieces.extend(piece.split(st))
#         pieces = new_pieces

#     counts: Counter = Counter()
#     for piece in pieces:
#         if not piece:
#             continue
#         for pt in re.findall(PAT, piece):
#             # iterating bytes yields ints 0..255 (base token ids)
#             counts[tuple(pt.encode("utf-8"))] += 1
#     return counts


# def _pretokenize_counts(input_path: str, special_tokens: list[str]) -> Counter:
#     """Parallel pre-tokenization → Counter mapping byte-id tuple → frequency."""
#     # Split on first special when possible (hard boundary); else newlines.
#     if special_tokens:
#         split_token = special_tokens[0].encode("utf-8")
#     else:
#         split_token = b"\n"

#     num_processes = max(1, min(8, cpu_count() or 1))
#     with open(input_path, "rb") as f:
#         boundaries = find_chunk_boundaries(f, num_processes, split_token)

#     jobs = [
#         (input_path, start, end, special_tokens)
#         for start, end in zip(boundaries[:-1], boundaries[1:])
#         if end > start
#     ]

#     counts: Counter = Counter()
#     if not jobs:
#         return counts

#     if len(jobs) == 1:
#         counts.update(_count_chunk(jobs[0]))
#         return counts

#     # Windows-safe: worker is module-level _count_chunk
#     with Pool(processes=len(jobs)) as pool:
#         for part in pool.imap_unordered(_count_chunk, jobs):
#             counts.update(part)
#     return counts


# def _merge_word_ids(word: list[int], left: int, right: int, new_id: int) -> list[int]:
#     """Left-to-right merge of (left, right) → new_id inside one word."""
#     out: list[int] = []
#     i = 0
#     n = len(word)
#     while i < n:
#         if i + 1 < n and word[i] == left and word[i + 1] == right:
#             out.append(new_id)
#             i += 2
#         else:
#             out.append(word[i])
#             i += 1
#     return out


# @dataclass(slots=True)
# class _HeapEntry:
#     """Max-heap entry for BPE pair selection (used with heapq min-heap).

#     Priority matches assignment tie-break:
#       higher count first, then lexicographically larger left bytes, then right bytes.
#     __lt__ is inverted so heapq.heappop returns the best pair.
#     """

#     count: int
#     left_b: bytes
#     right_b: bytes
#     pair: tuple[int, int]

#     def __lt__(self, other: "_HeapEntry") -> bool:
#         if self.count != other.count:
#             return self.count > other.count
#         if self.left_b != other.left_b:
#             return self.left_b > other.left_b
#         if self.right_b != other.right_b:
#             return self.right_b > other.right_b
#         # Deterministic residual ordering (does not change assignment semantics
#         # when counts/bytes already differ).
#         return self.pair < other.pair


# def train_bpe(
#     input_path: str,
#     vocab_size: int,
#     special_tokens: "list[str]",
# ) -> "tuple[dict[int, bytes], list[tuple[bytes, bytes]]]":
#     """Train a byte-level BPE tokenizer.

#     Performance-oriented version:
#     - parallel pre-tokenization (multiprocessing + chunk boundaries)
#     - incremental pair counts
#     - inverted index pair → word ids (only touch affected words each merge)
#     - lazy max-heap for best-pair selection (no full scan of pair_counts each merge)
#     """
#     t_all0 = time.perf_counter()

#     # ---- vocab: 256 bytes + specials ----
#     vocab_dict: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
#     for i, st in enumerate(special_tokens):
#         vocab_dict[256 + i] = st.encode("utf-8")

#     # ---- pre-tokenize + count unique byte words ----
#     t0 = time.perf_counter()
#     raw_counts = _pretokenize_counts(input_path, special_tokens)
#     t_pretok = time.perf_counter() - t0

#     # Dense word table for fast index-based updates
#     words: list[list[int]] = [list(seq) for seq in raw_counts.keys()]
#     freqs: list[int] = list(raw_counts.values())
#     del raw_counts

#     # pair_counts[(a,b)] = total frequency of adjacent pair across corpus (source of truth)
#     pair_counts: dict[tuple[int, int], int] = {}
#     # pair_index[(a,b)] = set of word indices that currently contain the pair
#     pair_index: dict[tuple[int, int], set[int]] = defaultdict(set)
#     # Lazy max-heap: may contain stale entries; always verified against pair_counts
#     pair_heap: list[_HeapEntry] = []

#     def push_pair(p: tuple[int, int]) -> None:
#         c = pair_counts.get(p)
#         if c is None or c <= 0:
#             return
#         a, b = p
#         heapq.heappush(
#             pair_heap,
#             _HeapEntry(c, vocab_dict[a], vocab_dict[b], p),
#         )

#     def add_word_pairs(widx: int, *, update_heap: bool = True) -> None:
#         """Add this word's adjacent pairs into pair_counts / pair_index.

#         During corpus-wide init, pass update_heap=False and build the heap once
#         after all counts are final (avoids O(#words) stale pushes per pair).
#         During merge updates, leave update_heap=True so only changed pairs are pushed.
#         """
#         w = words[widx]
#         f = freqs[widx]
#         if len(w) < 2:
#             return
#         touched: set[tuple[int, int]] = set()
#         for i in range(len(w) - 1):
#             p = (w[i], w[i + 1])
#             pair_counts[p] = pair_counts.get(p, 0) + f
#             pair_index[p].add(widx)
#             if update_heap:
#                 touched.add(p)
#         if update_heap:
#             for p in touched:
#                 push_pair(p)

#     def remove_word_pairs(widx: int, *, update_heap: bool = True) -> None:
#         w = words[widx]
#         f = freqs[widx]
#         if len(w) < 2:
#             return
#         touched: set[tuple[int, int]] = set()
#         for i in range(len(w) - 1):
#             p = (w[i], w[i + 1])
#             new_c = pair_counts.get(p, 0) - f
#             if new_c <= 0:
#                 pair_counts.pop(p, None)
#                 pair_index.pop(p, None)
#             else:
#                 pair_counts[p] = new_c
#                 s = pair_index.get(p)
#                 if s is not None:
#                     s.discard(widx)
#                     if not s:
#                         del pair_index[p]
#                 if update_heap:
#                     touched.add(p)
#         if update_heap:
#             for p in touched:
#                 push_pair(p)

#     def pop_best_pair() -> tuple[int, int] | None:
#         """Pop best live pair; skip heap entries whose count is stale."""
#         while pair_heap:
#             entry = heapq.heappop(pair_heap)
#             cur = pair_counts.get(entry.pair)
#             if cur is None or cur != entry.count:
#                 continue
#             # vocab bytes for an id never change after creation; count match is enough
#             return entry.pair
#         return None

#     def build_heap_from_counts() -> None:
#         """One heap entry per live pair after init (no stale duplicates)."""
#         nonlocal pair_heap
#         pair_heap = [
#             _HeapEntry(c, vocab_dict[a], vocab_dict[b], (a, b))
#             for (a, b), c in pair_counts.items()
#             if c > 0
#         ]
#         heapq.heapify(pair_heap)

#     t0 = time.perf_counter()
#     # Init: count only — do NOT push into the heap per word (would explode memory).
#     for widx in range(len(words)):
#         add_word_pairs(widx, update_heap=False)
#     build_heap_from_counts()
#     t_init_pairs = time.perf_counter() - t0

#     merges: list[tuple[bytes, bytes]] = []
#     num_merges = vocab_size - len(vocab_dict)
#     t_pair = 0.0
#     t_update = 0.0
#     progress_every = 500

#     for merge_i in range(num_merges):
#         if not pair_counts:
#             break

#         # ---- pick most frequent pair; tie-break by vocab bytes (lazy heap) ----
#         t0 = time.perf_counter()
#         best_pair = pop_best_pair()
#         t_pair += time.perf_counter() - t0
#         if best_pair is None:
#             break

#         left, right = best_pair
#         merges.append((vocab_dict[left], vocab_dict[right]))
#         new_id = len(vocab_dict)
#         vocab_dict[new_id] = vocab_dict[left] + vocab_dict[right]

#         # ---- only words that contain best_pair (pair_index is exact) ----
#         t0 = time.perf_counter()
#         affected = list(pair_index.get(best_pair, ()))
#         for widx in affected:
#             w = words[widx]
#             remove_word_pairs(widx)
#             words[widx] = _merge_word_ids(w, left, right, new_id)
#             add_word_pairs(widx)
#         t_update += time.perf_counter() - t0

#         if progress_every and (merge_i + 1) % progress_every == 0:
#             print(
#                 f"merge {merge_i + 1}/{num_merges}",
#                 "pairs:", len(pair_counts),
#                 "affected:", len(affected),
#                 "pair_t:", round(t_pair, 3),
#                 "update_t:", round(t_update, 3),
#                 flush=True,
#             )

#     t_all = time.perf_counter() - t_all0
#     print(
#         "pretok:", round(t_pretok, 3),
#         "init_pairs:", round(t_init_pairs, 3),
#         "pair:", round(t_pair, 3),
#         "update:", round(t_update, 3),
#         "total:", round(t_all, 3),
#     )
#     return vocab_dict, merges

# class Tokenizer():
#     def __init__(
#             self,
#             vocab: dict[int, bytes],
#             merges: list[tuple[bytes, bytes]],
#             special_tokens: list[str] | None = None
#     ):
#         self.vocab = vocab
#         self.byte_to_id = {b: i for i, b in vocab.items()}
#         self.merges = merges
#         self.special_tokens = special_tokens if special_tokens is not None else []
#         self.merges_rank = {pair: i for i, pair in enumerate(merges)}

#     def encode(self, text: str) -> list[int]:
#         """把文本编码成 token id 列表。

#         流程：
#         1) 若有 special tokens,先按它们切分(special 本身保留为独立一段）
#         2) special 段 → 直接查一个 id
#         3) 普通段 → PAT 预分词 → 每段做 BPE merge
#         """
#         all_ids = []

#         # ---- 1) 按 special 切成 parts ----
#         if not self.special_tokens:
#             parts = [text]
#         else:
#             # 长的 special 优先，避免短串抢匹配
#             ordered = sorted(self.special_tokens, key=len, reverse=True)
#             # 捕获组 () 让 re.split 把分隔符（special）也留在结果里
#             pattern = "(" + "|".join(re.escape(st) for st in ordered) + ")"
#             parts = re.split(pattern, text)

#         # ---- 2) 逐段处理 ----
#         for part in parts:
#             if part == "":
#                 continue

#             if part in self.special_tokens:
#                 # special：整段映射成一个 token id
#                 all_ids.append(self.byte_to_id[part.encode("utf-8")])
#             else:
#                 # 普通文本：PAT → 字节序列 → BPE merge
#                 pre_tokens = re.findall(PAT, part)
#                 for pt in pre_tokens:
#                     raw = pt.encode("utf-8")
#                     seq = [self.byte_to_id[bytes([b])] for b in raw]
#                     seq = self._merge_word(seq)
#                     all_ids.extend(seq)

#         return all_ids

#     def decode(self, ids: list[int]) -> str:
#         decode_bytes = b"".join(self.vocab[token_id] for token_id in ids)
#         # A single token may be only part of a UTF-8 sequence (e.g. b'\xc3');
#         # replace so decode([id]) still works. Full id lists that form valid
#         # UTF-8 still round-trip correctly.
#         return decode_bytes.decode("utf-8", errors="replace")
    
#     def _merge_word(self, seq: list[int]) -> list[int]:
#         while True:
#             # 找到最小的rank的pair
#             best_pair = None
#             best_rank = None
#             for i in range(len(seq)-1):
#                 pair_bytes = (self.vocab[seq[i]], self.vocab[seq[i+1]])
#                 if pair_bytes in self.merges_rank:
#                     r = self.merges_rank[pair_bytes]
#                     if best_rank is None or r < best_rank:
#                         best_rank = r
#                         best_pair = pair_bytes
#             # 没有合并的返回结果
#             if best_pair is None:
#                 return seq
#             # 合并
#             new_id = self.byte_to_id[best_pair[0] + best_pair[1]]
#             new_seq = []
#             i = 0
#             while i< len(seq):
#                 if(
#                     i + 1 < len(seq)
#                     and (self.vocab[seq[i]], self.vocab[seq[i+1]]) == best_pair
#                 ):
#                     new_seq.append(new_id)
#                     i += 2
#                 else:
#                     new_seq.append(seq[i])
#                     i += 1
#             seq = new_seq
#     def encode_iterable(self, iterable):
#         i = 0
#         for text in iterable:
#             for token_id in self.encode(text):
#                 yield token_id

#             print("line:", i)
#             i += 1
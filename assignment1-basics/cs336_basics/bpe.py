from cs336_basics.pretokenization import parallel_pretokenize, serial_pretokenize
from tqdm import tqdm

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = build_initial_vocab(special_tokens)
    merges = list()

    if len(vocab) >= vocab_size:
        return vocab, merges

    # pretoken_counts = serial_pretokenize(input_path, special_tokens)
    pretoken_counts = parallel_pretokenize(input_path, special_tokens, 8)
    print("预分词完成")

    pair_counts = get_pair_counts(pretoken_counts)
    print("初始化pair_counts完成")

    pbar = tqdm(
        total=vocab_size - len(vocab),
        desc="BPE Training",
        unit="merge",
    )

    while len(vocab) < vocab_size:
        best_pair = select_best_pair(pair_counts)
        merges.append(best_pair)
        new_token = merge_pair_and_update_counts(pretoken_counts, pair_counts, best_pair)
        del pair_counts[best_pair]
        vocab[len(vocab)] = new_token

        pbar.update(1)

    pbar.close()

    return vocab, merges


def build_initial_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = dict()
    for i in range(256):
        vocab[i] = bytes([i])

    for i, special_token in enumerate(special_tokens):
        vocab[i + 256] = special_token.encode("utf-8") 

    return vocab

def get_pair_counts(pretoken_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pair_counts = dict()
    for pretoken_bytes, cnt in pretoken_counts.items():
        for i in range(len(pretoken_bytes) - 1):
            pair = (pretoken_bytes[i], pretoken_bytes[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + cnt

    return pair_counts

def merge_pair_and_update_counts(pretoken_counts: dict[tuple[bytes, ...], int], pair_counts: dict[tuple[bytes, bytes], int], best_pair: tuple[bytes, bytes]) -> bytes:
    new_token = best_pair[0] + best_pair[1]
    to_delete = set()
    to_add = set()
    for pretoken_bytes, cnt in pretoken_counts.items():
        new_pretoken_bytes = []
        n = len(pretoken_bytes)
        i = 0
        changed = False
        while i < n:
            if i + 1 < n and pretoken_bytes[i] == best_pair[0] and pretoken_bytes[i + 1] == best_pair[1]:
                changed = True
                pair_counts[best_pair] -= cnt
                
                if i + 2 < n:
                    old_pair = (pretoken_bytes[i + 1], pretoken_bytes[i + 2])
                    new_pair = (new_token, pretoken_bytes[i + 2])
                    pair_counts[old_pair] -= cnt
                    pair_counts[new_pair] = pair_counts.get(new_pair, 0) + cnt

                # 处理特殊情况，两个要合并的pair相连，比如要合并ab，序列中有abab
                if new_pretoken_bytes and new_pretoken_bytes[len(new_pretoken_bytes) - 1] == new_token:
                    old_pair = (new_token, pretoken_bytes[i])
                    new_pair = (new_token, new_token)
                    pair_counts[old_pair] -= cnt
                    pair_counts[new_pair] = pair_counts.get(new_pair, 0) + cnt
                else:
                    if i - 1 >= 0:
                        old_pair = (pretoken_bytes[i - 1], pretoken_bytes[i])
                        new_pair = (pretoken_bytes[i - 1], new_token)
                        pair_counts[old_pair] -= cnt
                        pair_counts[new_pair] = pair_counts.get(new_pair, 0) + cnt

                new_pretoken_bytes.append(new_token)
                i += 2
            else:
                new_pretoken_bytes.append(pretoken_bytes[i])
                i += 1

        if changed:
            to_delete.add(pretoken_bytes)
            to_add.add((tuple(new_pretoken_bytes), cnt))

    for pretoken_bytes in to_delete:
        del pretoken_counts[pretoken_bytes]

    for new_pretoken_bytes in to_add:
        pretoken_counts[new_pretoken_bytes[0]] = new_pretoken_bytes[1]

    return new_token

# TODO: 优化
def select_best_pair(pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    best_pair = max(pair_counts.items(), key=lambda x : (x[1], x[0]))
    return best_pair[0]
import regex as re
import multiprocessing as mp
from typing import BinaryIO
import os
from collections import Counter

def parallel_pretokenize(input_path: str, special_tokens: list[str], num_processes: int = 4) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "rb") as f:
        split_special_tokens = [special_token.encode("utf-8") for special_token in special_tokens]
        boundaries = find_chunk_boundaries(f, num_processes, split_special_tokens)

    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path, special_tokens, (start, end)))

        
    with mp.Pool(processes=num_processes) as pool:
        partial_pretoken_counts = pool.starmap(pretokenize_worker, tasks)

    pretoken_counts = Counter()

    for counts in partial_pretoken_counts:
        pretoken_counts.update(counts)


    return pretoken_counts

def serial_pretokenize(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    pretoken_counts = pretokenize(text, special_tokens)
    return pretoken_counts

def pretokenize_worker(input_path: str, special_tokens: list[str], boundry: tuple[int, int]) -> dict[tuple[bytes, ...], int]:
    start, end = boundry[0], boundry[1]
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    partial_pretoken_counts = pretokenize(chunk, special_tokens)
    return partial_pretoken_counts

def pretokenize(chunk: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    segments = split_special_tokens(chunk, special_tokens)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretoken_counts = dict()

    for segment in segments:
        # 使用finditer而非findall不用一次读出所有匹配项可以节省内存
        matches = re.finditer(PAT, segment)
        for match in matches:
            pretoken_str = match.group()
            pretoken_bytes = tuple(bytes([byte]) for byte in pretoken_str.encode("utf-8"))
            pretoken_counts[pretoken_bytes] = pretoken_counts.get(pretoken_bytes, 0) + 1

    return pretoken_counts 

def split_special_tokens(chunk: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [chunk]
    
    PAT = "|".join(re.escape(token) for token in special_tokens)
    return re.split(PAT, chunk)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found = False
            for split_special_token in split_special_tokens:
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    found = True
                    break
            if found:
                break

            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
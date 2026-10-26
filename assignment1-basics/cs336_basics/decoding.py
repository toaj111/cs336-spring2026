from cs336_basics.checkpointing import load__model_from_checkpoint, load_checkpoint
from model import TransformerLM
from tokenizer import Tokenizer
import torch
from layers import softmax

MODEL_PATH = "checkpoints/best_model.pt"
TEMPERATURE = 1.0
TOP_P = 0.9
EOS_TOKEN = "<|endoftext|>"
VOCAB_PATH = "cs336_basics/scripts/tiny_story_vocab.json"
MERGES_PATH = "cs336_basics/scripts/tiny_story_merges.json"
MAX_NEW_TOKENS = 256

def softmax_with_temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("temperature must >= 0")
    if temperature == 0:
        _, idx = logits.max(dim=-1)
        probs = torch.zeros_like(logits)
        probs[idx] = 1.0
        return probs
    logits = logits / temperature
    return softmax(logits, dim=-1)


def top_p_sampling(logits: torch.Tensor, temperature: float, p: float) -> torch.Tensor:
    if p <= 0 or p > 1:
        raise ValueError("p must > 0 and <= 1")
    probs = softmax_with_temperature_scaling(logits, temperature)
    if p == 1:
        return probs
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    pre_sum = torch.cumsum(sorted_probs, dim=-1)
    remove_mask = (pre_sum - sorted_probs) >= p # 即对于每个i 如果 p_0 + ... + p_{i - 1} >= p 则说明该项需要被移除
    remove_idx = sorted_indices[remove_mask] # 得到要删除的索引号
    probs[remove_idx] = 0
    return probs / probs.sum(dim=-1)



def generate(
    model: TransformerLM,
    prompt_tokens: list[int],
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[int]:
    prompt_len = len(prompt_tokens)
    prefix = prompt_tokens.copy()
    while len(prefix) < max_new_tokens:
        logits = model(prefix)[-1]
        probs = top_p_sampling(logits, temperature, top_p)
        idx = torch.multinomial(probs, num_samples=1).item()
        if idx == eos_token_id:
            break
        prefix.append(idx)

    return prefix[prompt_len:]





if __name__ == "__main__":
    # model = load__model_from_checkpoint(MODEL_PATH)
    model = TransformerLM(10000, 256, 512, 4, 16, 1344, 10000, "cuda:0")
    load_checkpoint(MODEL_PATH, model)
    
    tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH)
    eos_token_id = tokenizer.reversed_vocab[EOS_TOKEN.encode()]

    prompt = input(">>> ")
    prompt_tokens = tokenizer.encode(prompt)

    generated_token_ids = generate(model, prompt_tokens, MAX_NEW_TOKENS, eos_token_id, TEMPERATURE, TOP_P)
    generated_text = tokenizer.decode(generated_token_ids)

    print(generated_text)


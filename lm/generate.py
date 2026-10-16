"""Matched decoding for every model: same prompts, same length, same temperature / top-k,
same random seed per prompt, and the same banned tokens (<pad>, optionally <unk>)."""
import numpy as np
import torch

import config


def _pick(logits: torch.Tensor, banned: list[int], greedy: bool, gen: torch.Generator) -> int:
    logits = logits.double().clone()
    logits[banned] = -float("inf")
    if greedy:
        return int(torch.argmax(logits))
    logits = logits / config.GEN_TEMPERATURE
    k = min(config.GEN_TOP_K, logits.numel())
    top_vals, top_idx = torch.topk(logits, k)
    probs = torch.softmax(top_vals, dim=-1)
    return int(top_idx[torch.multinomial(probs, 1, generator=gen)])


def _banned(vocab) -> list[int]:
    return [vocab.pad] + ([vocab.unk] if config.GEN_BAN_UNK else [])


@torch.no_grad()
def generate_neural(model, vocab, device, greedy: bool) -> list[dict]:
    model.eval()
    outs = []
    for i, prompt in enumerate(config.PROMPTS):
        gen = torch.Generator().manual_seed(config.GEN_SEED + i)
        ids = vocab.encode(prompt.split())
        for _ in range(config.GEN_TOKENS):
            x = torch.tensor([ids[-config.SEQ_LEN:]], device=device)
            logits = model(x)[0, -1].float().cpu()
            ids.append(_pick(logits, _banned(vocab), greedy, gen))
        outs.append({"prompt": prompt,
                     "continuation": vocab.decode(ids[len(prompt.split()):])})
    return outs


def generate_ngram(lm, vocab, greedy: bool) -> list[dict]:
    outs = []
    for i, prompt in enumerate(config.PROMPTS):
        gen = torch.Generator().manual_seed(config.GEN_SEED + i)
        ids = vocab.encode(prompt.split())
        for _ in range(config.GEN_TOKENS):
            logits = torch.from_numpy(np.log(lm.next_dist(ids)))
            ids.append(_pick(logits, _banned(vocab), greedy, gen))
        outs.append({"prompt": prompt,
                     "continuation": vocab.decode(ids[len(prompt.split()):])})
    return outs


def repetition_stats(texts: list[str]) -> dict:
    """Simple, model-agnostic repetition measures over generated continuations."""
    distinct1, distinct2, rep3 = [], [], []
    for t in texts:
        toks = t.split()
        if not toks:
            continue
        bi = list(zip(toks, toks[1:]))
        tri = list(zip(toks, toks[1:], toks[2:]))
        distinct1.append(len(set(toks)) / len(toks))
        distinct2.append(len(set(bi)) / max(len(bi), 1))
        rep3.append(1 - len(set(tri)) / max(len(tri), 1))
    mean = lambda xs: round(float(np.mean(xs)), 3) if xs else None
    return {"distinct_1": mean(distinct1), "distinct_2": mean(distinct2),
            "repeated_trigram_rate": mean(rep3)}

"""Part II input-embedding tables: own word2vec (trained here) and public GloVe-100d."""
import time
from collections import Counter

import numpy as np
import torch

import config
from lm import data
from lm.utils import results_dir, save_json


def w2v_path(smoke: bool):
    return results_dir(smoke) / "checkpoints" / "w2v100.kv"


def train_word2vec(smoke: bool = False) -> dict:
    """Skip-gram on the LM training split only (no extra data). Cost is reported separately."""
    from gensim.models import Word2Vec

    lines = data.read_split("train")
    if smoke:
        budget, kept = config.SMOKE["train_tokens"], []
        for line in lines:
            if budget <= 0:
                break
            kept.append(line)
            budget -= len(line)
        lines = kept
    params = dict(config.W2V)
    if smoke:
        params["epochs"] = config.SMOKE["w2v_epochs"]

    start = time.perf_counter()
    model = Word2Vec(sentences=lines, **params)
    seconds = time.perf_counter() - start

    model.wv.save(str(w2v_path(smoke)))
    stats = {"method": "skip-gram negative sampling (gensim)", "params": params,
             "train_seconds": round(seconds, 2), "sentences": len(lines),
             "tokens": int(sum(len(l) for l in lines)), "vectors": len(model.wv),
             "hardware": "CPU (gensim)"}
    save_json(stats, results_dir(smoke) / "runs" / "w2v_training.json")
    return stats


def build_table(kind: str, vocab, train_ids: np.ndarray, smoke: bool):
    """Returns a (V, EMB_DIM) table and coverage statistics.

    Words missing from the source get a vector drawn from N(0, sigma^2), where sigma is the
    standard deviation of the source vectors, using a fixed seed. <pad> is all zeros.
    The whole table (missing rows included) is then frozen.
    """
    from gensim.models import KeyedVectors

    if kind == "frozen_random":
        rng = np.random.default_rng(config.SEED)
        table = rng.normal(0.0, config.FROZEN_RANDOM_STD,
                           size=(len(vocab), config.EMB_DIM)).astype(np.float32)
        table[vocab.pad] = 0.0
        return torch.from_numpy(table), {
            "source": kind, "type_coverage": 0.0, "token_coverage": 0.0,
            "missing_types": len(vocab) - 1, "most_frequent_missing": [],
            "missing_init": f"every row N(0, {config.FROZEN_RANDOM_STD}^2), seed {config.SEED}; "
                            "<pad> = 0; whole table frozen (control: no pretraining)",
            "source_vector_std": config.FROZEN_RANDOM_STD}

    if kind == "w2v":
        kv = KeyedVectors.load(str(w2v_path(smoke)))
    elif kind == "glove":
        kv = KeyedVectors.load(str(data.GLOVE_PATH))
    else:
        raise ValueError(kind)
    if kv.vector_size != config.EMB_DIM:
        raise ValueError(f"{kind} vectors are {kv.vector_size}-d but EMB_DIM={config.EMB_DIM}")

    sigma = float(kv.vectors.std())
    rng = np.random.default_rng(config.SEED)
    table = rng.normal(0.0, sigma, size=(len(vocab), config.EMB_DIM)).astype(np.float32)
    table[vocab.pad] = 0.0

    freq = Counter(train_ids.tolist())
    found, missing = 0, []
    tok_found = tok_total = 0
    for i, w in enumerate(vocab.itos):
        if i == vocab.pad:
            continue
        tok_total += freq.get(i, 0)
        if w in kv.key_to_index:
            table[i] = kv[w]
            found += 1
            tok_found += freq.get(i, 0)
        else:
            missing.append((w, freq.get(i, 0)))
    missing.sort(key=lambda x: -x[1])
    coverage = {
        "source": kind,
        "type_coverage": round(found / (len(vocab) - 1), 4),
        "token_coverage": round(tok_found / max(tok_total, 1), 4),
        "missing_types": len(missing),
        "most_frequent_missing": missing[:25],
        "missing_init": f"N(0, {sigma:.4f}^2), seed {config.SEED}; <pad> = 0; whole table frozen",
        "source_vector_std": round(sigma, 4),
    }
    return torch.from_numpy(table), coverage

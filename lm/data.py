"""Data download, vocabulary, and the shared evaluation chunking.

Evaluation contract shared by EVERY language model (n-gram included):
  * the token stream of a split is cut into windows of SEQ_LEN + 1 tokens that overlap by one
    token, so every token after the first is a prediction target exactly once;
  * a model predicting the target at position j may only use the inputs of its own window that
    come before j (at most SEQ_LEN tokens of context, fewer near the start of a window);
  * the last window is right-padded with <pad>, and padded targets are excluded from the loss.
"""
import html
import json
import re
from collections import Counter

import numpy as np
import torch

import config

PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"
WT_DIR = config.DATA_DIR / "wikitext2"
AG_DIR = config.DATA_DIR / "agnews"
GLOVE_PATH = config.DATA_DIR / "glove100.kv"


# ------------------------------------------------------------------ download (run once per machine)
def prepare_data():
    from datasets import load_dataset
    import gensim.downloader as api

    WT_DIR.mkdir(parents=True, exist_ok=True)
    AG_DIR.mkdir(parents=True, exist_ok=True)

    name, cfg = config.WIKITEXT_HF
    wt = load_dataset(name, cfg)
    for split, fname in [("train", "train"), ("validation", "valid"), ("test", "test")]:
        with open(WT_DIR / f"{fname}.txt", "w", encoding="utf-8") as f:
            for row in wt[split]:
                f.write(row["text"].rstrip("\n") + "\n")
    print("WikiText-2 saved to", WT_DIR)

    ag = load_dataset(config.AGNEWS_HF)
    for split in ("train", "test"):
        with open(AG_DIR / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in ag[split]:
                f.write(json.dumps({"text": row["text"], "label": int(row["label"])}) + "\n")
    print("AG News saved to", AG_DIR)

    if not GLOVE_PATH.exists():
        kv = api.load(config.GLOVE_GENSIM)
        kv.save(str(GLOVE_PATH))
    print("GloVe saved to", GLOVE_PATH)


# ------------------------------------------------------------------ WikiText
def read_split(split: str) -> list[list[str]]:
    """One token list per non-empty line, lowercased, with <eos> appended."""
    lines = []
    with open(WT_DIR / f"{split}.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if config.LOWERCASE:
                line = line.lower()
            lines.append(line.split() + [EOS])
    return lines


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}
        self.pad, self.unk, self.eos = self.stoi[PAD], self.stoi[UNK], self.stoi[EOS]

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, lines: list[list[str]]):
        counts = Counter(tok for line in lines for tok in line)
        specials = [PAD, UNK, EOS]
        words = sorted((w for w in counts if w not in specials), key=lambda w: (-counts[w], w))
        return cls(specials + words)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, self.unk) for t in tokens]

    def decode(self, ids) -> str:
        return " ".join(self.itos[i] for i in ids)


def load_corpus(smoke: bool = False):
    """Returns vocab and flat id streams (np.int64) for train / valid / test."""
    raw = {s: read_split(s) for s in ("train", "valid", "test")}
    vocab = Vocab.build(raw["train"])
    streams = {}
    for s, lines in raw.items():
        ids = np.fromiter((i for line in lines for i in vocab.encode(line)), dtype=np.int64)
        if smoke:
            ids = ids[: config.SMOKE["train_tokens"] if s == "train" else config.SMOKE["eval_tokens"]]
        streams[s] = ids
    return vocab, streams


def corpus_stats(vocab, streams, raw_lines=None) -> dict:
    out = {"vocab_size": len(vocab)}
    for s, ids in streams.items():
        out[f"{s}_tokens"] = int(len(ids))
        out[f"{s}_unk_rate"] = float((ids == vocab.unk).mean())
    return out


def make_windows(ids: np.ndarray, seq_len: int, pad_id: int):
    """Cut a stream into (inputs, targets) windows; see the module docstring."""
    n_targets = len(ids) - 1
    n_win = (n_targets + seq_len - 1) // seq_len
    inputs = np.full((n_win, seq_len), pad_id, dtype=np.int64)
    targets = np.full((n_win, seq_len), pad_id, dtype=np.int64)
    for k in range(n_win):
        chunk = ids[k * seq_len: k * seq_len + seq_len + 1]
        L = len(chunk) - 1
        inputs[k, :L] = chunk[:-1]
        targets[k, :L] = chunk[1:]
    return torch.from_numpy(inputs), torch.from_numpy(targets)


# ------------------------------------------------------------------ AG News
_TOKEN_RE = re.compile(r"'s\b|n't\b|\w+?(?=n't\b)|\w+(?:-\w+)*|[^\w\s]")


def tokenize_news(text: str) -> list[str]:
    """Approximate WikiText's tokenisation so news words map onto the LM vocabulary."""
    text = html.unescape(text).replace("#39;", "'").replace("\\", " ")
    text = re.sub(r"&\w+;|<[^>]+>", " ", text)
    if config.LOWERCASE:
        text = text.lower()
    toks = []
    for t in _TOKEN_RE.findall(text):
        if "-" in t and t != "-":                    # well-known -> well @-@ known (WikiText style)
            parts = t.split("-")
            for i, p in enumerate(parts):
                if i:
                    toks.append("@-@")
                if p:
                    toks.append(p)
        else:
            toks.append(t)
    return toks


def load_agnews(split: str) -> list[dict]:
    with open(AG_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

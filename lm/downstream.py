"""Part III: frozen LM representations -> logistic regression on AG News topic classification.

Each text becomes one fixed-length vector in one of three ways (same frozen model, no fine-tuning):
  mean : average of the final-layer hidden states over real tokens (padding excluded)
  last : final-layer hidden state at the last real token (it has read the whole text)
  emb  : average of the input embeddings over real tokens (no context; a bag-of-words baseline)
Classifier: StandardScaler + multinomial logistic regression, C chosen on a validation split by
macro-F1, identical for every representation.
"""
import math
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from lm import data
from lm.models import build_model
from lm.utils import load_json, results_dir, save_json, set_seed


def _split_data(smoke: bool):
    rng = np.random.default_rng(config.SEED)
    train_all = data.load_agnews("train")
    test = data.load_agnews("test")
    n_tr = config.SMOKE["downstream_train_n"] if smoke else config.DOWNSTREAM_TRAIN_N
    n_va = config.SMOKE["downstream_val_n"] if smoke else config.DOWNSTREAM_VAL_N
    per_class_tr, per_class_va = n_tr // 4, n_va // 4
    train, val = [], []
    labels = np.array([r["label"] for r in train_all])
    for c in range(4):
        idx = rng.permutation(np.where(labels == c)[0])
        train += [train_all[i] for i in idx[:per_class_tr]]
        val += [train_all[i] for i in idx[per_class_tr:per_class_tr + per_class_va]]
    if smoke:
        test = [test[i] for i in rng.permutation(len(test))[: config.SMOKE["downstream_test_n"]]]
    return train, val, test


def _encode(rows, vocab):
    ids, unk, total, truncated = [], 0, 0, 0
    for r in rows:
        toks = data.tokenize_news(r["text"]) or [data.UNK]
        if len(toks) > config.DOWNSTREAM_MAX_LEN:
            truncated += 1
            toks = toks[: config.DOWNSTREAM_MAX_LEN]
        enc = vocab.encode(toks)
        unk += sum(i == vocab.unk for i in enc)
        total += len(enc)
        ids.append(enc)
    return ids, {"token_unk_rate": round(unk / total, 4), "mean_len": round(total / len(rows), 1),
                 "truncated_frac": round(truncated / len(rows), 4)}


@torch.no_grad()
def extract(model, seqs, pad_id, device, batch_size=256) -> dict:
    model.eval()
    feats = {p: [] for p in config.DOWNSTREAM_POOLING}
    order = np.argsort([len(s) for s in seqs])          # length-bucketed batches, restored after
    for i in range(0, len(seqs), batch_size):
        idx = order[i:i + batch_size]
        batch = [seqs[j] for j in idx]
        T = max(len(s) for s in batch)
        x = torch.full((len(batch), T), pad_id, dtype=torch.long)
        for k, s in enumerate(batch):
            x[k, :len(s)] = torch.tensor(s)
        x = x.to(device)
        mask = (x != pad_id).unsqueeze(-1).float()
        lengths = mask.sum(1).clamp(min=1)
        h = model.hidden_states(x).float()               # causal: padding never affects real tokens
        e = model.embedding(x).float()
        last_idx = (lengths.squeeze(-1).long() - 1)
        pooled = {
            "mean": (h * mask).sum(1) / lengths,
            "last": h[torch.arange(len(batch), device=device), last_idx],
            "emb": (e * mask).sum(1) / lengths,
        }
        for p in feats:
            feats[p].append((idx, pooled[p].cpu().numpy()))
    out = {}
    for p, parts in feats.items():
        dim = parts[0][1].shape[1]
        arr = np.zeros((len(seqs), dim), dtype=np.float32)
        for idx, f in parts:
            arr[idx] = f
        out[p] = arr
    return out


def _fit_eval(Xtr, ytr, Xva, yva, Xte, yte) -> dict:
    best = None
    for C in config.DOWNSTREAM_C_GRID:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=3000, random_state=config.SEED))
        clf.fit(Xtr, ytr)
        f1 = f1_score(yva, clf.predict(Xva), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    val_f1, C, clf = best
    pred = clf.predict(Xte)
    return {"C": C, "val_macro_f1": round(val_f1, 4),
            "test_accuracy": round(accuracy_score(yte, pred), 4),
            "test_macro_f1": round(f1_score(yte, pred, average="macro"), 4),
            "per_class_f1": [round(x, 4) for x in f1_score(yte, pred, average=None)],
            "confusion": confusion_matrix(yte, pred).tolist()}, pred


def run_downstream(smoke: bool = False, device=None) -> dict:
    set_seed()
    out_dir = results_dir(smoke)
    vocab, _ = data.load_corpus(smoke)
    train, val, test = _split_data(smoke)
    (tr_ids, tr_stats), (va_ids, _), (te_ids, te_stats) = (
        _encode(train, vocab), _encode(val, vocab), _encode(test, vocab))
    ytr = np.array([r["label"] for r in train])
    yva = np.array([r["label"] for r in val])
    yte = np.array([r["label"] for r in test])

    results, preds = {}, {}
    for name in config.DOWNSTREAM_MODELS:
        ckpt = out_dir / "checkpoints" / f"{name}.pt"
        run_json = out_dir / "runs" / f"{name}.json"
        if not ckpt.exists() or not run_json.exists():
            print(f"[downstream] skipping {name}: no trained checkpoint")
            continue
        run = load_json(run_json)
        model = build_model(run["arch"], len(vocab), vocab.pad)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(device)
        Ftr, Fva, Fte = (extract(model, s, vocab.pad, device) for s in (tr_ids, va_ids, te_ids))
        for pool in config.DOWNSTREAM_POOLING:
            key = f"{name}:{pool}"
            r, p = _fit_eval(Ftr[pool], ytr, Fva[pool], yva, Fte[pool], yte)
            r.update({"model": name, "pooling": pool, "dim": int(Ftr[pool].shape[1]),
                      "lm_test_ppl": run["test_ppl"]})
            results[key], preds[key] = r, p
            print(f"[downstream] {key:24s} acc {r['test_accuracy']:.4f}  "
                  f"macro-F1 {r['test_macro_f1']:.4f}  (C={r['C']})")

    errors = _error_analysis(test, yte, preds)
    summary = {"task": "AG News 4-way topic classification",
               "labels": config.AGNEWS_LABELS,
               "n_train": len(train), "n_val": len(val), "n_test": len(test),
               "train_text_stats": tr_stats, "test_text_stats": te_stats,
               "majority_baseline_acc": round(Counter(yte.tolist()).most_common(1)[0][1] / len(yte), 4),
               "results": results, "errors": errors}
    save_json(summary, out_dir / "downstream" / "summary.json")
    np.savez(out_dir / "downstream" / "predictions.npz", y_test=yte,
             **{k.replace(":", "__"): v for k, v in preds.items()})
    return summary


def _error_analysis(test, yte, preds, k=8) -> dict:
    """Examples where the two headline representations disagree, plus shared failures."""
    a, b = "lstm:mean", "transformer:mean"
    if a not in preds or b not in preds:
        return {}
    rng = np.random.default_rng(config.SEED)
    L = config.AGNEWS_LABELS
    pa, pb = preds[a], preds[b]

    def sample(mask):
        idx = np.where(mask)[0]
        idx = rng.permutation(idx)[:k]
        return [{"text": test[i]["text"][:300], "true": L[yte[i]],
                 a: L[pa[i]], b: L[pb[i]]} for i in idx]

    return {"compared": [a, b],
            "counts": {"only_" + a + "_right": int(((pa == yte) & (pb != yte)).sum()),
                       "only_" + b + "_right": int(((pb == yte) & (pa != yte)).sum()),
                       "both_wrong": int(((pa != yte) & (pb != yte)).sum())},
            "only_a_right": sample((pa == yte) & (pb != yte)),
            "only_b_right": sample((pb == yte) & (pa != yte)),
            "both_wrong": sample((pa != yte) & (pb != yte))}

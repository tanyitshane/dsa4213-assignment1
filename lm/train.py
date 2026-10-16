"""Training, evaluation, timing and generation for the n-gram and neural language models."""
import math
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
from lm import data, embeddings
from lm.generate import generate_neural, generate_ngram, repetition_stats
from lm.models import build_model
from lm.ngram import KNTrigram
from lm.utils import Timer, hardware_info, results_dir, save_json, set_seed, sync


def run_name(arch: str, emb: str) -> str:
    return arch if emb == "random" else f"{arch}_{emb}"


def _windows(vocab, streams):
    return {s: data.make_windows(ids, config.SEQ_LEN, vocab.pad) for s, ids in streams.items()}


@torch.no_grad()
def evaluate(model, inputs, targets, pad_id, device, batch_size=config.BATCH_SIZE) -> float:
    """Mean next-token cross-entropy (nats) over non-pad targets."""
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(inputs), batch_size):
        x = inputs[i:i + batch_size].to(device)
        y = targets[i:i + batch_size].to(device)
        logits = model(x)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1),
                                 ignore_index=pad_id, reduction="sum").item()
        count += (y != pad_id).sum().item()
    return total / count


def time_inference(fn, device, n_tokens: int) -> dict:
    """Common inference workload: score every test target once. One warm-up, then the median."""
    fn()
    times = []
    for _ in range(config.INFER_REPEATS):
        with Timer(device) as t:
            fn()
        times.append(t.seconds)
    med = statistics.median(times)
    return {"workload": "score all test targets once (batch 64, window 128)",
            "seconds_median": round(med, 4), "seconds_all": [round(x, 4) for x in times],
            "tokens": n_tokens, "tokens_per_second": round(n_tokens / med, 1)}


# ---------------------------------------------------------------------------------- n-gram
def run_ngram(smoke: bool = False, device=None) -> dict:
    set_seed()
    out_dir = results_dir(smoke)
    vocab, streams = data.load_corpus(smoke)
    win = _windows(vocab, streams)

    with Timer() as t_fit:
        lm = KNTrigram(len(vocab)).fit(streams["train"])

    def score(split):
        x, y = win[split]
        lp = lm.window_logprobs(x.numpy(), y.numpy(), vocab.pad)
        return float(-lp.mean()), len(lp)

    val_ce, _ = score("valid")
    test_ce, n_test = score("test")
    cpu = torch.device("cpu")
    timing = time_inference(lambda: score("test"), cpu, n_test)

    with Timer() as t_gen:
        samples = generate_ngram(lm, vocab, greedy=False)
    greedy = generate_ngram(lm, vocab, greedy=True)

    res = {
        "name": "ngram", "model": f"interpolated Kneser-Ney {config.NGRAM_ORDER}-gram",
        "discounts": {"D3": round(lm.D3, 4), "D2": round(lm.D2, 4)},
        "valid_ce": val_ce, "valid_ppl": math.exp(val_ce),
        "test_ce": test_ce, "test_ppl": math.exp(test_ce), "test_targets": n_test,
        "train_seconds": round(t_fit.seconds, 2),
        "size": {"stored_entries": lm.num_entries(), "size_mb": round(lm.nbytes() / 1e6, 2)},
        "inference": timing, "generation_seconds": round(t_gen.seconds, 3),
        "samples": samples, "greedy": greedy,
        "sample_repetition": repetition_stats([s["continuation"] for s in samples]),
        "greedy_repetition": repetition_stats([s["continuation"] for s in greedy]),
        "hardware": {**hardware_info(cpu), "note": "n-gram runs on CPU only"},
        "corpus": data.corpus_stats(vocab, streams),
    }
    save_json(res, out_dir / "runs" / "ngram.json")
    print(f"[ngram] valid ppl {res['valid_ppl']:.1f}  test ppl {res['test_ppl']:.1f}  "
          f"fit {res['train_seconds']}s")
    return res


# ---------------------------------------------------------------------------------- neural
def _lr_lambda(total_steps: int):
    def f(step):
        if step < config.WARMUP_STEPS:
            return (step + 1) / config.WARMUP_STEPS
        progress = (step - config.WARMUP_STEPS) / max(1, total_steps - config.WARMUP_STEPS)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return f


def run_neural(arch: str, emb: str = "random", smoke: bool = False, device=None) -> dict:
    set_seed()
    name = run_name(arch, emb)
    out_dir = results_dir(smoke)
    vocab, streams = data.load_corpus(smoke)
    win = _windows(vocab, streams)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_model(arch, len(vocab), vocab.pad)
    coverage = None
    if emb != "random":
        table, coverage = embeddings.build_table(emb, vocab, streams["train"], smoke)
        model.freeze_embeddings(table)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    epochs = config.SMOKE["epochs"] if smoke else config.EPOCHS
    x_tr, y_tr = win["train"]
    steps_per_epoch = math.ceil(len(x_tr) / config.BATCH_SIZE)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(epochs * steps_per_epoch))

    history = []
    best_val, best_epoch = float("inf"), -1
    ckpt = out_dir / "checkpoints" / f"{name}.pt"
    train_seconds = 0.0
    emb_before = model.embedding.weight.detach().clone()

    for epoch in range(1, epochs + 1):
        model.train()
        g = torch.Generator().manual_seed(config.SEED + epoch)
        order = torch.randperm(len(x_tr), generator=g)
        tot_loss, tot_tok, grad_norms, step_losses = 0.0, 0, [], []
        sync(device)
        t0 = time.perf_counter()
        for i in range(0, len(order), config.BATCH_SIZE):
            idx = order[i:i + config.BATCH_SIZE]
            x, y = x_tr[idx].to(device), y_tr[idx].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                   ignore_index=vocab.pad)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP)
            opt.step()
            sched.step()
            n = (y != vocab.pad).sum().item()
            tot_loss += loss.item() * n
            tot_tok += n
            grad_norms.append(float(gn))
            step_losses.append(loss.item())
        sync(device)
        epoch_seconds = time.perf_counter() - t0
        train_seconds += epoch_seconds

        val_ce = evaluate(model, *win["valid"], vocab.pad, device)
        train_ce = tot_loss / tot_tok
        if not math.isfinite(train_ce):
            print(f"[{name}] non-finite loss at epoch {epoch}; stopping")
            history.append({"epoch": epoch, "train_ce": train_ce, "valid_ce": val_ce})
            break
        history.append({
            "epoch": epoch, "train_ce": train_ce, "valid_ce": val_ce,
            "train_ppl": math.exp(train_ce), "valid_ppl": math.exp(val_ce),
            "epoch_seconds": round(epoch_seconds, 2), "lr_end": sched.get_last_lr()[0],
            "grad_norm_mean": float(np.mean(grad_norms)),
            "grad_norm_max": float(np.max(grad_norms)),
            "step_loss_std": float(np.std(step_losses[len(step_losses) // 2:])),
            "clipped_frac": float(np.mean(np.array(grad_norms) > config.GRAD_CLIP)),
        })
        if val_ce < best_val:
            best_val, best_epoch = val_ce, epoch
            torch.save(model.state_dict(), ckpt)
        print(f"[{name}] epoch {epoch:2d}  train ppl {math.exp(train_ce):8.1f}  "
              f"valid ppl {math.exp(val_ce):8.1f}  ({epoch_seconds:.1f}s)")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    emb_changed = bool((model.embedding.weight.detach() != emb_before.to(device)).any())
    test_ce = evaluate(model, *win["test"], vocab.pad, device)
    x_te, y_te = win["test"]
    n_test = int((y_te != vocab.pad).sum())
    timing = time_inference(lambda: evaluate(model, x_te, y_te, vocab.pad, device), device, n_test)

    with Timer(device) as t_gen:
        samples = generate_neural(model, vocab, device, greedy=False)
    greedy = generate_neural(model, vocab, device, greedy=True)

    res = {
        "name": name, "arch": arch, "embedding": emb,
        "embedding_trainable": emb == "random", "embedding_changed_during_training": emb_changed,
        "params": model.param_counts(),
        "config": {k: getattr(config, k) for k in
                   ["SEQ_LEN", "BATCH_SIZE", "EMB_DIM", "HIDDEN", "DROPOUT", "LR",
                    "WEIGHT_DECAY", "WARMUP_STEPS", "GRAD_CLIP", "SEED"]} | {"epochs": epochs},
        "best_epoch": best_epoch, "valid_ce": best_val, "valid_ppl": math.exp(best_val),
        "test_ce": test_ce, "test_ppl": math.exp(test_ce), "test_targets": n_test,
        "train_seconds": round(train_seconds, 2),
        "seconds_per_epoch": round(train_seconds / len(history), 2),
        "inference": timing, "generation_seconds": round(t_gen.seconds, 3),
        "history": history, "samples": samples, "greedy": greedy,
        "sample_repetition": repetition_stats([s["continuation"] for s in samples]),
        "greedy_repetition": repetition_stats([s["continuation"] for s in greedy]),
        "embedding_coverage": coverage,
        "hardware": hardware_info(device),
        "corpus": data.corpus_stats(vocab, streams),
    }
    save_json(res, out_dir / "runs" / f"{name}.json")
    print(f"[{name}] best epoch {best_epoch}  valid ppl {res['valid_ppl']:.1f}  "
          f"test ppl {res['test_ppl']:.1f}  train {train_seconds:.0f}s")
    return res

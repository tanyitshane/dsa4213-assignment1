"""Interpolated Kneser-Ney trigram language model (Chen & Goodman, 1999), vectorised with numpy.

P3(w|u,v) = max(c(uvw) - D3, 0) / c(uv.) + D3 * N1+(uv.) / c(uv.) * P2(w|v)
P2(w|v)   = max(N1+(.vw) - D2, 0) / N1+(.v.) + D2 * N1+(v.)' / N1+(.v.) * P1(w)
P1(w)     = (1 - eps) * N1+(.w) / N1+(..) + eps / V          (continuation unigram, uniform floor)

If a context was never seen, the model falls back to the next lower order.
Near the start of an evaluation window (fewer than 2 preceding tokens) the lower orders are used
directly, matching the context the neural models are allowed to see.
"""
import numpy as np

EPS = 1e-6


def _discount(counts: np.ndarray) -> float:
    n1, n2 = (counts == 1).sum(), (counts == 2).sum()
    return float(n1 / (n1 + 2 * n2)) if n1 + n2 > 0 else 0.5


def _lookup(sorted_keys: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_keys, query)
    idx = np.clip(idx, 0, len(sorted_keys) - 1)
    hit = sorted_keys[idx] == query
    return np.where(hit, values[idx], 0)


class KNTrigram:
    def __init__(self, vocab_size: int):
        self.V = vocab_size

    # ---------------------------------------------------------------- training
    def fit(self, ids: np.ndarray):
        V = self.V
        ids = ids.astype(np.int64)

        # trigram counts c(uvw) and their contexts (uv)
        k3 = (ids[:-2] * V + ids[1:-1]) * V + ids[2:]
        self.k3, self.c3 = np.unique(k3, return_counts=True)
        ctx3 = self.k3 // V
        self.ctx_keys, inv = np.unique(ctx3, return_inverse=True)
        self.ctx_count = np.bincount(inv, weights=self.c3).astype(np.int64)    # c(uv.)
        self.ctx_types = np.bincount(inv).astype(np.int64)                     # N1+(uv.)
        self.D3 = _discount(self.c3)

        # bigram continuation counts N1+(.vw): number of distinct u before (v, w)
        self.k2, self.n2 = np.unique(self.k3 % (V * V), return_counts=True)
        v_of_k2 = self.k2 // V
        self.n2_total = np.bincount(v_of_k2, weights=self.n2, minlength=V)   # N1+(.v.)
        self.n2_types = np.bincount(v_of_k2, minlength=V)                    # N1+(v.)'
        self.D2 = _discount(self.n2)

        # unigram continuation N1+(.w): number of distinct v before w
        bigram_types = np.unique(ids[:-1] * V + ids[1:])
        cont = np.bincount(bigram_types % V, minlength=V).astype(np.float64)
        self.p1 = (1 - EPS) * cont / cont.sum() + EPS / V
        return self

    def num_entries(self) -> int:
        return int(len(self.k3) + len(self.ctx_keys) + len(self.k2) + 2 * self.V)

    def nbytes(self) -> int:
        arrays = [self.k3, self.c3, self.ctx_keys, self.ctx_count, self.ctx_types, self.k2,
                  self.n2, self.n2_total, self.n2_types, self.p1]
        return int(sum(a.nbytes for a in arrays))

    # ---------------------------------------------------------------- scoring
    def _p2(self, v, w):
        V = self.V
        p1 = self.p1[w]
        denom = self.n2_total[v]
        n = _lookup(self.k2, self.n2, v * V + w)
        safe = np.maximum(denom, 1)
        p = np.maximum(n - self.D2, 0) / safe + self.D2 * self.n2_types[v] / safe * p1
        return np.where(denom > 0, p, p1)

    def _p3(self, u, v, w):
        V = self.V
        p2 = self._p2(v, w)
        ctx = u * V + v
        c_ctx = _lookup(self.ctx_keys, self.ctx_count, ctx)
        t_ctx = _lookup(self.ctx_keys, self.ctx_types, ctx)
        c = _lookup(self.k3, self.c3, ctx * V + w)
        safe = np.maximum(c_ctx, 1)
        p = np.maximum(c - self.D3, 0) / safe + self.D3 * t_ctx / safe * p2
        return np.where(c_ctx > 0, p, p2)

    def window_logprobs(self, inputs: np.ndarray, targets: np.ndarray, pad_id: int):
        """Log-probabilities for every non-pad target of the shared evaluation windows."""
        inputs, targets = inputs.astype(np.int64), targets.astype(np.int64)
        n_win, L = targets.shape
        pos = np.broadcast_to(np.arange(L), (n_win, L))
        v = inputs                                           # token right before the target
        u = np.concatenate([np.zeros((n_win, 1), np.int64), inputs[:, :-1]], axis=1)
        mask = targets != pad_id
        u, v, w, pos = u[mask], v[mask], targets[mask], pos[mask]

        p = np.empty(len(w))
        first = pos == 0                                     # only one token of context
        p[first] = self._p2(v[first], w[first])
        rest = ~first
        p[rest] = self._p3(u[rest], v[rest], w[rest])
        return np.log(p)

    # ---------------------------------------------------------------- generation
    def next_dist(self, context: list[int]) -> np.ndarray:
        """Full next-token distribution for a context (uses at most the last two tokens)."""
        V = self.V
        p = self.p1.copy()
        if len(context) >= 1:
            v = context[-1]
            denom = self.n2_total[v]
            if denom > 0:
                lo, hi = np.searchsorted(self.k2, [v * V, (v + 1) * V])
                p = self.D2 * self.n2_types[v] / denom * p
                ws = self.k2[lo:hi] % V
                p[ws] += np.maximum(self.n2[lo:hi] - self.D2, 0) / denom
        if len(context) >= 2:
            ctx = context[-2] * V + context[-1]
            j = np.searchsorted(self.ctx_keys, ctx)
            if j < len(self.ctx_keys) and self.ctx_keys[j] == ctx:
                c_ctx, t_ctx = self.ctx_count[j], self.ctx_types[j]
                lo, hi = np.searchsorted(self.k3, [ctx * V, (ctx + 1) * V])
                p = self.D3 * t_ctx / c_ctx * p
                ws = self.k3[lo:hi] % V
                p[ws] += np.maximum(self.c3[lo:hi] - self.D3, 0) / c_ctx
        return p

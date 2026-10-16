"""Neural next-token language models. All share the same input embedding (EMB_DIM),
hidden width (HIDDEN) and output layer (Linear HIDDEN -> V), so they differ only in the sequence
model in the middle. Every model only sees preceding tokens:
  * RNN / LSTM are left-to-right by construction;
  * the Transformer uses a causal (upper-triangular) attention mask.
"""
import torch
import torch.nn as nn

import config


class _Base(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.EMB_DIM, padding_idx=pad_id)
        self.drop = nn.Dropout(config.DROPOUT)
        self.out = nn.Linear(config.HIDDEN, vocab_size)

    def hidden_states(self, x: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T, HIDDEN)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, T) -> logits (B, T, V)
        return self.out(self.drop(self.hidden_states(x)))

    def freeze_embeddings(self, weights: torch.Tensor):
        """Load a pretrained table and keep the WHOLE table fixed (including OOV rows)."""
        with torch.no_grad():
            self.embedding.weight.copy_(weights)
        self.embedding.weight.requires_grad_(False)

    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb = self.embedding.weight.numel()
        out = sum(p.numel() for p in self.out.parameters())
        return {"total": total, "trainable": trainable, "embedding": emb, "output_layer": out,
                "core": total - emb - out, "size_mb_fp32": round(total * 4 / 1e6, 2)}


class RNNLM(_Base):
    def __init__(self, vocab_size, pad_id):
        super().__init__(vocab_size, pad_id)
        self.rnn = nn.RNN(config.EMB_DIM, config.HIDDEN, num_layers=config.RNN_LAYERS,
                          nonlinearity="tanh", batch_first=True,
                          dropout=config.DROPOUT if config.RNN_LAYERS > 1 else 0.0)

    def hidden_states(self, x):
        h, _ = self.rnn(self.drop(self.embedding(x)))
        return h


class LSTMLM(_Base):
    def __init__(self, vocab_size, pad_id):
        super().__init__(vocab_size, pad_id)
        self.rnn = nn.LSTM(config.EMB_DIM, config.HIDDEN, num_layers=config.LSTM_LAYERS,
                           batch_first=True,
                           dropout=config.DROPOUT if config.LSTM_LAYERS > 1 else 0.0)

    def hidden_states(self, x):
        h, _ = self.rnn(self.drop(self.embedding(x)))
        return h


class TransformerLM(_Base):
    """Pre-LayerNorm decoder-only Transformer with learned positions and a causal mask."""

    def __init__(self, vocab_size, pad_id):
        super().__init__(vocab_size, pad_id)
        d = config.HIDDEN
        self.in_proj = nn.Linear(config.EMB_DIM, d)      # 100-d embeddings -> d_model
        self.pos = nn.Embedding(config.SEQ_LEN, d)
        layer = nn.TransformerEncoderLayer(d, config.TF_HEADS, config.TF_FFN, config.DROPOUT,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, config.TF_LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def hidden_states(self, x):
        T = x.size(1)
        if T > config.SEQ_LEN:
            raise ValueError(f"sequence length {T} exceeds the trained window {config.SEQ_LEN}")
        pos = torch.arange(T, device=x.device)
        h = self.in_proj(self.embedding(x)) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.blocks(self.drop(h), mask=mask, is_causal=True)
        return self.norm(h)


ARCHS = {"rnn": RNNLM, "lstm": LSTMLM, "transformer": TransformerLM}


def build_model(arch: str, vocab_size: int, pad_id: int) -> _Base:
    return ARCHS[arch](vocab_size, pad_id)

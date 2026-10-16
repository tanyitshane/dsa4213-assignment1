# DSA4213 Assignment 1 — Language Models, Embeddings, and Representations

Comparison of an n-gram, RNN, LSTM and Transformer language model trained from scratch on
WikiText-2 (Part I); an ablation of the LSTM's input embeddings (Part II); and frozen LM
representations used for AG News topic classification (Part III).

## Setup

Python 3.11 or 3.12 recommended.

```bash
python -m venv venv
# macOS / Linux:  source venv/bin/activate
# Windows:        venv\Scripts\activate

# PyTorch first — the command depends on the machine:
pip install torch                                                    # macOS (Apple silicon, MPS)
pip install torch --index-url https://download.pytorch.org/whl/cu128 # Windows/Linux, NVIDIA RTX 50-series

pip install -r requirements.txt
```

## Data

Nothing is committed to this repository; `python run.py prep` downloads everything into `data/`:

| data | source | used for |
|---|---|---|
| WikiText-2 (`wikitext-2-v1`, tokenised, rare words already `<unk>`) | HuggingFace `Salesforce/wikitext` (Merity et al., 2016) | Parts I–II language modelling |
| AG News | HuggingFace `fancyzhx/ag_news` (Zhang et al., 2015) | Part III classification |
| GloVe 6B, 100-d | `gensim` downloader `glove-wiki-gigaword-100` (Pennington et al., 2014) | Part II public embeddings |

### Preprocessing

- **WikiText-2:** text is lowercased because GloVe 6B is lowercased. Empty lines are dropped, and an `<eos>` token is appended to every remaining line. The official train / validation / test splits are used unchanged. The vocabulary is every token type in the training split plus `<pad>`, `<unk>` and `<eos>`. Validation and test tokens outside it map to `<unk>`.
- **Shared evaluation windows (all models, including the n-gram):** each split's token stream is cut into windows of 129 tokens that overlap by one token. Each window yields 128 inputs and 128 next-token targets, and every token after the first is scored exactly once. A model predicting position *j* may use only the earlier tokens in its own window. The last window is right-padded, and `<pad>` targets are excluded from the loss (`ignore_index`). Perplexity is exp(mean cross-entropy over non-pad targets).
- **AG News:** HTML entities and tags are stripped, and the text is lowercased. It is then tokenised to approximate WikiText's conventions: `'s` and `n't` become separate tokens, and hyphens are written `@-@`. Each text is mapped onto the LM vocabulary, unknown words become `<unk>`, and texts are truncated to 128 tokens. The data is split as follows (seed 42):
  - training: a class-stratified subset of 10,000 examples from the official train split;
  - validation: another 2,000 examples from the official train split, used only to choose the regularisation strength C;
  - test: the full official test set (7,600 examples).

## Reproducing the results

```bash
python run.py prep          # download data (once per machine)
python run.py all           # every experiment below, then the report
```

or step by step:

```bash
python run.py ngram                              # Part I: interpolated Kneser-Ney trigram
python run.py train --model rnn                  # Part I (also reused in Part III)
python run.py train --model lstm                 # Part I and Part II "random, trainable" setting
python run.py train --model transformer          # Part I
python run.py w2v                                # Part II: own skip-gram embeddings (cost logged separately)
python run.py train --model lstm --emb w2v       # Part II: own embeddings, frozen
python run.py train --model lstm --emb glove     # Part II: GloVe, frozen
python run.py train --model lstm --emb frozen_random  # Part II control: frozen but not pretrained
python run.py downstream                         # Part III
python run.py report                             # figures + results/summary.md
```

Sensitivity runs use their own results folder, so they never overwrite the main results:

```bash
python run.py all --epochs 40 --tag ep40    # longer budget  -> results_ep40/
python run.py all --seed 7 --tag seed7      # second seed    -> results_seed7/
python run.py all --lr 2e-3 --tag lr2e3     # different learning rate
```

Add `--smoke` to any command for a quick debugging run on a small subset; it writes to `results_smoke/`. `--device cuda|mps|cpu` overrides automatic device selection.

### Outputs

- `results/runs/<model>.json`: perplexities, per-epoch train and validation loss, gradient-norm statistics, parameter counts, timings, generated text, embedding coverage and hardware.
- `results/runs/w2v_training.json`: the cost of training our own word2vec embeddings.
- `results/downstream/summary.json`: metrics, confusion matrices and error examples for every representation.
- `results/figures/*.png`: the report figures.
- `results/summary.md`: every table and figure in one place.
- Checkpoints are written to `results/checkpoints/` and are not committed.

## Configuration

All settings live in `config.py`; the values reported are the defaults there.

| setting | value |
|---|---|
| seed | 42 (Python, NumPy, PyTorch; word2vec `workers=1` for determinism) |
| context window | 128 tokens, training and evaluation |
| embedding dim | 100 (matches GloVe-100d), randomly initialised and trainable in Part I |
| hidden size | 256 (RNN/LSTM hidden, Transformer d_model) |
| RNN / LSTM | 2 layers, dropout 0.3 |
| Transformer | 2 pre-LN layers, 4 heads, FFN 512, learned positions, causal mask, dropout 0.3 |
| output layer | Linear(256 → V), identical for all neural models |
| optimiser | AdamW, lr 1e-3, weight decay 0.01, 300 warm-up steps then cosine decay to 10%, gradient clipping 1.0 |
| budget | 20 epochs, batch 64; the best-validation checkpoint is evaluated on test |
| n-gram | interpolated Kneser-Ney trigram, discounts n1/(n1+2n2) |
| word2vec | skip-gram, 100-d, window 5, 5 negatives, 5 epochs, min_count 1, training split only |
| frozen-random control | every row N(0, 0.4²) (≈ GloVe's scale), `<pad>` = 0, table frozen: separates "frozen" from "pretrained" |
| frozen settings | whole embedding table frozen; missing words initialised N(0, σ²), σ = std of the source vectors |
| generation | 5 fixed prompts, 40 tokens, temperature 0.8, top-k 40, seed 1234 + prompt index, `<unk>` banned; greedy also reported |
| timing | `cuda.synchronize()` around timed regions; inference = scoring all test targets once (batch 64), median of 3 passes after 1 warm-up |
| Part III classifier | StandardScaler + logistic regression, C ∈ {0.01, 0.1, 1, 10} chosen on validation macro-F1 |

## Reproducibility notes

- Results in the report were produced on: **[fill in: GPU, CPU, RAM, OS, PyTorch version]**. Each run JSON also records this.
- TF32 matrix multiplication is enabled on CUDA for speed. Together with non-deterministic cuDNN kernels, this means a rerun can differ in the later decimal places.
- Timings depend on hardware. Neural timings are on the GPU; the n-gram runs on the CPU (stated in the tables).

## Code layout

```
config.py          all hyperparameters and seeds
run.py             command-line entry point
lm/data.py         download, vocabulary, shared evaluation windows, AG News tokeniser
lm/ngram.py        interpolated Kneser-Ney trigram (numpy)
lm/models.py       RNN / LSTM / Transformer language models
lm/train.py        training loop, evaluation, timing, generation
lm/generate.py     matched decoding and repetition statistics
lm/embeddings.py   word2vec training, GloVe loading, coverage, frozen tables
lm/downstream.py   Part III feature extraction and classification
lm/report.py       figures and results/summary.md
```

## Attribution and tool use

- Libraries: PyTorch, gensim, scikit-learn, HuggingFace `datasets`, matplotlib.
- Datasets and embeddings: as cited in the table above.
- The code was written with assistance from Claude, Anthropic
- All ouputs and results were verified by me

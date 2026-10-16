"""All experiment settings in one place.

Every number reported in the write-up comes from these values. If you change one,
re-run the affected steps and say so in the report.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SMOKE_RESULTS_DIR = ROOT / "results_smoke"

SEED = 42              # overridden by --seed
TAG = ""               # overridden by --tag; results go to results_<TAG>/ so runs never overwrite each other

# ---------------------------------------------------------------- data
WIKITEXT_HF = ("Salesforce/wikitext", "wikitext-2-v1")   # tokenised version with <unk>
AGNEWS_HF = "fancyzhx/ag_news"
GLOVE_GENSIM = "glove-wiki-gigaword-100"                 # = GloVe 6B, 100d (Pennington et al., 2014)
LOWERCASE = True                                         # GloVe 6B is lowercased

# ---------------------------------------------------------------- shared LM setup
SEQ_LEN = 128          # context window for every neural model, in training and evaluation
BATCH_SIZE = 64
EMB_DIM = 100          # chosen to match GloVe-100d so Part I LSTM doubles as Part II's "random" setting
HIDDEN = 256           # hidden size for RNN / LSTM, d_model for the Transformer
DROPOUT = 0.3
EPOCHS = 20            # same fixed budget for every neural model; best-validation checkpoint is kept
LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 300
GRAD_CLIP = 1.0

RNN_LAYERS = 2
LSTM_LAYERS = 2
TF_LAYERS = 2          # 2 layers keeps Transformer non-embedding params close to the LSTM's
TF_HEADS = 4
TF_FFN = 512

# ---------------------------------------------------------------- n-gram
NGRAM_ORDER = 3        # interpolated Kneser-Ney trigram

# ---------------------------------------------------------------- word2vec (Part II, own embeddings)
W2V = dict(vector_size=EMB_DIM, window=5, min_count=1, sg=1, negative=5,
           epochs=5, workers=1, seed=SEED)   # workers=1 -> reproducible

FROZEN_RANDOM_STD = 0.4   # control for Part II: frozen random table, scale ~ GloVe-100d (std 0.41)

# ---------------------------------------------------------------- generation (matched across models)
PROMPTS = [
    "the city is located in",
    "in 1995 , the band released",
    "the game received",
    "he was born in",
    "during the war , the",
]
GEN_TOKENS = 40
GEN_TEMPERATURE = 0.8
GEN_TOP_K = 40
GEN_SEED = 1234
GEN_BAN_UNK = True     # applied identically to every model, n-gram included

# ---------------------------------------------------------------- timing
INFER_REPEATS = 3      # timed passes over the test set (after one warm-up pass); median is reported

# ---------------------------------------------------------------- Part III
DOWNSTREAM_TRAIN_N = 10000      # stratified subset of AG News train
DOWNSTREAM_VAL_N = 2000         # held out from the train split for choosing C
DOWNSTREAM_MAX_LEN = SEQ_LEN    # same window the LMs were trained on
DOWNSTREAM_C_GRID = [0.01, 0.1, 1.0, 10.0]
DOWNSTREAM_MODELS = ["rnn", "lstm", "transformer", "lstm_frozen_random", "lstm_w2v", "lstm_glove"]
DOWNSTREAM_POOLING = ["mean", "last", "emb"]   # emb = mean of input embeddings (no context)
AGNEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# ---------------------------------------------------------------- smoke test (debugging on a laptop)
SMOKE = dict(train_tokens=60000, eval_tokens=12000, epochs=2, downstream_train_n=400,
             downstream_val_n=100, downstream_test_n=300, w2v_epochs=1)

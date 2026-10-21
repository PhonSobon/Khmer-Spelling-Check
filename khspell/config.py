"""All paths and settings in one place.

Everything the training produces goes into ONE folder, outputs/ (created automatically):

    outputs/
      data/train.parquet, dev.parquet, test.parquet   cleaned Khmer sentences (step 2)
      lexicon.tsv                                     known words + frequencies (step 3)
      g2p.pt, pron.tsv                                pronunciation model (step 4)
      tagger.pt                                       character tagger (step 5)
      topk_model/                                     ByT5 spelling model (step 6)
      rank_weights.json                               tuned top-k ranking (step 7)
      main.log                                        log of every step and the results
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")   # HF_TOKEN: dataset downloads and uploading the models
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # harmless on Windows
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")         # slow networks: wait 2 min, not 10 s
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

# ---- output files -------------------------------------------------------------
OUT = Path(os.environ.get("KHSPELL_OUTPUTS", ROOT / "outputs"))
DATA_DIR = OUT / "data"
TRAIN_DATA = DATA_DIR / "train.parquet"
DEV_DATA = DATA_DIR / "dev.parquet"
TEST_DATA = DATA_DIR / "test.parquet"
LEXICON_TSV = OUT / "lexicon.tsv"
PRON_TSV = OUT / "pron.tsv"
G2P_CKPT = OUT / "g2p.pt"
TAGGER_CKPT = OUT / "tagger.pt"
TOPK_DIR = OUT / "topk_model"
RANK_WEIGHTS = OUT / "rank_weights.json"
LOG = OUT / "main.log"

# ---- datasets -----------------------------------------------------------------
DICT_ID = "seanghay/khmer-dictionary-44k"                  # official dictionary + pronunciations
REAL_TYPOS_ID = "seanghay/khmer-spelling-corrections"      # 917 real misspellings (test only)

# ---- lexicon ------------------------------------------------------------------
CORPUS_MIN_FREQ = 20          # a word not in the dictionary is trusted when it appears this often
CORPUS_NEIGHBOR_RATIO = 5.0   # ...unless it is a rare near-copy of a much more frequent dictionary word
SINGLE_CLUSTER_MIN_FREQ = 300 # one-cluster words (ជា, នៅ ...) only count as words when frequent

# ---- candidate search ---------------------------------------------------------
MAX_GRAPHEME_DIST = 2   # spelling edits (grapheme clusters) when searching the lexicon
MAX_PHONEME_DIST = 1    # sound edits when searching by pronunciation

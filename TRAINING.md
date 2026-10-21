# Training guide: Khmer spell checker on your desktop (RTX 3090)

The finished system finds misspelled Khmer words and gives the user the **top-3 corrections**:

```
ខ្ញុំជាសស្សដែលមានសក្សានៅកម្ពុច
  សស្ស   → 1. សិស្ស  2. ...  3. ...
  សក្សា  → 1. សិក្សា ...
  កម្ពុច → 1. កម្ពុជា ...
```

Everything is trained with **one command: `python main.py`**. It takes about **4-5 hours** on an RTX 3090. This is an estimate; the logs print the real speed as training runs.

---

## What gets trained

| # | Model / resource | What it does in the spell checker | Time (3090, est.) |
|---|---|---|---|
| 1 | Training data (~900k clean, Khmer-only sentences) | text to learn from; English, emoji and markup removed | 20-40 min |
| 2 | Lexicon (dictionary + corpus words) | keeps only **real words** in the suggestions | 10-20 min |
| 3 | **G2P** (Transformer, word → pronunciation) | finds **same-sound** spellings (ចំរៀង → ចម្រៀង); 100 epochs, best dev checkpoint kept | ~20-30 min |
| 4 | **Character tagger** (Transformer, per character) | **cuts words and marks the misspelled ones** | 40-80 min (Kaggle T4: ~4 h) |
| 5 | **ByT5-small** (seq2seq, word in context) | proposes corrections using the sentence context | 2-3 h |
| 6 | Ranking weights (tuned) + evaluation | combines everything into the final top-3 | 20-30 min |

How they work together when a user checks text:

```
text ─► ④ tagger: words + "is this word wrong?"
            │ only wrong words
            ▼
        candidates: ⑤ ByT5 beams + ② dictionary (edit distance) + ③ phoneme (G2P)
            │ keep real words only
            ▼
        rank: ByT5 probability + dictionary prior (weights from ⑥) ─► top-3 to the user
```

---

## 1. Copy the project to the desktop

Copy the `Khmer-Spelling-Check` folder **without** `outputs/` (or an old `artifacts/`); training creates `outputs/` itself. Put it on a local disk, e.g. `D:\Khmer-Spelling-Check`, **not inside OneDrive**: syncing multi-GB checkpoints slows training.

## 2. Install (once, ~30 min)

```powershell
nvidia-smi                                   # must list the RTX 3090
cd D:\Khmer-Spelling-Check
python -m venv .venv                         # Python 3.11 or 3.12, 64-bit
.\.venv\Scripts\Activate.ps1                 # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA build (see pytorch.org)
pip install -r requirements.txt
python scripts\check_gpu.py                  # expect: RTX 3090 | 24 GB | bf16: True
```

If `check_gpu.py` says *CUDA not available*, the CPU build of PyTorch is installed. Run `pip uninstall torch`, then repeat the `pip install torch ...` line.

## 3. Hugging Face token

Create `.env` in the project folder:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

The token needs **write** permission, because the models are uploaded to the public repo `phonsobon/khmer-spell-checker`.

## 4. Train

**First download the datasets** (about 1.8 GB, done only once). If the network drops, run the same command again; finished files are kept and partial ones resume:

```powershell
python scripts\prepare_data.py --download-only
```

**Then train:**

```powershell
python main.py --quick     # test the whole process first (~40 min, 10% data, small models)
python main.py             # full training (~4-5 h)
```

What you will see:

| Step | Watch for |
|---|---|
| 1/7 Check GPU | `RTX 3090 ... bf16: True`, then `OK: logged in as phonsobon` |
| 2/7 Build training data | per source: sentences kept and how many were dropped by each cleaning rule; 5 noisy/clean example pairs |
| 3/7 Build lexicon | `lexicon: ~38,000 words` |
| 4/7 Train G2P | `dev CER` going down; the paper reaches 6.6% |
| 5/7 Train character tagger | per epoch: `seg F1` (word cutting, aim > 0.95) and `error F1` (detection) |
| 6/7 Train ByT5 | `loss` falling from ~3 to below ~0.2; `eval_acc@1` every 2,000 steps, rising above 0.8 |
| 7/7 Tune + evaluate | the final results table (below) |

**If it stops** (error, restart, power cut): fix the cause and run **the same command again**. Finished steps are skipped, and ByT5 training continues from its last checkpoint (saved every 2,000 steps).

Other options:

```powershell
python main.py --serve       # after training: start the API (skips finished steps)
python main.py --no-push     # do not upload to Hugging Face
python main.py --retrain     # train all models again
python main.py --rebuild-data  # rebuild data + lexicon
```

## 5. Read the results (step 7)

The ranking weights are tuned first (`outputs/rank_weights.json`). Then three tables are printed:

1. **Real misspellings** (917 words users actually typed, `seanghay/khmer-spelling-corrections`, no context).
2. **Test words in context** (held-out sentences with injected errors).
3. **Sentences end to end** (tagger + everything).

Tables 1 and 2 are each shown twice: **"ByT5 beams only"** and **"FULL"**. FULL should be clearly better; that is the benefit of the dictionary and phoneme candidates.

| Metric | Meaning | Good value |
|---|---|---|
| `detected` | wrong words that get flagged | high |
| `acc@1` / `acc@3` | the right word is suggestion #1 / in the top 3 | the paper reports 80.4% / 91.9% |
| `MRR` | how high the right word is ranked on average | close to 1 |
| `false alarm` | correct words wrongly flagged | **low** (< 5%) |
| `detect precision` / `recall` | sentence level: flagged words that are really wrong / wrong words that get flagged | both high |
| `clean sentences flagged` | error-free sentences with any flag | low |

**Too many false alarms?** Raise the tagger threshold when checking, e.g. `"threshold": 0.7` in the API request. **Missing errors?** Lower it, e.g. `0.3`.

## 6. Use it

```powershell
python main.py --serve       # http://localhost:8000/docs
```

- `POST /spell` with `{"text": "ខ្ញុំជាសស្សដែលមានសក្សានៅកម្ពុច", "top_k": 3}` returns each wrong word, its position and the top-3 suggestions.
- `POST /suggest` with `{"word": "សំណើរ", "top_k": 3}` returns the top-3 for one word.

From Python:

```python
from khspell.topk import Speller
sp = Speller.load()                       # outputs/topk_model + outputs/tagger.pt + lexicon
text, errors = sp.check("ខ្ញុំជាសស្សដែលមានសក្សានៅកម្ពុច", k=3)
sp.suggest("សំណើរ", k=3)
```

## Output files (`outputs/`)

`outputs/` is simply where training saves its results. It is created automatically; you don't need it before training and should not copy it between computers.

| File | What |
|---|---|
| `data/` | cleaned Khmer sentences: train / dev / test |
| `topk_model/` | ByT5 model (+ `checkpoint-*` during training) |
| `tagger.pt` | character tagger |
| `g2p.pt`, `pron.tsv` | G2P model and pronunciations of all lexicon words |
| `lexicon.tsv` | lexicon with word frequencies |
| `rank_weights.json` | tuned ranking weights |
| `main.log` | full log of every step, including the results |

With uploading on, the ByT5 model is in the repo root of `phonsobon/khmer-spell-checker`, and the other files are in its `speller/` folder.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` in step 6 | `python scripts\train_topk.py --per-device-batch 32 --batch 64 --resume`, then `python main.py` |
| `loss = nan` | check `bf16: True` in step 1; lower the learning rate: `--lr 3e-4` |
| very slow (< 1 it/s in step 6) | check `nvidia-smi` shows python on the GPU; don't train inside OneDrive |
| `ConnectionError` / `WinError 10038` | temporary network error: run `python main.py` again |
| `Cannot write to phonsobon/...` | the token needs write permission, or use `--no-push` |
| `ModuleNotFoundError` | activate `.venv` and run from the project folder |

## Training on Kaggle (free GPU T4 x2)

Upload `kaggle_train.ipynb` to Kaggle (*Create → New Notebook → File → Import Notebook*), then:

1. **Settings:** Accelerator **GPU T4 x2**, Internet **on**.
2. **Hugging Face login** (a token with write permission):
   - interactive run: the login cell shows a `notebook_login()` box; log in, then run the next cells;
   - background run (*Save & Run All*): add **Add-ons → Secrets → `HF_TOKEN`** (no box can be filled there).

   `GITHUB_TOKEN` secret only if the GitHub repo is private.
3. Put your GitHub repo URL in the first code cell. Try `QUICK = True` once first.
4. **Save Version → Save & Run All** (runs in the background, up to 12 h).

T4 has no fast bf16, so ByT5 trains in fp32 on both GPUs (`torchrun`) and is slower than on a 3090.
Kaggle gives ~30 GPU hours per week and stops a session after 12 h, so training is built to resume.
Everything goes to the **public** model repo `phonsobon/khmer-spell-checker`:

| In the repo | What | When |
|---|---|---|
| `last-checkpoint/` | newest ByT5 checkpoint (model + optimizer) | every 2,000 steps |
| root, `speller/` | final ByT5 model; tagger, lexicon, G2P, ranking weights | at the end |
| `training-state/` | data, lexicon, G2P/tagger resume states (`*.resume`) | every 10 minutes |

**After any stop** (time limit, network error, crash) **run the notebook again**: it downloads
`training-state/`, finished steps are skipped, the G2P and the tagger continue from their `.resume`
state (the tagger even mid-epoch) and ByT5 continues from `last-checkpoint/`.
The same works on your PC: rerun `python main.py` after an interruption.

| Problem | Fix |
|---|---|
| `CUDA out of memory` in ByT5 | `PER_DEVICE_BATCH = 4` (default 8) |
| ByT5 hangs at the start with 2 GPUs | add `env["NCCL_P2P_DISABLE"] = "1"` in the train cell |
| `Repository not found` when cloning | private repo: add the `GITHUB_TOKEN` secret |

## After the first model

- **Real errors:** collect real misspellings (MPTC documents, user logs) and add them to training. Keep the 917 seanghay pairs for testing only.
- **Bigger model:** `python main.py --retrain --model google/byt5-base` (about 3x longer, usually a few points better).
- **Recurring mistakes:** if the results show an error type it keeps missing, add that pattern to `khspell/noise.py` and retrain.

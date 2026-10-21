# Khmer Spell Checker

Finds misspelled Khmer words and gives the **top-3 corrections**:

```
ខ្ញុំជាសស្សដែលមានសក្សានៅកម្ពុច
  សស្ស   → 1. សិស្ស  …
  សក្សា  → 1. សិក្សា …
  កម្ពុច → 1. កម្ពុជា …
```

**Train everything with one command: `python main.py`.** The desktop guide (setup, commands, reading the results) is in [TRAINING.md](TRAINING.md).

## How it works

```
user text
   │
   ▼
① Character tagger          cuts the text into words and marks the misspelled ones
   │  (only marked words)    (a misspelled word stays in one piece)
   ▼
② Candidates                 ByT5 model (uses the sentence context)
                             + dictionary words spelled alike     (សក្សា → សិក្សា)
                             + dictionary words sounding alike    (ចំរៀង → ចម្រៀង, via G2P)
   │  keep real words only
   ▼
③ Ranking                    ByT5 probability + dictionary prior (frequency, distance)
   │
   ▼
④ Top-3 suggestions          reported only if the best choice differs from the typed word
```

| Model | Architecture | Learns from |
|---|---|---|
| Character tagger | 6-layer bidirectional Transformer over characters, 2 outputs per character: word start, error | clean sentences with errors injected |
| ByT5 speller | pretrained `google/byt5-small` encoder-decoder (reads bytes), fine-tuned | words in context with errors injected |
| G2P | 3+3-layer Transformer encoder-decoder over characters | 44k dictionary word → pronunciation pairs |
| Ranking | 2 weights, tuned automatically | dev set |

The ideas follow Buoy et al., *Khmer Spellchecking: A Holistic Approach* (arXiv 2511.09812). They are extended with a trained ByT5 corrector, a combined word-cutting + error-detection tagger, and tuned ranking.

## Training data

Real Khmer typos are rare, so the models learn from **clean sentences with realistic errors injected on the fly** (`khspell/noise.py`). New errors are drawn every time a sentence is used:

| Error type | Example |
|---|---|
| dropped vowel | សិស្ស → សស្ស, កម្ពុជា → កម្ពុជ |
| similar vowel | ុ ↔ ូ, ិ ↔ ី, េ ↔ ែ |
| consonant of the other series | ជ ↔ ច, ត ↔ ទ, ប ↔ ព |
| extra final រ | សំណើ → សំណើរ |
| same-sound spelling | ចម្រៀង ↔ ចំរៀង |
| missing or extra mark | ់ ៍ ៉ ៊ ះ, dropped subscript ្ |
| **real typo** of that word | ប្រជាពលរដ្ឋ → ប្រជាពឡរដ្ធ, សិស្ស → ហិស្ស (~79k real typos of 7k words, mined from the human errors in `phonsobon/khmer-spellcheck-pairs-clean-v2` by `khspell/real_typos.py`) |

**Clean sentences** (`scripts/prepare_data.py`) come from about 1.2M sentences of news, Wikipedia, `nphearum/khmer-raw-text-3M-v2`, `phonsobon/khmer-lm-corpus`, the dictionary's examples and your own datasets. They are cleaned to **Khmer only** (`khspell/clean.py`):
- any sentence with English letters, other scripts, emoji or markup is dropped (about 20% of the raw text);
- keyboard noise (`ណាស់ស់ស់ស់`), duplicates and very short lines are removed.

**Evaluation** uses held-out sentences, plus **917 real misspellings** users actually typed (`seanghay/khmer-spelling-corrections`, never used for training).

## Files

```
main.py              runs every step (python main.py)
app.py               API: POST /spell, POST /suggest
khspell/             the spell checker code
  topk.py              Speller: the complete checker (and the ByT5 model wrapper)
  tagger.py            character tagger
  candidates.py        dictionary candidates (spelling + sound)
  g2p.py, phonetics.py pronunciation model and phone keys
  lexicon.py           known words, compound words
  noise.py             error generator
  clean.py             Khmer-only cleaning
  normalize.py         Khmer Unicode normalisation, character clusters
  distance.py, symdel.py   edit distance, fast similar-word search
  hf.py, config.py     downloads, paths and settings
scripts/             one script per step (main.py calls them)
outputs/             created by training: data, models, log (not needed before training)
```

## Use it after training

```powershell
python main.py --serve        # http://localhost:8000/docs
```

```python
from khspell.topk import Speller
sp = Speller.load()
text, errors = sp.check("ខ្ញុំជាសស្សដែលមានសក្សានៅកម្ពុច", k=3)
sp.suggest("សំណើរ", k=3)
```

## Citation

If you use this project, please cite:

```bibtex
@misc{khmerspell2026,
  author = {Phon Sobon},
  title = {Khmer Spell Checker: Error Detection and Top-k Correction for Khmer},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PhonSobon/Khmer-Spelling-Check}},
  note = {Open source Khmer spell checker: character tagger for word segmentation and error detection, ByT5 corrector, Transformer G2P for same-sound spellings, trained on real and synthetic misspellings}
}
```

## Credits

**Author:** Phon Sobon, MPTC AI ([Hugging Face: phonsobon](https://huggingface.co/phonsobon), khemthay0118@gmail.com)

**Data and research this project builds on:**
- *Khmer Dictionary 2022*, Royal Academy of Cambodia (RAC), via `seanghay/khmer-dictionary-44k` (research use only)
- Real misspellings: `seanghay/khmer-spelling-corrections`, `S-Sethisak/khmer-orthography-correction-dataset`
- Text: `nphearum/khmer-raw-text-3M-v2`, `CADT-IDRI/Khmer_News_classification_unsegmented`, `navinnseab/khmer-wikipedia-article`
- Word segmentation: [khmer-nltk](https://github.com/VietHoang1512/khmer-nltk)
- Method: Buoy et al., *Khmer Spellchecking: A Holistic Approach* (arXiv 2511.09812)
- Base model: [`google/byt5-small`](https://huggingface.co/google/byt5-small)

"""Step 2: collect clean, Khmer-only sentences -> outputs/data/{train,dev,test}.parquet

Cleaning (khspell/clean.py): Unicode normalisation, markup/URL removal, and only
sentences made of Khmer characters, digits and basic punctuation are kept: any
English word, other script or emoji drops the sentence. Duplicates are removed.

    python scripts/prepare_data.py                  # default caps (~900k sentences)
    python scripts/prepare_data.py --scale 0.1      # 10% (quick test)
    python scripts/prepare_data.py --download-only  # only download the datasets

Phases: [1] download all datasets first (parallel, cached), [2] read the local files and
clean + word-split them on all CPU cores, stopping each source at its cap, [3] mine real
misspellings (khspell/real_typos.py -> real_typos.tsv, used by the error generator), [4] split.

train.parquet keeps the clean sentence + its word tokens; errors are injected on the
fly during training, so every epoch sees new misspellings.
dev/test get fixed noisy sources (seeded) so scores are comparable between runs.
"""
import argparse
import collections
import logging
import os
import random
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from khspell import config, hf, real_typos
from khspell.clean import clean_sentence
from khspell.noise import noisy_sentence
from khspell.normalize import normalize

OUT = config.DATA_DIR

# (dataset id, text column, split, max sentences). Diverse general text first; your own
# (templated, MPTC-domain) corpora get a small share so the model knows that domain too.
SOURCES = [
    ("nphearum/khmer-raw-text-3M-v2", "text", "train", 450_000),
    ("phonsobon/khmer-lm-corpus", "text", "train", 350_000),          # news, Wikipedia, web (cleaned)
    ("CADT-IDRI/Khmer_News_classification_unsegmented", "content", "train", 150_000),
    ("navinnseab/khmer-wikipedia-article", "content", "train", 120_000),
    ("seanghay/khmer-dictionary-44k", "definition", "train", 40_000),
    ("seanghay/khmer-dictionary-44k", "example", "train", 25_000),
    ("S-Sethisak/khmer-orthography-correction-dataset", "target_text", "train", 20_000),
    ("phonsobon/khmer_auto_complete_v2", "text", "train", 15_000),
    ("phonsobon/khmer_auto_complete_v4", "text", "train", 15_000),
    ("phonsobon/khmer-word-segmentation", "sentence", "train", 10_000),
    ("phonsobon/Text_Emotional_kh", "text", "train", 10_000),
    # government news (13 ministry websites, Khmer only); its first 200k rows are balanced across the websites
    ("phonsobon/khmer-gov-news", "text", "train", 200_000),
]
if os.environ.get("KHSPELL_TEST_SOURCES"):      # tiny sources for automated tests only
    SOURCES = [("seanghay/khmer-dictionary-44k", "example", "train", 1500),
               ("CADT-IDRI/Khmer_News_classification_unsegmented", "content", "train", 1500),
               ("phonsobon/khmer-word-segmentation", "sentence", "train", 300)]
MAX_CHARS = 120
MIN_CHARS = 8

_SENT_END = re.compile(r"(?<=[។៕!?])\s*|\n+")
_WIKI_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_JUNK = re.compile(r"\{\{.*?\}\}|<[^>]+>|https?://\S+|\[\[|\]\]|'''|''")
_BAD_CHARS = re.compile(r"[=|{}\[\]<>@#_\\/*]")
_SEG_SPACE = re.compile(r"(?<=[ក-៿]) (?=[ក-៿])")


def chunk(sent):
    """Split long sentences at spaces (Khmer phrase breaks) into <= MAX_CHARS pieces."""
    if len(sent) <= MAX_CHARS:
        return [sent]
    out, cur = [], ""
    for part in sent.split(" "):
        if cur and len(cur) + 1 + len(part) > MAX_CHARS:
            out.append(cur)
            cur = part
        else:
            cur = f"{cur} {part}" if cur else part
    if cur:
        out.append(cur)
    return [c for c in out if len(c) <= MAX_CHARS]


def sentences(text, segmented=False, stats=None):
    """Clean Khmer-only sentences from one document; `stats` counts why pieces were dropped."""
    text = _JUNK.sub(" ", _WIKI_LINK.sub(r"\1", text))
    if segmented:
        text = _SEG_SPACE.sub("", text)
    for s in _SENT_END.split(text):
        s = normalize(s)
        if s.count(" ") > len(s) / 12:          # word-segmented text: restore natural spacing
            s = _SEG_SPACE.sub("", s)
        if len(s) < MIN_CHARS:
            continue
        if _BAD_CHARS.search(s):
            if stats is not None:
                stats["markup/code"] += 1
            continue
        for c in chunk(s):
            c, reason = clean_sentence(c)
            if stats is not None:
                stats[reason] += 1
            if c and len(c) >= MIN_CHARS:
                yield c


SEGMENTED = {"phonsobon/khmer_auto_complete_v4", "phonsobon/khmer-word-segmentation"}


def _init_worker():
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)            # khmer-nltk logs "Loaded model" in every process
    from khmernltk import word_tokenize  # noqa: F401  (load the CRF model once per process)


def clean_and_tokenize(task):
    """Worker: documents -> [(sentence, word tokens)] plus counts of what was dropped."""
    from khmernltk import word_tokenize

    docs, segmented = task
    stats, out = collections.Counter(), []
    for doc in docs:
        for s in sentences(doc, segmented, stats):
            toks = word_tokenize(s)
            if "".join(toks) == s:
                out.append((s, toks))
            else:
                stats["tokenizer mismatch"] += 1
    return out, stats


def doc_tasks(texts, segmented, chars_per_task=200_000):
    """Group documents into tasks of roughly equal size for the process pool."""
    batch, size = [], 0
    for t in texts:
        batch.append(t)
        size += len(t)
        if size >= chars_per_task:
            yield batch, segmented
            batch, size = [], 0
    if batch:
        yield batch, segmented


def download_all(parallel=2, attempts=6):
    """Phase 1: download every source first, so cleaning never waits on the network.

    Few parallel downloads (default 2) so each gets enough bandwidth on a slow line;
    failed downloads are retried with growing waits and resume where they stopped.
    """
    from concurrent.futures import ThreadPoolExecutor

    ids = sorted({(ds_id, split) for ds_id, _, split, _ in SOURCES}
                 | {(real_typos.REAL_PAIRS_ID, real_typos.REAL_PAIRS_SPLIT)})
    try:
        import hf_xet  # noqa: F401
    except ImportError:
        print("  note: 'pip install hf_xet' makes these downloads faster and more reliable", flush=True)
    print(f"[1/4] downloading {len(ids)} datasets, {parallel} at a time (cached after the first run) ...",
          flush=True)

    def fetch(item):
        ds_id, split = item
        for attempt in range(1, attempts + 1):
            try:
                return item, hf.download(ds_id, split)
            except Exception as e:
                if attempt == attempts:
                    raise RuntimeError(
                        f"could not download {ds_id} after {attempts} tries ({e}). This is a network "
                        f"problem: check the connection and run the same command again - finished "
                        f"downloads are kept and partial ones resume.") from e
                wait = 15 * attempt
                print(f"  {ds_id}: network error ({type(e).__name__}), retry {attempt}/{attempts - 1} "
                      f"in {wait}s", flush=True)
                time.sleep(wait)

    with ThreadPoolExecutor(parallel) as ex:
        paths = dict(ex.map(fetch, ids))
    for (ds_id, _), files in sorted(paths.items()):
        mb = sum(os.path.getsize(f) for f in files) / 1e6
        print(f"  ok  {ds_id}: {len(files)} file(s), {mb:,.0f} MB", flush=True)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every source cap")
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--download-only", action="store_true", help="only download the datasets, then stop")
    ap.add_argument("--parallel-downloads", type=int, default=2)
    args = ap.parse_args()
    logging.getLogger("khmer-nltk").setLevel(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    paths = download_all(args.parallel_downloads)
    if args.download_only:
        print("download finished.")
        return

    print(f"\n[2/4] cleaning + word splitting on {args.workers} processes ...", flush=True)
    seen, rows = set(), []
    with Pool(args.workers, initializer=_init_worker) as pool:
        for ds_id, col, split, cap in SOURCES:
            cap = int(cap * args.scale)
            kept, stats = 0, collections.Counter()
            tasks = doc_tasks(hf.iter_local_text(paths[(ds_id, split)], col), ds_id in SEGMENTED)
            bar = tqdm(total=cap, desc=f"  {ds_id.split('/')[-1]}[{col}]", mininterval=5)
            while kept < cap:
                wave = [t for _, t in zip(range(args.workers * 2), tasks)]   # bounded: never reads ahead far
                if not wave:
                    break
                for out, st in pool.map(clean_and_tokenize, wave):
                    stats.update(st)
                    for s, toks in out:
                        if kept >= cap:
                            break
                        if s in seen:
                            stats["duplicate"] += 1
                            continue
                        seen.add(s)
                        rows.append((s, toks, ds_id))
                        kept += 1
                        bar.update()
            bar.close()
            dropped = ", ".join(f"{r} {n:,}" for r, n in stats.most_common() if r != "ok")
            print(f"  {ds_id}[{col}]: {kept:,} kept | dropped: {dropped or 'none'}", flush=True)

    print("\n[3/4] mining real misspellings from human-corrected sentences ...", flush=True)
    real_paths = paths[(real_typos.REAL_PAIRS_ID, real_typos.REAL_PAIRS_SPLIT)]
    pairs = zip(hf.iter_local_text(real_paths, "source"), hf.iter_local_text(real_paths, "target"))
    eval_typos = hf.load(config.REAL_TYPOS_ID)["misspelling"]      # keep the evaluation set unseen
    typos = real_typos.build(pairs, exclude=eval_typos, real_words=hf.load(config.DICT_ID)["word"])
    real_typos.save(typos)
    print(f"  {sum(typos.values()):,} real typos of {len({w for w, _ in typos}):,} words -> {real_typos.REAL_TYPOS_TSV}")
    for (w, t), n in typos.most_common(8):
        print(f"    {w} -> {t}  ({n}x)")

    print("\n[4/4] splitting train / dev / test ...", flush=True)
    rng = random.Random(42)
    rng.shuffle(rows)
    df = pd.DataFrame(rows, columns=["text", "tokens", "source"])
    held = df.iloc[: args.n_dev + args.n_test].copy()
    train = df.iloc[args.n_dev + args.n_test:]
    held["noisy"] = [noisy_sentence(t, rng) for t in held["tokens"]]
    dev, test = held.iloc[: args.n_dev], held.iloc[args.n_dev:]
    train.to_parquet(OUT / "train.parquet", index=False)
    dev[["noisy", "text", "source"]].to_parquet(OUT / "dev.parquet", index=False)
    test[["noisy", "text", "source"]].to_parquet(OUT / "test.parquet", index=False)
    print(f"train {len(train):,}  dev {len(dev):,}  test {len(test):,}  -> {OUT}  ({(time.time() - t0) / 60:.1f} min)")
    print(df.source.value_counts().to_string())
    for _, r in test.head(5).iterrows():
        print(f"  noisy: {r.noisy}\n  clean: {r.text}")


if __name__ == "__main__":
    main()

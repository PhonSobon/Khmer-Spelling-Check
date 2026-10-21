"""Real misspellings mined from human-written (wrong, correct) sentence pairs.

The sentence pairs of phonsobon/khmer-spellcheck-pairs-clean-v2 (split test_real: real
human errors) are aligned character by character; every correct word whose typed form
differs gives one (correct word -> typo) pair. khspell.noise then uses these real typos
next to its rule-based errors, so the models also learn errors the rules never make
(ប្រជាពលរដ្ឋ -> ប្រជាពលរដ្ធ, ស -> ហ, ម -> ព ...).

The 917 evaluation misspellings (seanghay/khmer-spelling-corrections) are excluded.
"""
import collections
import csv
import difflib

from . import config
from .normalize import clusters, normalize

REAL_PAIRS_ID = "phonsobon/khmer-spellcheck-pairs-clean-v2"
REAL_PAIRS_SPLIT = "test_real"
REAL_TYPOS_TSV = config.DATA_DIR / "real_typos.tsv"

_table = None


def _is_khmer_word(t):
    return bool(t) and all("ក" <= ch <= "៓" or ch in "ៜ៝" for ch in t)


def word_typos(source, target, tokenize):
    """[(correct word, typo)] from one sentence pair (spaces are ignored: real text spaces freely)."""
    s, t = normalize(source).replace(" ", ""), normalize(target).replace(" ", "")
    if not s or not t or s == t:
        return []
    # src_at[j]: position in s that matches word boundary j of t (None inside an edited block)
    src_at = [None] * (len(t) + 1)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s, t, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1 + 1):
                src_at[j1 + k] = i1 + k
        else:
            src_at[j1], src_at[j2] = i1, i2
    out, pos = [], 0
    for word in tokenize(t):
        a, b = pos, pos + len(word)
        pos = b
        if src_at[a] is None or src_at[b] is None:
            continue                          # the word boundary itself was edited: ambiguous
        typo = s[src_at[a]:src_at[b]]
        if typo == word or not _is_khmer_word(word) or not _is_khmer_word(typo):
            continue
        n = len(clusters(word))
        if n < 2 or abs(len(clusters(typo)) - n) > 1:
            continue                          # one-cluster words / split or merged words: too noisy
        if difflib.SequenceMatcher(None, typo, word, autojunk=False).ratio() < 0.6:
            continue                          # a different word, not a misspelling
        out.append((word, typo))
    return out


def _well_formed(word):
    """Every character cluster starts with a consonant or independent vowel (no stray ិ ្ ...)."""
    return all("ក" <= c[0] <= "ឳ" for c in clusters(word))


def build(pairs, exclude=(), real_words=(), min_count=1):
    """pairs: iterable of (source, target). Returns Counter{(correct, typo): count}.

    exclude: typos never to use (the evaluation set); real_words: correct words (dictionary),
    a "typo" that is itself a word (មាន -> បាន) is a real-word error, not a misspelling.
    """
    from khmernltk import word_tokenize

    exclude = {normalize(w) for w in exclude} | {normalize(w) for w in real_words}
    counts = collections.Counter()
    for src, tgt in pairs:
        for word, typo in word_typos(src, tgt, word_tokenize):
            if typo not in exclude and typo.replace("ឲ", "ឱ") != word and _well_formed(typo):
                counts[(word, typo)] += 1
    return collections.Counter({k: n for k, n in counts.items() if n >= min_count})


def save(counts, path=REAL_TYPOS_TSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["correct", "typo", "count"])
        for (word, typo), n in counts.most_common():
            w.writerow([word, typo, n])


def table(path=REAL_TYPOS_TSV):
    """{correct word: ([typos], [counts])}, loaded once; empty if the file was not built."""
    global _table
    if _table is None:
        _table = {}
        if path.exists():
            with open(path, encoding="utf-8", newline="") as f:
                rows = csv.reader(f, delimiter="\t")
                next(rows)
                for word, typo, n in rows:
                    typos, counts = _table.setdefault(word, ([], []))
                    typos.append(typo)
                    counts.append(int(n))
    return _table

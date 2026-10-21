"""Synthetic Khmer spelling errors (for evaluation, weight tuning and noise-robust training)."""
import random
import re

from . import real_typos
from .normalize import clusters, normalize

VOWEL_SWAPS = ["ុូ", "ិី", "ឹឺ", "េែ", "ែៃ", "ោៅ", "ៀឿ", "ួូ", "ះៈ"]
CONS_SWAPS = ["តទ", "កគ", "បព", "ចជ", "ណន", "លឡ", "ថធ", "ដឌ", "ផភ", "ខឃ", "ឆឈ", "សឝ", "រល",
              # frequent in real human errors (phonsobon/khmer-spellcheck-pairs-clean-v2)
              "ហស", "មព", "មប", "យញ", "ញជ", "បវ", "តដ", "តឌ", "ទដ", "ឡរ", "ឋធថ", "ឍធ"]
MARKS_ADD = ["់", "៍", "៉", "៊", "ះ"]
CONS = "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ"


def _swap_char(word, pairs, rng):
    opts = [(i, p) for i, ch in enumerate(word) for p in pairs if ch in p]
    if not opts:
        return None
    i, p = rng.choice(opts)
    other = p.replace(word[i], "")
    return word[:i] + rng.choice(other) + word[i + 1:]


def op_vowel(w, rng):
    return _swap_char(w, VOWEL_SWAPS, rng)


def op_consonant(w, rng):
    return _swap_char(w, CONS_SWAPS, rng)


def op_final_ro(w, rng):          # សំណើ -> សំណើរ, ដៃគូ -> ដៃគូរ (very common in real data)
    return None if w.endswith("រ") else w + "រ"


def op_add_mark(w, rng):
    cl = clusters(w)
    i = rng.randrange(len(cl))
    cl[i] += rng.choice(MARKS_ADD)
    return "".join(cl)


def op_drop_mark(w, rng):
    idx = [i for i, ch in enumerate(w) if ch in "់៍៉៊ះៈំ"]
    if not idx:
        return None
    i = rng.choice(idx)
    return w[:i] + w[i + 1:]


def op_drop_vowel(w, rng):        # សិស្ស -> សស្ស, កម្ពុជា -> កម្ពុជ
    idx = [i for i, ch in enumerate(w) if "ា" <= ch <= "ៅ"]
    if not idx:
        return None
    i = rng.choice(idx)
    return w[:i] + w[i + 1:]


def op_nikahit(w, rng):
    """Same-sound respelling: Xំ + C <-> X + ម្C   (បំលែង <-> បម្លែង, ចំរៀង <-> ចម្រៀង)"""
    a = re.sub(r"ំ([ក-អ])", r"ម្\1", w, count=1)
    if a != w:
        return a
    b = re.sub(r"(?<=[ក-អ])ម្([ក-អ])", r"ំ\1", w, count=1)
    return b if b != w else None


def op_drop_coeng(w, rng):
    idx = [i for i, ch in enumerate(w) if ch == "្"]
    if not idx:
        return None
    i = rng.choice(idx)
    return w[:i] + w[i + 1:]


def op_delete_cluster(w, rng):
    cl = clusters(w)
    if len(cl) < 3:
        return None
    del cl[rng.randrange(len(cl))]
    return "".join(cl)


def op_real(w, rng):              # a typo people really made for this word (khspell.real_typos)
    found = real_typos.table().get(w)
    if not found:
        return None
    return rng.choices(*found)[0]


def op_transpose(w, rng):
    cl = clusters(w)
    if len(cl) < 2:
        return None
    i = rng.randrange(len(cl) - 1)
    cl[i], cl[i + 1] = cl[i + 1], cl[i]
    return "".join(cl)


def op_insert_cons(w, rng):
    cl = clusters(w)
    cl.insert(rng.randrange(len(cl) + 1), rng.choice(CONS))
    return "".join(cl)


OPS = [
    (op_vowel, 3.0), (op_consonant, 3.0), (op_drop_vowel, 2.0), (op_final_ro, 1.0), (op_add_mark, 1.0),
    (op_drop_mark, 1.5),
    (op_nikahit, 1.5), (op_drop_coeng, 1.0), (op_delete_cluster, 0.7), (op_transpose, 0.5), (op_insert_cons, 0.5),
    (op_real, 4.0),
]


def corrupt(word, rng=random, is_valid=None, n_ops=None, tries=20):
    """Return a misspelling of `word` that `is_valid` rejects (a non-word), or None."""
    ops, weights = zip(*OPS)
    for _ in range(tries):
        k = n_ops or (1 if rng.random() < 0.8 else 2)
        w = word
        for _ in range(k):
            out = rng.choices(ops, weights)[0](w, rng)
            if out:
                w = out
        w = normalize(w)
        if w != word and (is_valid is None or not is_valid(w)):
            return w
    return None


# ---------------------------------------------------------------- sentences
N_ERRORS = [(0, 0.15), (1, 0.35), (2, 0.25), (3, 0.15), (4, 0.10)]   # errors per sentence


def noisy_tokens(tokens, rng=random, n_errors=None):
    """Corrupt some words of a tokenised sentence.

    Returns (noisy_tokens, flags) where flags[i] is True for corrupted words.
    About 15% of sentences are left clean.
    """
    if n_errors is None:
        n_errors = rng.choices(*zip(*N_ERRORS))[0]
    toks = list(tokens)
    flags = [False] * len(toks)
    idx = [i for i, t in enumerate(toks) if is_khmer_token(t)]
    rng.shuffle(idx)
    done = 0
    for i in idx:
        if done >= n_errors:
            break
        bad = corrupt(toks[i], rng, tries=5)
        if bad:
            toks[i] = bad
            flags[i] = True
            done += 1
    return toks, flags


def noisy_sentence(tokens, rng=random, n_errors=None):
    """Corrupt a word-tokenised sentence (tokens join back to the clean text); returns the noisy text."""
    toks, _ = noisy_tokens(tokens, rng, n_errors)
    text = "".join(toks)
    if rng.random() < 0.05 and " " in text:             # occasional spacing noise
        j = rng.choice([k for k, ch in enumerate(text) if ch == " "])
        text = text[:j] + text[j + 1:]
    return text


def is_khmer_token(t):
    return bool(re.fullmatch(r"[\u1780-\u17D3\u17DC\u17DD]+", t))

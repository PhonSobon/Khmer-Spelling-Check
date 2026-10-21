"""Khmer Unicode normalisation and grapheme-cluster (KCC) splitting.

Normalisation fixes the "invisible" differences that make two identical-looking
words compare unequal: zero-width characters, mark order inside a cluster,
decomposed vowels and deprecated code points.
"""
import re
import unicodedata

COENG = "្"
RO = "រ"

_REMOVE = dict.fromkeys(map(ord, "​‌‍⁠﻿­឴឵"), None)
_REPLACE = {
    " ": " ",
    "ឣ": "អ",          # deprecated ឣ -> អ
    "ឤ": "អា",    # deprecated ឤ -> អា
    "៘": "។លះ",  # deprecated ៘ -> ។ល។
}
# split vowels typed as two code points
_COMPOSE = [("េី", "ើ"), ("េា", "ោ")]


def _is_cons(ch):
    return "ក" <= ch <= "ឳ"


def _mark_rank(unit):
    """Canonical order of the units that follow the base character of a cluster."""
    ch = unit[0]
    if ch == COENG:
        return 1.5 if unit[1] == RO else 1   # subscript RO goes after other subscripts
    if ch in "៉៊":                   # register shifters ៉ ៊
        return 2
    if ch == "៌":                        # robat ៌
        return 2
    if "ា" <= ch <= "ៅ" or ch == "័":  # dependent vowels, samyok sannya
        return 3
    return 4                                  # ំ ះ ៈ ់ ៍ ៎ ៏ ៑ ...


_CLUSTER_RE = re.compile(
    r"[ក-ឳៜ](?:្[ក-ឳ])*[ា-៑៓៝]*"
    r"|[០-៩]+|[0-9]+|[A-Za-z]+|\s+|.",
    re.S,
)


def _units(cluster):
    """Split the tail of a cluster into sortable units (coeng+consonant stays together)."""
    out, i = [], 1
    while i < len(cluster):
        if cluster[i] == COENG and i + 1 < len(cluster) and _is_cons(cluster[i + 1]):
            out.append(cluster[i:i + 2])
            i += 2
        else:
            out.append(cluster[i])
            i += 1
    return out


def _reorder(text):
    # Loose cluster: base + any mix of coeng pairs / marks, in any typed order.
    loose = re.compile(r"[ក-ឳៜ](?:្[ក-ឳ]|[ា-៑៓៝])*")

    def fix(m):
        c = m.group(0)
        units = _units(c)
        units = sorted(units, key=_mark_rank)
        dedup = []
        for u in units:               # collapse accidental double marks (ុុ -> ុ)
            if not dedup or u != dedup[-1] or u[0] == COENG:
                dedup.append(u)
        return c[0] + "".join(dedup)

    return loose.sub(fix, text)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_REMOVE)
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    for a, b in _COMPOSE:
        text = text.replace(a, b)
    text = _reorder(text)
    for a, b in _COMPOSE:             # reordering can expose new split vowels
        text = text.replace(a, b)
    return re.sub(r"[ \t]+", " ", text).strip()


def clusters(text: str):
    """Khmer character clusters: ['ប', 'ម្លែ', 'ង'] for បម្លែង."""
    return _CLUSTER_RE.findall(text)


def match_key(word: str) -> str:
    """Looser key for lexicon membership: subscript ដ and ត are accepted interchangeably."""
    return word.replace(COENG + "ដ", COENG + "ត")


KHMER_WORD_RE = re.compile(r"^[ក-៓ៜ៝]+$")


def is_khmer_word(s: str) -> bool:
    return bool(KHMER_WORD_RE.match(s))

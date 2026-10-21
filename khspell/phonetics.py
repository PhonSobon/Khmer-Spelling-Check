"""Pronunciation -> phone tokens.

The dictionary writes pronunciations in Khmer script, e.g. បម្លែង -> [បំ-ឡែង].
We turn such a respelling into a sequence of series-neutral phone tokens, so
that spellings which sound the same get the same (or a very close) key:

    បម្លែង [បំ-ឡែង]  -> b ំ l ែ ŋ
    បំលែង  [បំ-លែង]  -> b ំ l ែ ŋ
"""
import re

from .normalize import clusters, normalize

# consonant -> sound class; consonant pairs that differ only by series are merged
CONS = {
    "ក": "k", "គ": "k", "ខ": "kh", "ឃ": "kh", "ង": "ŋ",
    "ច": "c", "ជ": "c", "ឆ": "ch", "ឈ": "ch", "ញ": "ɲ",
    "ដ": "d", "ឌ": "d", "ឋ": "th", "ឍ": "th", "ណ": "n",
    "ត": "t", "ទ": "t", "ថ": "th", "ធ": "th", "ន": "n",
    "ប": "b", "ព": "p", "ផ": "ph", "ភ": "ph", "ម": "m",
    "យ": "j", "រ": "r", "ល": "l", "ឡ": "l", "វ": "v",
    "ឝ": "s", "ឞ": "s", "ស": "s", "ហ": "h", "អ": "ʔ",
}
INDEP = {
    "ឥ": "ʔ ិ", "ឦ": "ʔ ី", "ឧ": "ʔ ុ", "ឩ": "ʔ ុ", "ឪ": "ʔ ៅ", "ឫ": "r ឹ", "ឬ": "r ឺ",
    "ឭ": "l ឹ", "ឮ": "l ឺ", "ឯ": "ʔ េ", "ឰ": "ʔ ៃ", "ឱ": "ʔ ោ", "ឲ": "ʔ ោ", "ឳ": "ʔ ៅ",
}
SILENT_MARKS = set("៉៊់៌៍៎៏័៑")   # register shifters, bantoc, robat, ...
MARK_TOKEN = {"ះ": "h", "ៈ": "ʔ"}


def cluster_phones(cl):
    """Phone tokens for one grapheme cluster (series-neutral)."""
    if "៍" in cl:                      # toandakhiat: the letter is silent
        return []
    base, rest = cl[0], cl[1:]
    toks = []
    if base in CONS:
        toks.append(CONS[base])
    elif base in INDEP:
        toks.extend(INDEP[base].split())
    else:
        return []
    i = 0
    while i < len(rest):
        ch = rest[i]
        if ch == "្" and i + 1 < len(rest):
            sub = rest[i + 1]
            toks.append(CONS.get(sub, sub))
            i += 2
            continue
        if ch in MARK_TOKEN:
            toks.append(MARK_TOKEN[ch])
        elif ch not in SILENT_MARKS:
            toks.append(ch)            # vowel signs and ំ are kept as-is
        i += 1
    return toks


def split_alternatives(pro: str):
    """'[កាំ-ម៉ៈ-វិ-ធី]ឬ[កាំ-វិ-ធី]' -> ['កាំ-ម៉ៈ-វិ-ធី', 'កាំ-វិ-ធី']"""
    alts = re.findall(r"\[([^\[\]]+)\]", pro)
    if not alts:
        alts = [pro]
    out = []
    for a in alts:
        a = re.sub(r"\(.*?\)", "", a)
        a = normalize(a).replace(" ", "").replace("—", "-")
        if a.strip("-"):
            out.append(a.strip("-"))
    return out


def phones(pron: str):
    """Pronunciation respelling (syllables joined by '-') -> phone token list."""
    toks = []
    for syl in pron.split("-"):
        cls = clusters(syl)
        if len(cls) > 1 and cls[-1] in ("រ", "រ៍"):
            cls = cls[:-1]             # final រ is silent: សំណើរ sounds like សំណើ, ហួរ like ហួ
        for cl in cls:
            toks.extend(cluster_phones(cl))
    return toks


def grapheme_phones(word: str):
    """Rough fallback when no pronunciation / G2P model is available."""
    return phones(word)

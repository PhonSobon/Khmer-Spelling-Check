"""Khmer-only cleaning of training sentences.

A sentence is kept only if it is (almost) pure Khmer. Allowed characters:
Khmer letters/signs/punctuation/digits, spaces, Arabic digits and a few basic
punctuation marks. Anything else (English words, other scripts, emoji, symbols)
drops the whole sentence, because a spell checker for Khmer should learn from
Khmer text only.

    clean_sentence("បានប្តឹងក្រុមហ៊ុន Netflix ...")  -> (None, "latin")
    clean_sentence('"វាប្រែថាគាត់ចង់ស្តាប់សំឡេង។')  -> ("វាប្រែថាគាត់ចង់ស្តាប់សំឡេង។", "ok")
"""
import re

MIN_KHMER_CHARS = 8          # at least this many Khmer characters
MIN_KHMER_RATIO = 0.6        # Khmer letters among non-space characters (drops number lists)

_QUOTES = re.compile(r"[\"'`“”„‟«»‘’‚‹›]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
# everything that is allowed; any other character rejects the sentence
_ALLOWED = re.compile(r"^[ក-៿᧠-᧿ 0-9.,!?:;%()\-–]+$")
_REPEAT_PUNCT = re.compile(r"([.,!?:;\-–។៕៖])\1+")
_REPEAT_CHAR = re.compile(r"(.{1,3})\1{3,}")     # a character or cluster 4+ times: ស់ស់ស់ស់, ហាហាហាហា
_SPACES = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?:;។៕៖])")
_EDGE_JUNK = re.compile(r"^[\s\-–.,:;]+|[\s\-–,:;]+$")
_LIST_NUMBER = re.compile(r"^\s*(?:[0-9០-៩]+|[ក-អ])\s*[.)]\s+")   # "14. ", "១) ", "ក. "
_WIKI_META = re.compile(r"^[ក-៿]+:\S")        # ចំណាត់ថ្នាក់ក្រុម:..., ឯកសារ:..., ទំព័រគំរូ:...


def clean_sentence(s):
    """Returns (cleaned sentence, "ok") or (None, reason it was dropped)."""
    if _WIKI_META.match(s.strip()):
        return None, "wiki category/meta"
    s = _LIST_NUMBER.sub("", s)
    s = _QUOTES.sub("", s)
    s = _REPEAT_PUNCT.sub(r"\1", s)
    s = _SPACE_BEFORE_PUNCT.sub(r"\1", _SPACES.sub(" ", s))
    s = _EDGE_JUNK.sub("", s).strip()
    if not s:
        return None, "empty"
    if _LATIN.search(s):
        return None, "english/latin"
    if not _ALLOWED.match(s):
        return None, "other script/symbol"
    if s.count("(") != s.count(")"):
        return None, "unbalanced brackets"
    if _REPEAT_CHAR.search(s):
        return None, "repeated characters"
    chars = [c for c in s if c != " "]
    khmer = sum("ក" <= c <= "៝" for c in chars)
    if khmer < MIN_KHMER_CHARS:
        return None, "too short"
    if khmer / len(chars) < MIN_KHMER_RATIO:
        return None, "mostly digits/punctuation"
    return s, "ok"

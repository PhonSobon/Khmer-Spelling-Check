"""Weighted Damerau-Levenshtein distance with Khmer-specific confusion costs."""

# Pairs that are commonly confused (same sound in the other consonant series,
# similar-looking vowels, ...). Substituting within a group is cheap.
CONFUSION_GROUPS = [
    "ុូួ", "ិី", "ឹឺ", "េែៃ", "ោៅ", "ៀឿ",
    "កគ", "ខឃ", "ចជ", "ឆឈ", "ដឌ", "ថធឋឍ", "តទ", "បព", "ផភ", "ណន", "លឡ",
    "សឝឞ", "៉៊", "រល", "ឫឬរ", "ឭឮល", "ឥឦ", "ឧឩ", "ឱឲ", "៌័",
    "ះៈ",                      # look alike: សារៈ -> សារះ, ចំពោះ -> ចំពោៈ (frequent in real errors)
]
# Marks that are easy to forget or add by accident.
LIGHT_MARKS = set("់៉៊៍ៈះំ៌័្រ")

SUB_CHEAP = 0.5
INDEL_LIGHT = 0.5

_group_of = {}
for gi, g in enumerate(CONFUSION_GROUPS):
    for ch in g:
        _group_of.setdefault(ch, set()).add(gi)


def sub_cost(a, b):
    if a == b:
        return 0.0
    if _group_of.get(a, set()) & _group_of.get(b, set()):
        return SUB_CHEAP
    return 1.0


def indel_cost(ch):
    return INDEL_LIGHT if ch in LIGHT_MARKS else 1.0


def weighted_distance(s, t, max_dist=None):
    """Optimal-string-alignment distance with confusion-aware costs (chars or token lists)."""
    n, m = len(s), len(t)
    if n == 0:
        return float(sum(indel_cost(c) for c in t))
    if m == 0:
        return float(sum(indel_cost(c) for c in s))
    prev2 = None
    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + indel_cost(t[j - 1])
    for i in range(1, n + 1):
        cur = [prev[0] + indel_cost(s[i - 1])] + [0.0] * m
        si = s[i - 1]
        for j in range(1, m + 1):
            tj = t[j - 1]
            best = min(prev[j] + indel_cost(si), cur[j - 1] + indel_cost(tj), prev[j - 1] + sub_cost(si, tj))
            if i > 1 and j > 1 and si == t[j - 2] and s[i - 2] == tj:
                best = min(best, prev2[j - 2] + 1.0)
            cur[j] = best
        if max_dist is not None and min(cur) > max_dist:
            return float(max_dist) + 1.0
        prev2, prev = prev, cur
    return prev[m]


def levenshtein(a, b):
    """Plain unit-cost edit distance for sequences (used on phone tokens)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

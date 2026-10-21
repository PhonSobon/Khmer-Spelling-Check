"""Symmetric-delete (SymSpell-style) index over arbitrary unit sequences.

Units (grapheme clusters or phone tokens) are mapped to single private-use
characters so each word becomes a short string; deletes are then generated on
those strings. Lookup returns every indexed item within `max_dist` unit edits.
"""
from itertools import combinations

from .distance import levenshtein

_PUA_START = 0xF0000  # supplementary private use area: plenty of room


class UnitCodec:
    def __init__(self):
        self.code = {}

    def encode(self, units, grow=True):
        out = []
        for u in units:
            c = self.code.get(u)
            if c is None:
                if not grow:
                    c = "�"  # unknown unit: never matches an indexed one
                else:
                    c = chr(_PUA_START + len(self.code))
                    self.code[u] = c
            out.append(c)
        return "".join(out)


def _deletes(s, max_dist):
    out = {s}
    for d in range(1, min(max_dist, len(s) - 1) + 1):
        for idx in combinations(range(len(s)), d):
            out.add("".join(ch for i, ch in enumerate(s) if i not in idx))
    return out


class SymDeleteIndex:
    def __init__(self, max_dist=2, short_len=2):
        self.max_dist = max_dist
        self.short_len = short_len     # items this short only get distance 1
        self.codec = UnitCodec()
        self.index = {}                # delete-string -> set(item ids)
        self.items = []                # id -> (encoded, payload)

    def _dist_for(self, n):
        return 1 if n <= self.short_len else self.max_dist

    def add(self, units, payload):
        enc = self.codec.encode(units)
        iid = len(self.items)
        self.items.append((enc, payload))
        for d in _deletes(enc, self._dist_for(len(enc))):
            self.index.setdefault(d, []).append(iid)

    def lookup(self, units, max_dist=None):
        """Returns [(payload, unit_distance)] sorted by distance."""
        enc = self.codec.encode(units, grow=False)
        md = self._dist_for(len(enc)) if max_dist is None else max_dist
        seen, out = set(), []
        for d in _deletes(enc, md):
            for iid in self.index.get(d, ()):
                if iid in seen:
                    continue
                seen.add(iid)
                item_enc, payload = self.items[iid]
                if abs(len(item_enc) - len(enc)) > md:
                    continue
                dist = levenshtein(enc, item_enc)
                if dist <= md:
                    out.append((payload, dist))
        out.sort(key=lambda x: x[1])
        return out

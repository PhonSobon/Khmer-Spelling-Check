"""Lexicon: dictionary words + trusted corpus words, with frequencies."""
import csv
import math

from . import config
from .normalize import clusters, match_key


class Lexicon:
    def __init__(self, entries):
        """entries: iterable of (word, freq, in_dict)."""
        self.entries = {}          # match_key -> (word, freq, in_dict)
        for word, freq, in_dict in entries:
            k = match_key(word)
            old = self.entries.get(k)
            if old is None or (in_dict, freq) > (old[2], old[1]):
                self.entries[k] = (word, int(freq), bool(in_dict))
        self.total = sum(e[1] for e in self.entries.values()) + len(self.entries)
        self.max_clusters = max(len(clusters(e[0])) for e in self.entries.values())

    @classmethod
    def load(cls, path=config.LEXICON_TSV):
        with open(path, encoding="utf-8", newline="") as f:
            rows = csv.reader(f, delimiter="\t")
            next(rows)
            return cls((w, int(fr), d == "1") for w, fr, d in rows)

    def save(self, path=config.LEXICON_TSV):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["word", "freq", "in_dict"])
            for word, freq, in_dict in sorted(self.entries.values(), key=lambda e: -e[1]):
                w.writerow([word, freq, int(in_dict)])

    def __contains__(self, word):
        return match_key(word) in self.entries

    def __len__(self):
        return len(self.entries)

    def get(self, word):
        return self.entries.get(match_key(word))

    def freq(self, word):
        e = self.get(word)
        return e[1] if e else 0

    def in_dict(self, word):
        e = self.get(word)
        return bool(e and e[2])

    def logp(self, word):
        return math.log((self.freq(word) + 1) / self.total)

    def words(self):
        return [e[0] for e in self.entries.values()]

    def is_segment_word(self, piece, n_clusters=None):
        """Can `piece` stand as a word during segmentation / compound splitting?"""
        e = self.get(piece)
        if e is None:
            return False
        n = n_clusters if n_clusters is not None else len(clusters(piece))
        return n > 1 or e[1] >= config.SINGLE_CLUSTER_MIN_FREQ

    def compound_split(self, word, max_parts=4):
        """Split an out-of-lexicon word into known words (e.g. អ្នក+ផឹក), or None."""
        cl = clusters(word)
        n = len(cl)
        best = [None] * (n + 1)      # best[i] = (parts, -logp) for prefix of i clusters
        best[0] = ([], 0.0)
        for i in range(n):
            if best[i] is None or len(best[i][0]) >= max_parts:
                continue
            for j in range(i + 1, min(n, i + self.max_clusters) + 1):
                piece = "".join(cl[i:j])
                if not self.is_segment_word(piece, j - i):
                    continue
                cost = best[i][1] - self.logp(piece)
                if best[j] is None or cost < best[j][1]:
                    best[j] = (best[i][0] + [piece], cost)
        if best[n] is None or len(best[n][0]) < 2:
            return None
        return best[n][0]

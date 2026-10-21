"""Dictionary candidates for a misspelled word.

Two searches over the lexicon:
  * spelling: words within 2 grapheme-cluster edits           សក្សា  -> សិក្សា
  * sound:    words whose pronunciation (G2P) is within 1 edit ចំរៀង  -> ចម្រៀងplus corrected compounds (សាលារៀង -> សាលា + រៀន).
"""
import csv
import math

from . import config
from .distance import levenshtein, weighted_distance
from .lexicon import Lexicon
from .normalize import clusters
from .phonetics import grapheme_phones, phones
from .symdel import SymDeleteIndex


class CandidateGenerator:
    def __init__(self, lexicon, pron=None, g2p=None):
        self.lex, self.g2p = lexicon, g2p
        pron = pron or {}
        self.spell_index = SymDeleteIndex(max_dist=config.MAX_GRAPHEME_DIST)
        self.sound_index = SymDeleteIndex(max_dist=config.MAX_PHONEME_DIST, short_len=0)
        self.word_phones = {}
        for w in lexicon.words():
            self.spell_index.add(clusters(w), w)
            keys = {tuple(phones(p)) for p in pron.get(w, [])} or {tuple(grapheme_phones(w))}
            keys.discard(())
            self.word_phones[w] = list(keys)
            for k in keys:
                self.sound_index.add(k, w)

    @classmethod
    def load(cls):
        """Lexicon + pronunciations + G2P model from outputs/ (G2P and pronunciations are optional)."""
        pron = {}
        if config.PRON_TSV.exists():
            with open(config.PRON_TSV, encoding="utf-8", newline="") as f:
                for row in csv.reader(f, delimiter="\t"):
                    if len(row) == 2:
                        pron.setdefault(row[0], []).append(row[1])
        g2p = None
        if config.G2P_CKPT.exists():
            from .g2p import G2P
            g2p = G2P.load(config.G2P_CKPT)
        return cls(Lexicon.load(), pron, g2p)

    # ------------------------------------------------------------------ word facts
    def is_known(self, word):
        """A lexicon word, or a compound of lexicon words (អ្នក + ផឹក)."""
        return word in self.lex or self.lex.compound_split(word) is not None

    def freq(self, word):
        f = self.lex.freq(word)
        if f == 0 and word not in self.lex:          # a compound: use its rarest part
            parts = self.lex.compound_split(word) or []
            f = min((self.lex.freq(p) for p in parts), default=0)
        return f

    def in_dict(self, word):
        if self.lex.in_dict(word):
            return True
        if word in self.lex:
            return False
        parts = self.lex.compound_split(word)
        return bool(parts) and all(self.lex.in_dict(p) for p in parts)

    def prior(self, typed, cand):
        """How plausible `cand` is as the intended word, without context:
        0.3*log(frequency) + in official dictionary - spelling distance."""
        return 0.3 * math.log(self.freq(cand) + 1) + float(self.in_dict(cand)) - weighted_distance(typed, cand,
                                                                                                  max_dist=4)

    # ------------------------------------------------------------------ search
    def _phones_of(self, word):
        if self.g2p is not None:
            return [tuple(phones(self.g2p([word])[0]))]
        return [tuple(grapheme_phones(word))]

    def candidates(self, word, limit=20):
        """Up to `limit` real words that `word` may have been meant as, best first."""
        cl = clusters(word)
        found = {w for w, _ in self.spell_index.lookup(cl)}
        typed_phones = self._phones_of(word)
        for key in typed_phones:
            found |= {w for w, _ in self.sound_index.lookup(key)}
        if len(cl) >= 3:     # compounds: fix one part, keep the other (សាលារៀង -> សាលារៀន)
            for cut in range(1, len(cl)):
                left, right = "".join(cl[:cut]), "".join(cl[cut:])
                if self.lex.is_segment_word(left):
                    found |= {left + w for w, _ in self.spell_index.lookup(clusters(right), max_dist=1)}
                if self.lex.is_segment_word(right):
                    found |= {w + right for w, _ in self.spell_index.lookup(clusters(left), max_dist=1)}
        found.discard(word)

        def score(w):     # sounds-alike words count as close even when spelled very differently
            sound = min(levenshtein(a, b) for a in typed_phones for b in (self.word_phones.get(w) or
                                                                           [tuple(grapheme_phones(w))]))
            dist = min(weighted_distance(word, w, max_dist=4), sound + 0.25)
            return 0.3 * math.log(self.freq(w) + 1) + float(self.in_dict(w)) - 3.0 * dist

        return sorted(sorted(found), key=score, reverse=True)[:limit]   # ties: same order every run

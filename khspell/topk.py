"""The Khmer spell checker: find misspelled words and return the top-k corrections.

    from khspell.topk import Speller
    sp = Speller.load()                                   # models from outputs/
    text, errors = sp.check("ខ្ញុំជាសស្សដែលមានសក្សានៅកម្ពុច", k=3)
    sp.suggest("សំណើរ", k=3)

How one text is checked:
  1. character tagger  -> cuts the text into words, marks the likely misspelled ones
  2. candidates        -> ByT5 beams (context) + dictionary words spelled or sounding alike
  3. filter            -> real words only (lexicon), plus "keep the word as typed"
  4. rank              -> ByT5 log P(candidate | context) + prior_weight * dictionary prior
  A word is reported only if its best candidate is a different word.

The ByT5 model input marks the word to check:   <left context>[[word]]<right context>
and it outputs the correct spelling of that word.
"""
import json
import math
import os

import torch

from . import config
from .names import name_positions
from .noise import corrupt, is_khmer_token
from .normalize import normalize

L_MARK, R_MARK = "[[", "]]"
CONTEXT = 40          # characters of context on each side of the word


def make_input(left, word, right):
    return f"{left[-CONTEXT:]}{L_MARK}{word}{R_MARK}{right[:CONTEXT]}"


def make_example(tokens, rng, p_error=0.8, p_context_error=0.1):
    """One training pair from a clean, word-tokenised sentence: (model input, correct word).

    80% of the time the chosen word is misspelled; 20% it stays correct, so the model
    also learns to leave correct words alone. Other words get a few errors too (10%)."""
    idx = [i for i, t in enumerate(tokens) if is_khmer_token(t)]
    if not idx:
        return None
    i = rng.choice(idx)
    toks = list(tokens)
    for j in idx:
        if j != i and rng.random() < p_context_error:
            toks[j] = corrupt(toks[j], rng, tries=3) or toks[j]
    gold = word = tokens[i]
    if rng.random() < p_error:
        word = corrupt(gold, rng, tries=5) or gold
    return make_input("".join(toks[:i]), word, "".join(toks[i + 1:])), gold


class ByT5Speller:
    """The trained ByT5 model: proposes corrections (beam search) and scores any candidate."""

    def __init__(self, model_dir, device=None):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device).eval()

    def _encode(self, inputs):
        return self.tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=384).to(self.device)

    @torch.no_grad()
    def beams(self, inputs, k=10, batch_size=32):
        """For each input: the k best outputs, [(word, relative prob)], best first."""
        out = []
        for b in range(0, len(inputs), batch_size):
            gen = self.model.generate(**self._encode(inputs[b:b + batch_size]), num_beams=k,
                                      num_return_sequences=k, max_new_tokens=96, max_length=None,
                                      output_scores=True, return_dict_in_generate=True)
            words = self.tok.batch_decode(gen.sequences, skip_special_tokens=True)
            probs = torch.softmax(gen.sequences_scores.view(-1, k).float(), -1).tolist()
            for n, p in enumerate(probs):
                merged = {}
                for w, pw in zip(words[n * k:(n + 1) * k], p):
                    w = normalize(w)
                    if w:
                        merged[w] = merged.get(w, 0.0) + pw
                out.append(sorted(merged.items(), key=lambda x: -x[1]))
        return out

    @torch.no_grad()
    def score(self, inputs, targets, batch_size=64):
        """log P(target | input) for each pair, so dictionary candidates get a fair score too."""
        out = []
        for b in range(0, len(inputs), batch_size):
            labels = self.tok(text_target=targets[b:b + batch_size], return_tensors="pt", padding=True,
                              truncation=True, max_length=96).input_ids.to(self.device)
            labels[labels == self.tok.pad_token_id] = -100
            logits = self.model(**self._encode(inputs[b:b + batch_size]), labels=labels).logits.float()
            lp = torch.log_softmax(logits, -1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            out.extend((lp * (labels != -100)).sum(1).tolist())
        return out


class RankWeights:
    """score = log P_ByT5(candidate) + prior_weight * prior(candidate) (+ keep_bias for the typed word).
    keep_bias > 0 means fewer false alarms. Both are tuned by scripts/eval_topk.py --tune."""

    def __init__(self, prior_weight=0.5, keep_bias=0.0):
        self.prior_weight, self.keep_bias = prior_weight, keep_bias

    @classmethod
    def load(cls, path=None):
        try:
            with open(path or config.RANK_WEIGHTS, encoding="utf-8") as f:
                return cls(**json.load(f))
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path=None):
        with open(path or config.RANK_WEIGHTS, "w", encoding="utf-8") as f:
            json.dump({"prior_weight": self.prior_weight, "keep_bias": self.keep_bias}, f, indent=2)


class Speller:
    def __init__(self, byt5, tagger=None, candidates=None, weights=None, n_beams=10, n_dict=20):
        self.byt5, self.tagger, self.gen = byt5, tagger, candidates
        self.w = weights or RankWeights()
        self.n_beams, self.n_dict = n_beams, n_dict

    @classmethod
    def load(cls, topk_model=None, tagger_path=None):
        """Load everything from outputs/ (paths can be overridden, e.g. for the --quick models)."""
        topk_model = topk_model or os.environ.get("KHSPELL_TOPK_MODEL", str(config.TOPK_DIR))
        tagger_path = tagger_path or os.environ.get("KHSPELL_TAGGER", str(config.TAGGER_CKPT))
        tagger = gen = None
        if os.path.exists(tagger_path):
            from .tagger import Tagger
            tagger = Tagger.load(tagger_path)
        if config.LEXICON_TSV.exists():
            from .candidates import CandidateGenerator
            gen = CandidateGenerator.load()
        return cls(ByT5Speller(topk_model), tagger, gen, RankWeights.load())

    # ------------------------------------------------------------------ one word
    def features(self, word, left="", right=""):
        """Every candidate with its features: [(candidate, byt5_logp, prior)]."""
        inp = make_input(left, word, right)
        cands = {w for w, _ in self.byt5.beams([inp], k=self.n_beams)[0]}
        if self.gen is not None:
            cands |= set(self.gen.candidates(word, limit=self.n_dict))
            real = {c for c in cands if self.gen.is_known(c)}
            cands = real or cands                  # only real words, when there are any
        cands.add(word)                            # "keep as typed" competes too
        cands = sorted(cands)
        logp = self.byt5.score([inp] * len(cands), cands)
        prior = [self.gen.prior(word, c) if self.gen is not None else 0.0 for c in cands]
        return list(zip(cands, logp, prior))

    def rank_features(self, word, feats, w=None):
        w = w or self.w
        scored = [(c, lp + w.prior_weight * pr + (w.keep_bias if c == word else 0.0)) for c, lp, pr in feats]
        m = max(s for _, s in scored)
        z = sum(math.exp(s - m) for _, s in scored)
        return sorted(((c, math.exp(s - m) / z) for c, s in scored), key=lambda x: -x[1])

    def rank(self, word, left="", right=""):
        """[(candidate, probability)] best first, including the word as typed."""
        word = normalize(word)
        return self.rank_features(word, self.features(word, left, right))

    def suggest(self, word, left="", right="", k=3):
        word = normalize(word)
        return [(c, round(p, 4)) for c, p in self.rank(word, left, right) if c != word][:k]

    # ------------------------------------------------------------------ text
    def words(self, text):
        """[(start, end, word, error_prob or None)] - tagger if trained, else khmer-nltk."""
        if self.tagger is not None:
            return self.tagger.tag([text])[0]
        from khmernltk import word_tokenize
        out, pos = [], 0
        for t in word_tokenize(text):
            out.append((pos, pos + len(t), t, None))
            pos += len(t)
        return out

    def check(self, text, k=3, tag_threshold=0.5):
        """(normalised text, [{start, end, word, error_prob, suggestions: [(word, prob)]}])"""
        text = normalize(text)
        errors = []
        words = self.words(text)
        names = name_positions([w for _, _, w, _ in words], self.gen.is_known) if self.gen is not None else set()
        for i, (s, e, w, p_err) in enumerate(words):
            if not is_khmer_token(w):
                continue                           # English, numbers, punctuation: not checked
            if i in names:
                continue                           # unknown word after លោក / ខេត្ត / ក្រុមហ៊ុន ...: a name
            if p_err is not None and p_err < tag_threshold:
                continue                           # the tagger says the word is fine
            ranked = self.rank(w, text[:s], text[e:])
            if ranked[0][0] == w:
                continue                           # best candidate is the word itself
            errors.append({"start": s, "end": e, "word": w,
                           "error_prob": None if p_err is None else round(p_err, 3),
                           "suggestions": [(c, round(p, 4)) for c, p in ranked if c != w][:k]})
        return text, errors

"""Character tagger: cuts Khmer text into words AND marks misspelled words, in one pass.

For every character it predicts
    boundary: B (a word starts here) / I (inside a word)
    error   : ok / err (the character belongs to a misspelled word)

    text    : ខ  ្ញុំ ជ  ា  ស  ស  ្ស  ដ  ែ  ល
    boundary: B  I   B  I  B  I  I   B  I  I
    error   : ok ok  ok ok err err err ok ok ok

It is trained on clean segmented sentences with errors injected by khspell.noise,
so it keeps a misspelled word together (សស្ស stays one word) - unlike segmenters
trained on clean text only. Architecture: bidirectional Transformer encoder over
characters (as in Buoy et al. 2025), with two small classification heads.
"""
import math

import torch
import torch.nn as nn

from .noise import noisy_tokens

PAD_ID, UNK_ID = 0, 1
IGNORE = -100
MAX_LEN = 256
# characters that can never start a word: dependent vowels, signs, coeng
_NO_START = set(chr(c) for c in range(0x17B6, 0x17D4)) | {"៝"}


class CharTagger(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, layers=6, ff=1024, dropout=0.1, max_len=MAX_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.boundary = nn.Linear(d_model, 2)
        self.error = nn.Linear(d_model, 2)
        self.d_model = d_model

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.emb(x) * math.sqrt(self.d_model) + self.pos(pos)[None]
        h = self.norm(self.encoder(h, src_key_padding_mask=x == PAD_ID))
        return self.boundary(h), self.error(h)


class CharVocab:
    def __init__(self, chars):
        base = {chr(c) for c in range(0x1780, 0x1800)} | {" "}   # whole Khmer block is always known
        self.itos = ["<pad>", "<unk>"] + sorted(base | set(chars))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, text):
        return [self.stoi.get(c, UNK_ID) for c in text[:MAX_LEN]]


def make_example(tokens, rng):
    """(noisy text, boundary labels, error labels) from a clean tokenised sentence."""
    toks, flags = noisy_tokens(tokens, rng)
    b, e = [], []
    for t, bad in zip(toks, flags):
        b += [1] + [0] * (len(t) - 1)
        e += [int(bad)] * len(t)
    return "".join(toks), b, e


def gold_example(tokens, flags):
    """Labels for already-noisy tokens (used for fixed dev/test sets)."""
    b, e = [], []
    for t, bad in zip(tokens, flags):
        b += [1] + [0] * (len(t) - 1)
        e += [int(bad)] * len(t)
    return "".join(tokens), b, e


def pad(seqs, value, device):
    n = max(len(s) for s in seqs)
    t = torch.full((len(seqs), n), value, dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, :len(s)] = torch.tensor(s[:n], dtype=torch.long)
    return t.to(device)


def decode(text, p_start, p_err):
    """Character probabilities -> [(start, end, word, error_prob)]."""
    starts = [0]
    for i in range(1, len(text)):
        ch, prev = text[i], text[i - 1]
        if ch in _NO_START or prev == "្":
            continue                      # never cut inside a character cluster
        script_change = ("ក" <= ch <= "៝") != ("ក" <= prev <= "៝")
        if ch == " " or prev == " " or script_change or p_start[i] >= 0.5:
            starts.append(i)              # English words / digits are always separate (and not checked)
    starts.append(len(text))
    words = []
    for s, e in zip(starts, starts[1:]):
        words.append((s, e, text[s:e], float(sum(p_err[s:e]) / (e - s))))
    return words


class Tagger:
    """Inference: Tagger.load(path).tag(["ខ្ញុំជាសស្ស..."]) -> [[(start, end, word, err_prob), ...]]"""

    def __init__(self, model, vocab, device="cpu"):
        self.model, self.vocab, self.device = model.eval(), vocab, device

    @classmethod
    def load(cls, path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location=device, weights_only=False)
        vocab = CharVocab([])
        vocab.itos = ck["itos"]
        vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
        model = CharTagger(len(vocab.itos), **ck["config"]).to(device)
        model.load_state_dict(ck["model"])
        return cls(model, vocab, device)

    @torch.no_grad()
    def probs(self, texts, batch_size=128):
        out = []
        for b in range(0, len(texts), batch_size):
            chunk = texts[b:b + batch_size]
            x = pad([self.vocab.encode(t) for t in chunk], PAD_ID, self.device)
            lb, le = self.model(x)
            pb = torch.softmax(lb.float(), -1)[..., 1].tolist()
            pe = torch.softmax(le.float(), -1)[..., 1].tolist()
            out.extend((pb[i][:len(t)], pe[i][:len(t)]) for i, t in enumerate(chunk))
        return out

    def tag(self, texts):
        res = []
        for text in texts:
            # the model sees at most MAX_LEN characters: tag long text in windows
            words, off = [], 0
            while off < len(text):
                piece = text[off:off + MAX_LEN]
                if len(piece) == MAX_LEN and " " in piece[MAX_LEN // 2:]:
                    piece = piece[:piece.rindex(" ") + 1]     # cut long text at a space
                (pb, pe), = self.probs([piece])
                words += [(s + off, e + off, w, p) for s, e, w, p in decode(piece, pb, pe)]
                off += len(piece)
            res.append(words)
        return res

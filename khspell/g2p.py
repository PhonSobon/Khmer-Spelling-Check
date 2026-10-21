"""Character-level Transformer encoder-decoder G2P: word -> pronunciation respelling.

Architecture follows Buoy et al. 2025 (arXiv 2511.09812): 3 encoder + 3 decoder
layers, d_model=192, 8 heads. Target is the dictionary's `pro` respelling
(e.g. បម្លែង -> បំ-ឡែង), which `phonetics.phones` turns into phone tokens.
"""
import math

import torch
import torch.nn as nn

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]


class CharVocab:
    def __init__(self, chars):
        self.itos = SPECIALS + sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, s, bos=False, eos=False):
        ids = [self.stoi.get(c, UNK) for c in s]
        return ([BOS] if bos else []) + ids + ([EOS] if eos else [])

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS:
                break
            if i > UNK:
                out.append(self.itos[i])
        return "".join(out)

    def __len__(self):
        return len(self.itos)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=192, nhead=8, enc_layers=3, dec_layers=3,
                 ff=768, dropout=0.1, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        self.transformer = nn.Transformer(d_model, nhead, enc_layers, dec_layers, ff, dropout,
                                          batch_first=True, norm_first=True)
        self.out = nn.Linear(d_model, tgt_vocab)

    def _embed(self, emb, x):
        pos = torch.arange(x.size(1), device=x.device)
        return emb(x) * math.sqrt(self.d_model) + self.pos(pos)[None]

    def encode(self, src):
        return self.transformer.encoder(self._embed(self.src_emb, src), src_key_padding_mask=src == PAD)

    def decode(self, tgt, memory, src_pad):
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        h = self.transformer.decoder(self._embed(self.tgt_emb, tgt), memory, tgt_mask=mask, tgt_is_causal=True,
                                     tgt_key_padding_mask=tgt == PAD, memory_key_padding_mask=src_pad)
        return self.out(h)

    def forward(self, src, tgt_in):
        return self.decode(tgt_in, self.encode(src), src == PAD)


def pad_batch(seqs, device):
    n = max(len(s) for s in seqs)
    t = torch.full((len(seqs), n), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, :len(s)] = torch.tensor(s)
    return t.to(device)


class G2P:
    """Inference wrapper. `G2P.load(path)(["បំលែង"]) -> ["បំ-លែង"]`"""

    def __init__(self, model, src_vocab, tgt_vocab, device="cpu"):
        self.model, self.src, self.tgt, self.device = model.eval(), src_vocab, tgt_vocab, device
        self.cache = {}

    @classmethod
    def load(cls, path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location=device, weights_only=False)
        src, tgt = CharVocab([]), CharVocab([])
        src.itos, tgt.itos = ck["src_itos"], ck["tgt_itos"]
        src.stoi = {c: i for i, c in enumerate(src.itos)}
        tgt.stoi = {c: i for i, c in enumerate(tgt.itos)}
        model = Seq2SeqTransformer(len(src), len(tgt), **ck["config"]).to(device)
        model.load_state_dict(ck["model"])
        return cls(model, src, tgt, device)

    @torch.no_grad()
    def __call__(self, words, max_len=60, batch_size=512):
        todo = [w for w in dict.fromkeys(words) if w not in self.cache]
        for b in range(0, len(todo), batch_size):
            chunk = todo[b:b + batch_size]
            src = pad_batch([self.src.encode(w[:max_len - 2]) for w in chunk], self.device)
            memory, src_pad = self.model.encode(src), src == PAD
            ys = torch.full((len(chunk), 1), BOS, dtype=torch.long, device=self.device)
            done = torch.zeros(len(chunk), dtype=torch.bool, device=self.device)
            for _ in range(max_len):
                nxt = self.model.decode(ys, memory, src_pad)[:, -1].argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
                ys = torch.cat([ys, nxt[:, None]], 1)
                done |= nxt == EOS
                if done.all():
                    break
            for w, row in zip(chunk, ys[:, 1:].tolist()):
                self.cache[w] = self.tgt.decode(row)
        return [self.cache[w] for w in words]


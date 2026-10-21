"""Step 4: train the G2P Transformer on the dictionary's word -> pronunciation pairs.

    python scripts/train_g2p.py --epochs 40
Reports character error rate (CER), exact-match accuracy and phone-key accuracy
on a held-out 5% test split, like the paper (their CER: 6.6%).
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from khspell import config, resume
from khspell.distance import levenshtein
from khspell.g2p import G2P, PAD, CharVocab, Seq2SeqTransformer, pad_batch
from khspell.normalize import is_khmer_word
from khspell.phonetics import dictionary_pronunciations, phones


def load_pairs():
    pairs = {}
    for w, alts in dictionary_pronunciations().items():
        if is_khmer_word(w) and all(is_khmer_word(a.replace("-", "")) for a in alts):
            pairs[w] = alts[0]             # first listed pronunciation is the main one
    return sorted(pairs.items())


def evaluate(g2p, pairs):
    hyps = g2p([w for w, _ in pairs])
    edits = chars = exact = key_ok = 0
    for (w, ref), hyp in zip(pairs, hyps):
        edits += levenshtein(hyp, ref)
        chars += len(ref)
        exact += hyp == ref
        key_ok += phones(hyp) == phones(ref)
    n = len(pairs)
    return {"CER": edits / chars, "exact": exact / n, "phone_key_acc": key_ok / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0, help="use only N word pairs (quick test)")
    ap.add_argument("--fresh", action="store_true", help="ignore a saved resume state and start over")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = load_pairs()
    random.shuffle(pairs)
    if args.limit:
        pairs = pairs[:args.limit]
    n = len(pairs)
    test, dev, train = pairs[: n // 20], pairs[n // 20: n // 10], pairs[n // 10:]
    print(f"pairs: train {len(train):,}  dev {len(dev):,}  test {len(test):,}  device={device}")

    src_v = CharVocab("".join(w for w, _ in pairs))
    tgt_v = CharVocab("".join(p for _, p in pairs))
    cfg = dict(d_model=192, nhead=8, enc_layers=3, dec_layers=3, ff=768, dropout=0.1, max_len=100)
    model = Seq2SeqTransformer(len(src_v), len(tgt_v), **cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(train) // args.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    def save(path):
        torch.save({"model": model.state_dict(), "config": cfg,
                    "src_itos": src_v.itos, "tgt_itos": tgt_v.itos}, path)

    best, start_ep = float("inf"), 1
    st = resume.load(config.G2P_CKPT, args.fresh)      # after the data split: the split never changes
    if st:
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        best, start_ep = st["best"], st["epoch"] + 1
        print(f"resuming after epoch {st['epoch']} (best dev CER so far {best:.3%})", flush=True)
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        random.shuffle(train)
        t0, total = time.time(), 0.0
        for b in range(0, len(train), args.batch):
            chunk = sorted(train[b:b + args.batch], key=lambda x: len(x[0]))
            src = pad_batch([src_v.encode(w) for w, _ in chunk], device)
            tgt = pad_batch([tgt_v.encode(p, bos=True, eos=True) for _, p in chunk], device)
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(chunk)
        if ep % 5 == 0 or ep == args.epochs:
            model.eval()
            m = evaluate(G2P(model, src_v, tgt_v, device), dev)
            print(f"epoch {ep:3d}  loss {total / len(train):.4f}  dev CER {m['CER']:.3%}  "
                  f"exact {m['exact']:.1%}  phone-key {m['phone_key_acc']:.1%}  ({time.time() - t0:.0f}s/epoch)",
                  flush=True)
            if m["CER"] < best:
                best = m["CER"]
                save(config.G2P_CKPT)
        else:
            print(f"epoch {ep:3d}  loss {total / len(train):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        resume.save(config.G2P_CKPT, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                    best=best, epoch=ep)
    resume.finish(config.G2P_CKPT)

    g2p = G2P.load(config.G2P_CKPT, device)
    m = evaluate(g2p, test)
    print(f"TEST  CER {m['CER']:.3%}  exact {m['exact']:.1%}  phone-key {m['phone_key_acc']:.1%}")
    for w in ["បម្លែង", "បំលែង", "ចម្រៀង", "ចំរៀង", "កម្ទេច", "កំទេច", "ស្រឡាញ់", "ស្រលាញ់"]:
        p = g2p([w])[0]
        print(f"  {w} -> {p}  {' '.join(phones(p))}")


if __name__ == "__main__":
    main()

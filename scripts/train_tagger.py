"""Train the character tagger (word cutting + error marking).

Needs outputs/data/train.parquet from scripts/prepare_data.py.

    python scripts/train_tagger.py                 # RTX 3090: ~20-40 min
    python scripts/train_tagger.py --epochs 1 --limit 20000   # quick test

Reports on a fixed dev set:
  seg F1  : word-boundary F1 (are words cut correctly, even misspelled ones?)
  err P/R/F1 : word-level error detection
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn.functional as F

from khspell import config, resume
from khspell.noise import noisy_tokens
from khspell.tagger import IGNORE, PAD_ID, CharTagger, CharVocab, Tagger, make_example, pad


def encode_batch(examples, vocab, device):
    x = pad([vocab.encode(t) for t, _, _ in examples], PAD_ID, device)
    yb = pad([b[:x.size(1)] for _, b, _ in examples], IGNORE, device)
    ye = pad([e[:x.size(1)] for _, _, e in examples], IGNORE, device)
    return x, yb, ye


def evaluate(tagger, dev):
    """dev: list of (noisy_tokens, flags)."""
    tp_seg = n_pred = n_gold = 0
    tp_err = fp_err = fn_err = 0
    texts = ["".join(t) for t, _ in dev]
    tagged = tagger.tag(texts)
    for (toks, flags), words in zip(dev, tagged):
        gold, pos = {}, 0
        for t, bad in zip(toks, flags):
            gold[(pos, pos + len(t))] = bad
            pos += len(t)
        pred = {(s, e): p >= 0.5 for s, e, _, p in words}
        n_pred += len(pred)
        n_gold += len(gold)
        tp_seg += len(pred.keys() & gold.keys())
        # error detection, word level: a gold error is found if a flagged predicted word overlaps it;
        # a false alarm is a flagged predicted word that overlaps no gold error
        gold_err = [(s, e) for (s, e), bad in gold.items() if bad]
        flagged = [(s, e) for (s, e), flag in pred.items() if flag]
        for s, e in gold_err:
            found = any(ps < e and pe > s for ps, pe in flagged)
            tp_err += found
            fn_err += not found
        fp_err += sum(not any(ps < e and pe > s for s, e in gold_err) for ps, pe in flagged)
    p = tp_seg / max(1, n_pred)
    r = tp_seg / max(1, n_gold)
    ep = tp_err / max(1, tp_err + fp_err)
    er = tp_err / max(1, tp_err + fn_err)
    return {"seg_f1": 2 * p * r / max(1e-9, p + r), "err_p": ep, "err_r": er,
            "err_f1": 2 * ep * er / max(1e-9, ep + er)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.TAGGER_CKPT))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="use only N sentences (quick test)")
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--state-every", type=int, default=2000, help="save a resume state every N steps")
    ap.add_argument("--fresh", action="store_true", help="ignore a saved resume state and start over")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = config.bf16_ok()

    df = pd.read_parquet(config.TRAIN_DATA, columns=["tokens"])
    if args.limit:
        df = df.head(args.limit)
    sents = [list(t) for t in df["tokens"]]
    dev_rng = random.Random(0)
    dev = [noisy_tokens(t, dev_rng) for t in sents[:args.n_dev]]
    train = sents[args.n_dev:]
    vocab = CharVocab("".join("".join(t) for t in train[:200_000]))
    print(f"train {len(train):,} sentences | dev {len(dev):,} | vocab {len(vocab.itos)} | device {device}")

    cfg = dict(d_model=args.d_model, nhead=8, layers=args.layers, ff=4 * args.d_model)
    model = CharTagger(len(vocab.itos), **cfg).to(device)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    steps_per_epoch = (len(train) + args.batch - 1) // args.batch
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.05)
    err_weight = torch.tensor([1.0, 3.0], device=device)      # errors are rare: weight them up

    def save():
        torch.save({"model": model.state_dict(), "config": cfg, "itos": vocab.itos}, args.out)

    best, step, start_ep, start_b, order = -1.0, 0, 1, 0, None
    st = resume.load(args.out, args.fresh)
    if st:
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rng.setstate(st["rng"])
        best, step, start_ep, start_b, order = st["best"], st["step"], st["epoch"], st["next_b"], st["order"]
        print(f"resuming from step {step:,} (epoch {start_ep}, sentence {start_b:,})", flush=True)

    def save_state(ep, next_b, order):
        resume.save(args.out, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                    rng=rng.getstate(), best=best, step=step, epoch=ep, next_b=next_b, order=order)

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        if order is None:
            order = list(range(len(train)))
            rng.shuffle(order)
        t0 = time.time()
        for b in range(start_b, len(order), args.batch):
            batch = [make_example(train[i], rng) for i in order[b:b + args.batch]]
            x, yb, ye = encode_batch(batch, vocab, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                lb, le = model(x)
            loss = (F.cross_entropy(lb.float().reshape(-1, 2), yb.reshape(-1), ignore_index=IGNORE)
                    + F.cross_entropy(le.float().reshape(-1, 2), ye.reshape(-1), weight=err_weight,
                                      ignore_index=IGNORE))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                done = (b + args.batch) / len(order)
                eta = (time.time() - t0) / done * (1 - done) / 60
                print(f"epoch {ep} step {step:6d}/{args.epochs * steps_per_epoch}  loss {loss.item():.4f}  "
                      f"(epoch eta {eta:.1f} min)", flush=True)
            if step % args.state_every == 0:
                save_state(ep, b + args.batch, order)
        start_b = 0
        model.eval()
        m = evaluate(Tagger(model, vocab, device), dev)
        print(f"== epoch {ep}: seg F1 {m['seg_f1']:.3f} | error P {m['err_p']:.3f} R {m['err_r']:.3f} "
              f"F1 {m['err_f1']:.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if m["seg_f1"] + m["err_f1"] > best:
            best = m["seg_f1"] + m["err_f1"]
            save()
            print(f"   saved best -> {args.out}")
        order = None
        save_state(ep + 1, 0, None)
    resume.finish(args.out)


if __name__ == "__main__":
    main()

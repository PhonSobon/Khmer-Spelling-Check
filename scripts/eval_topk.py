"""Evaluate (and tune) the complete spell checker.

    python scripts/eval_topk.py --tune          # models from outputs/

1. --tune: grid-search the ranking weights on dev words in context -> outputs/rank_weights.json
2. Word level, real misspellings (917 pairs from seanghay/khmer-spelling-corrections, no context)
   and held-out test words in context: detection, acc@1/@3, MRR, false alarms on correct words.
   Compared: "ByT5 beams only" vs "full" (ByT5 + dictionary + phoneme candidates, real words only).
3. Sentence level, end to end with the tagger: detection precision/recall, top-3 accuracy,
   false alarms on clean sentences.
"""
import argparse
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from khspell import config, hf
from khspell.noise import noisy_tokens
from khspell.normalize import normalize
from khspell.topk import RankWeights, Speller, make_example

KS = (1, 3)
EXAMPLES = ["ខ្ញុំជាសស្សដែលមានសក្សានៅកម្ពុច", "គាត់បានបំលែងឯកសារនេះ", "សូមផ្ញើសំណើរមកក្រសួង"]


def word_items_real(limit=0):
    ds = hf.load(config.REAL_TYPOS_ID)
    miss = [normalize(m) for m in ds["misspelling"]]
    corr = [normalize(c) for c in ds["correction"]]
    if limit:
        miss, corr = miss[:limit], corr[:limit]
    # (typed word, left, right, gold): misspellings and, for false alarms, the correct words themselves
    return [(m, "", "", c) for m, c in zip(miss, corr)] + [(c, "", "", c) for c in corr]


def word_items_context(texts, seed, n):
    from khmernltk import word_tokenize
    rng = random.Random(seed)
    items = []
    for t in texts:
        ex = make_example(word_tokenize(t), rng)
        if not ex:
            continue
        inp, gold = ex
        left, rest = inp.split("[[", 1)
        word, right = rest.split("]]", 1)
        items.append((word, left, right, gold))
        if len(items) >= n:
            break
    return items


def word_metrics(ranked_lists, items):
    ranked_lists = [r or [(it[0], 1.0)] for r, it in zip(ranked_lists, items)]   # empty output = keep word
    err = [i for i, (w, _, _, g) in enumerate(items) if w != g]
    ok = [i for i, (w, _, _, g) in enumerate(items) if w == g]
    res = {"errors": len(err), "correct words": len(ok)}
    if err:
        res["detected"] = sum(ranked_lists[i][0][0] != items[i][0] for i in err) / len(err)
        for k in KS:
            res[f"acc@{k}"] = sum(items[i][3] in [c for c, _ in ranked_lists[i] if c != items[i][0]][:k]
                                  for i in err) / len(err)
        rr = []
        for i in err:
            sugg = [c for c, _ in ranked_lists[i] if c != items[i][0]]
            rr.append(1 / (sugg.index(items[i][3]) + 1) if items[i][3] in sugg else 0.0)
        res["MRR"] = sum(rr) / len(err)
    if ok:
        res["false alarm"] = sum(ranked_lists[i][0][0] != items[i][0] for i in ok) / len(ok)
    return res


def beams_only(sp, items):
    """Baseline: the ByT5 beams alone, without dictionary candidates or ranking."""
    from khspell.topk import make_input
    return sp.byt5.beams([make_input(l, w, r) for w, l, r, _ in items], k=5)


def tune(sp, items):
    feats = [sp.features(w, l, r) for w, l, r, _ in tqdm(items, desc="tune: candidate features", mininterval=10)]
    best, best_score = RankWeights(), float("-inf")
    for pw, kb in itertools.product([0.0, 0.25, 0.5, 1.0, 2.0], [0.0, 0.5, 1.0, 2.0]):
        w = RankWeights(pw, kb)
        m = word_metrics([sp.rank_features(it[0], f, w) for it, f in zip(items, feats)], items)
        score = m.get("acc@1", 0) - m.get("false alarm", 0)       # correct fixes minus false alarms
        print(f"  prior_weight={pw:<4} keep_bias={kb:<4} acc@1={m.get('acc@1', 0):.3f} "
              f"false_alarm={m.get('false alarm', 0):.3f}")
        if score > best_score:
            best, best_score = w, score
    print(f"best: prior_weight={best.prior_weight} keep_bias={best.keep_bias}")
    return best


def sentence_eval(sp, texts, seed, tag_threshold):
    from khmernltk import word_tokenize
    rng = random.Random(seed)
    tp = fp = fn = top3 = clean_alarm = n_clean = 0
    for t in tqdm(texts, desc="sentences", mininterval=10):
        clean_toks = word_tokenize(t)
        toks, flags = noisy_tokens(clean_toks, rng)
        noisy = "".join(toks)
        gold, pos = [], 0
        for tok, bad, clean_tok in zip(toks, flags, clean_toks):
            if bad:
                gold.append((pos, pos + len(tok), clean_tok))
            pos += len(tok)
        if not gold:
            n_clean += 1
            clean_alarm += bool(sp.check(noisy, tag_threshold=tag_threshold)[1])
            continue
        _, errors = sp.check(noisy, k=3, tag_threshold=tag_threshold)
        for s, e, clean_word in gold:
            hit = [er for er in errors if er["start"] < e and er["end"] > s]
            if hit:
                tp += 1
                top3 += any(clean_word == c for c, _ in hit[0]["suggestions"])
            else:
                fn += 1
        fp += sum(not any(er["start"] < e and er["end"] > s for s, e, _ in gold) for er in errors)
    return {"error words": tp + fn, "detect precision": tp / max(1, tp + fp), "detect recall": tp / max(1, tp + fn),
            "top-3 fixed (of all errors)": top3 / max(1, tp + fn),
            "clean sentences flagged": clean_alarm / max(1, n_clean)}


def show(title, res):
    print(f"\n[{title}]")
    for k, v in res.items():
        print(f"  {k:<28} {v:.1%}" if isinstance(v, float) else f"  {k:<28} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(config.TOPK_DIR))
    ap.add_argument("--tagger", default=str(config.TAGGER_CKPT))
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--n-tune", type=int, default=600)
    ap.add_argument("--n-words", type=int, default=1500)
    ap.add_argument("--n-sent", type=int, default=500)
    ap.add_argument("--n-real", type=int, default=0, help="limit the real misspellings (0 = all 917)")
    ap.add_argument("--tag-threshold", type=float, default=0.5)
    args = ap.parse_args()

    sp = Speller.load(args.model, args.tagger)
    print(f"tagger: {'yes' if sp.tagger else 'no (khmer-nltk)'} | lexicon: {'yes' if sp.gen else 'no'} | "
          f"G2P: {'yes' if sp.gen is not None and sp.gen.g2p else 'no'}")

    if args.tune:
        dev_texts = list(pd.read_parquet(config.DEV_DATA)["text"])
        sp.w = tune(sp, word_items_context(dev_texts, seed=11, n=args.n_tune))
        sp.w.save()
        print("saved", config.RANK_WEIGHTS)

    for text in EXAMPLES:
        fixed, errors = sp.check(text, k=3, tag_threshold=args.tag_threshold)
        print(f"\n{fixed}")
        for e in errors:
            print(f"  {e['word']} -> {e['suggestions']}")

    test_texts = list(pd.read_parquet(config.TEST_DATA)["text"])
    for name, items in [("real misspellings, no context", word_items_real(args.n_real)),
                        ("test words in context", word_items_context(test_texts, seed=12, n=args.n_words))]:
        show(f"{name} | ByT5 beams only", word_metrics(beams_only(sp, items), items))
        full = [sp.rank(w, l, r) for w, l, r, _ in tqdm(items, desc="full system", mininterval=10)]
        show(f"{name} | FULL (ByT5 + dictionary + phoneme)", word_metrics(full, items))

    show("sentences end to end", sentence_eval(sp, test_texts[:args.n_sent], seed=13,
                                               tag_threshold=args.tag_threshold))


if __name__ == "__main__":
    main()

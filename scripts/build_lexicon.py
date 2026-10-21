"""Step 3: build the lexicon (known words + frequencies) -> outputs/lexicon.tsv

    python scripts/build_lexicon.py

Words = the official dictionary (seanghay/khmer-dictionary-44k)
      + frequent words of the cleaned training sentences (outputs/data/train.parquet),
        e.g. names, places and new words that are not in the dictionary.
A frequent corpus word that is just a rare variant of a much more frequent dictionary
word (e.g. សំណើរ next to សំណើ) is treated as a common misspelling and left out; those
go to outputs/rejected_corpus_words.tsv so you can check them.
"""
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from khspell import config, hf
from khspell.distance import weighted_distance
from khspell.lexicon import Lexicon
from khspell.normalize import clusters, is_khmer_word, match_key, normalize
from khspell.symdel import SymDeleteIndex


def dictionary_words():
    words = set()
    for w in hf.load(config.DICT_ID)["word"]:
        w = normalize(w)
        if is_khmer_word(w):
            words.add(w)
    return words


def count_training_words():
    if not config.TRAIN_DATA.exists():
        sys.exit(f"missing {config.TRAIN_DATA}: run scripts/prepare_data.py first")
    counts = collections.Counter()
    for toks in pd.read_parquet(config.TRAIN_DATA, columns=["tokens"])["tokens"]:
        counts.update(t for t in toks if is_khmer_word(t))
    return counts


def main():
    dict_words = dictionary_words()
    print(f"dictionary: {len(dict_words):,} words")
    counts = count_training_words()
    print(f"training data: {len(counts):,} different Khmer words, {sum(counts.values()):,} in total")

    key_counts = collections.Counter()
    for w, n in counts.items():
        key_counts[match_key(w)] += n
    dict_keys = {match_key(w) for w in dict_words}
    entries = [(w, key_counts[match_key(w)], True) for w in dict_words]

    near_index = SymDeleteIndex(max_dist=1)          # to spot "rare variant of a frequent word"
    for w in dict_words:
        near_index.add(clusters(w), w)
    accepted, rejected = [], []
    for w, n in counts.most_common():
        if n < config.CORPUS_MIN_FREQ:
            break
        if match_key(w) in dict_keys:
            continue
        near = next((c for c, _ in near_index.lookup(clusters(w), max_dist=1)
                     if weighted_distance(w, c) <= 1.0 and key_counts[match_key(c)] >= config.CORPUS_NEIGHBOR_RATIO * n),
                    None)
        (rejected if near else accepted).append((w, n, near))

    lex = Lexicon(entries + [(w, n, False) for w, n, _ in accepted])
    lex.save()
    with open(config.OUT / "rejected_corpus_words.tsv", "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t", lineterminator="\n")
        wr.writerow(["word", "freq", "likely_misspelling_of"])
        wr.writerows(rejected)
    print(f"lexicon: {len(lex):,} words ({len(dict_words):,} dictionary + {len(accepted):,} from the training data)"
          f" -> {config.LEXICON_TSV}")
    print(f"left out {len(rejected):,} frequent words that look like misspellings -> rejected_corpus_words.tsv")


if __name__ == "__main__":
    main()

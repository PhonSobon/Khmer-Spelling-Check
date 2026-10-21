"""Step 4b: pronunciations for every lexicon word -> outputs/pron.tsv.

Dictionary words use the combined pronunciation dictionary (config.PRON_DICT_ID);
other words are pronounced by the G2P model.
    python scripts/build_pron.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from khspell import config
from khspell.g2p import G2P
from khspell.lexicon import Lexicon
from khspell.phonetics import dictionary_pronunciations


def main():
    lex = Lexicon.load()
    gold = {w: alts for w, alts in dictionary_pronunciations().items() if w in lex}
    todo = [w for w in lex.words() if w not in gold]
    print(f"{len(gold):,} words with dictionary pronunciation, {len(todo):,} need G2P")
    g2p = G2P.load(config.G2P_CKPT)
    pred = {}
    for b in tqdm(range(0, len(todo), 1024), desc="G2P"):
        chunk = todo[b:b + 1024]
        pred.update(zip(chunk, g2p(chunk)))
    with open(config.PRON_TSV, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t", lineterminator="\n")
        for w, alts in gold.items():
            for a in alts:
                wr.writerow([w, a])
        for w, p in pred.items():
            if p:
                wr.writerow([w, p])
    print("wrote", config.PRON_TSV)


if __name__ == "__main__":
    main()

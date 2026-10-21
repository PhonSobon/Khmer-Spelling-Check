"""Names (people, places, organizations): not misspellings.

A name is usually not in the lexicon, so a spell checker flags it and "corrects" it into
a real word (ដារ៉ា -> តារា). In Khmer, names follow a title or a head word:

    លោក / អ្នកស្រី / ឯកឧត្តម ...  + person        លោក ដារ៉ា
    ខេត្ត / ស្រុក / ប្រទេស ...     + place         ទីក្រុង តូក្យូ
    ក្រុមហ៊ុន / ធនាគារ / ក្រសួង ... + organization  ធនាគារ អេស៊ីលីដា

`name_positions` marks the UNKNOWN words right after such a word as names; known words
are checked as usual. Used when checking text, by the error generator (names are never
misspelled on purpose, so the tagger learns to leave them alone) and to mine names for
the lexicon from the corpus.
"""
from .normalize import normalize

PERSON_TITLES = """
លោក លោកស្រី អ្នកស្រី អ្នកនាង នាង កញ្ញា ឯកឧត្តម លោកជំទាវ សម្ដេច សម្តេច ឧកញ៉ា អ្នកឧកញ៉ា
បណ្ឌិត វេជ្ជបណ្ឌិត សាស្ត្រាចារ្យ សាស្ដ្រាចារ្យ ព្រះតេជគុណ ព្រះភិក្ខុ ព្រះបាទ ព្រះករុណា
ឧត្តមសេនីយ៍ វរសេនីយ៍ អនុសេនីយ៍ លោកគ្រូ អ្នកគ្រូ ព្រះនាង ស្ដេច ស្តេច ឈ្មោះ
""".split()

PLACE_HEADS = """
ខេត្ត ស្រុក ក្រុង ទីក្រុង រាជធានី ឃុំ សង្កាត់ ភូមិ ខណ្ឌ ប្រទេស ទ្វីប ផ្សារ ស្ទឹង ទន្លេ ភ្នំ
វត្ត ផ្លូវ មហាវិថី កោះ បឹង ព្រលានយន្តហោះ
""".split()

ORG_HEADS = """
ក្រសួង ក្រុមហ៊ុន ធនាគារ មន្ទីរពេទ្យ មន្ទីរ សាលារៀន សាលា សាកលវិទ្យាល័យ វិទ្យាល័យ វិទ្យាស្ថាន
អង្គការ សហគ្រាស គណបក្ស សមាគម ស្ថាប័ន នាយកដ្ឋាន អគ្គនាយកដ្ឋាន មជ្ឈមណ្ឌល ហាង សណ្ឋាគារ
ភោជនីយដ្ឋាន រោងចក្រ ក្លឹប ក្រុមការងារ
""".split()

NAME_HEADS = {normalize(w) for w in PERSON_TITLES + PLACE_HEADS + ORG_HEADS}
MAX_NAME_WORDS = 3      # a name has at most this many words after its head (ហេង សុវណ្ណ, អង្គរ អេឡិចត្រូនិក)


def _is_word(t):
    return bool(t) and all("ក" <= ch <= "៓" or ch in "ៜ៝" for ch in t)


def name_positions(words, is_known):
    """Indices of `words` that are probably (part of) a name: unknown words among the
    MAX_NAME_WORDS words after a title / head word. Spaces are skipped (លោក ហ៊ុន សែន);
    punctuation, digits or another title / head word end the name."""
    out = set()
    for i, w in enumerate(words):
        if w not in NAME_HEADS:
            continue
        seen, j = 0, i + 1
        while j < len(words) and seen < MAX_NAME_WORDS:
            t = words[j]
            if t.isspace():
                j += 1
                continue
            if not _is_word(t) or t in NAME_HEADS:
                break
            if not is_known(t):
                out.add(j)
            seen += 1
            j += 1
    return out

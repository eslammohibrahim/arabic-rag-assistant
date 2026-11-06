"""Arabic text normalization and tokenization for keyword search and answer scoring."""

import re

DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰ]")
TATWEEL = "ـ"
NOT_ALLOWED = re.compile(r"[^ء-غف-يa-z0-9\s]")
SPACES = re.compile(r"\s+")
ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
YA_TA = str.maketrans({"ى": "ي", "ة": "ه"})
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Very common words that carry no meaning for search ("from", "in", "what", "who", ...).
STOPWORDS = {
    "في", "من", "علي", "الي", "عن", "مع", "هو", "هي", "ما", "ماذا", "متي", "اين", "كيف", "كم",
    "هل", "التي", "الذي", "الذين", "ان", "او", "ثم", "قد", "كان", "كانت", "هذا", "هذه",
    "ذلك", "تلك", "بين", "عند", "لم", "لا", "و", "ب", "ل", "ف", "اي", "تم", "به", "بها", "له", "لها",
}
PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")


def normalize(text: str) -> str:
    text = str(text).lower()
    text = DIACRITICS.sub("", text).replace(TATWEEL, "")
    text = text.translate(ALEF).translate(YA_TA).translate(ARABIC_DIGITS)
    text = NOT_ALLOWED.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def light_stem(word: str) -> str:
    """Strip the definite article and attached prepositions: 'والمدينة' -> 'مدينه'."""
    for p in PREFIXES:
        if word.startswith(p) and len(word) - len(p) >= 3:
            return word[len(p):]
    return word


def tokenize(text: str, stem: bool = True) -> list[str]:
    words = [w for w in normalize(text).split() if w not in STOPWORDS]
    return [light_stem(w) for w in words] if stem else words

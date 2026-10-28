"""
Module 2: NLP text preprocessing with NLTK.

The same `preprocess_text` function is used during training and when the web
app classifies a new message: it is plugged into the TF-IDF vectorizer, which
is saved inside the trained pipeline.

Set up NLTK data once with:

    python -m src.preprocess --setup
"""
import argparse
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# NLTK data packages this module needs: (lookup path, download name)
REQUIRED_NLTK_RESOURCES = [
    ("tokenizers/punkt_tab", "punkt_tab"),   # tokenizer models for word_tokenize
    ("corpora/stopwords", "stopwords"),      # English stopword list
    ("corpora/wordnet", "wordnet"),          # dictionary used by the lemmatizer
    ("corpora/omw-1.4", "omw-1.4"),          # extra WordNet data
]

# Words that flip or change meaning ("I can NOT log in"). NLTK's stopword list
# contains them, so we put them back in.
NEGATION_WORDS = {"not", "no", "never", "nor", "cannot", "without"}

# Regular expressions used for cleaning
PLACEHOLDER_RE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")   # e.g. {{Order Number}}
NUMBER_RE = re.compile(r"\d+")                           # e.g. 12588
VALID_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_\-]*$")      # words, codes like "err-404"


class NLTKResourceError(RuntimeError):
    """Raised when NLTK data needed for preprocessing is not installed."""


def ensure_nltk_resources(download: bool = True) -> None:
    """Make sure the NLTK data is available, downloading it if allowed."""
    def is_installed(lookup_path: str) -> bool:
        # Some packages (e.g. wordnet) stay as .zip files, which NLTK can read
        for candidate in (lookup_path, lookup_path + ".zip"):
            try:
                nltk.data.find(candidate)
                return True
            except LookupError:
                continue
        return False

    missing = []
    for lookup_path, package in REQUIRED_NLTK_RESOURCES:
        if is_installed(lookup_path):
            continue
        if download:
            nltk.download(package, quiet=True)
            if is_installed(lookup_path):
                continue
        missing.append(package)
    if missing:
        raise NLTKResourceError(
            f"Missing NLTK data: {missing}. Run this command once with an internet "
            "connection:\n  python -m src.preprocess --setup"
        )


# Built lazily so importing this module never fails before the data is set up
_STOPWORDS: set[str] | None = None
_LEMMATIZER: WordNetLemmatizer | None = None


def _get_tools() -> tuple[set[str], WordNetLemmatizer]:
    global _STOPWORDS, _LEMMATIZER
    if _STOPWORDS is None or _LEMMATIZER is None:
        ensure_nltk_resources()
        _STOPWORDS = set(stopwords.words("english")) - NEGATION_WORDS
        _LEMMATIZER = WordNetLemmatizer()
    return _STOPWORDS, _LEMMATIZER


def get_stopwords() -> set[str]:
    """NLTK English stopwords, minus the negation words we keep."""
    return _get_tools()[0]


def preprocess_text(text) -> str:
    """
    Turn a raw customer message into a clean string of lemmatized tokens.

    Steps, with the reason for each:
      1. Placeholders: the dataset writes entities as {{Order Number}}. They are
         turned into plain words ("order number"), which is close to what a
         real customer would type.
      2. Numbers: replaced with the word "number", so "#12588" and "#99" look
         the same to the model (the exact number says nothing about the category).
      3. Lowercasing, so "Refund" and "refund" are the same word.
      4. Tokenization with NLTK's word_tokenize, which splits text into words
         and handles contractions ("can't" -> "ca", "n't").
      5. "n't" becomes "not", so negations are kept.
      6. Punctuation and symbols are removed. Words, and codes made of letters
         and digits, are kept.
      7. Stopwords ("the", "is", "to", ...) are removed, except negations
         such as "not", "no" and "never".
      8. Lemmatization with WordNet reduces words to their base form
         ("cancelling" -> "cancel", "orders" -> "order").

    Missing or empty input returns an empty string instead of failing.
    """
    if text is None or (isinstance(text, float) and text != text):  # None or NaN
        return ""
    text = str(text)
    if not text.strip():
        return ""

    stop_words, lemmatizer = _get_tools()

    text = PLACEHOLDER_RE.sub(lambda m: f" {m.group(1)} ", text)   # step 1
    text = NUMBER_RE.sub(" number ", text)                         # step 2
    text = text.lower().replace("’", "'")                     # step 3
    text = text.replace("can't", "can not").replace("won't", "will not")
    tokens = word_tokenize(text)                                   # step 4

    cleaned = []
    for token in tokens:
        if token in ("n't", "nt"):                                 # step 5
            token = "not"
        if not VALID_TOKEN_RE.match(token):                        # step 6
            continue
        if token in stop_words:                                    # step 7
            continue
        lemma = lemmatizer.lemmatize(token, pos="v")               # step 8 (verb form)
        lemma = lemmatizer.lemmatize(lemma, pos="n")               #        (noun form)
        cleaned.append(lemma)
    return " ".join(cleaned)


def main():
    parser = argparse.ArgumentParser(description="NLTK preprocessing utilities.")
    parser.add_argument("--setup", action="store_true", help="download required NLTK data")
    args = parser.parse_args()

    if args.setup:
        ensure_nltk_resources(download=True)
        print("All NLTK resources are installed.")

    examples = [
        "I was charged twice for my monthly subscription!!",
        "I can't access my account after resetting my password",
        "hw can I modify purchase {{Order Number}}?",
        "how can I check bill  #12588?",
        "I never received a refund for the cancelled orders",
    ]
    print("\nPreprocessing examples:")
    for example in examples:
        print(f"  RAW : {example}")
        print(f"  NLTK: {preprocess_text(example)}\n")


if __name__ == "__main__":
    main()

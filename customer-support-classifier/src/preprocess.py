"""
Module 2: NLP text preprocessing with NLTK.

The same `preprocess_text` function is used during training and when the web
app classifies a new message: it is plugged into the TF-IDF vectorizer, which
is saved inside the trained model.

    python -m src.preprocess --setup   # download the NLTK data once
    python -m src.preprocess           # show preprocessing examples
"""
import argparse
import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# NLTK data this module needs: (lookup path, download name)
REQUIRED_NLTK_DATA = [
    ("tokenizers/punkt_tab", "punkt_tab"),   # tokenizer used by word_tokenize
    ("corpora/stopwords", "stopwords"),      # English stopword list
    ("corpora/wordnet", "wordnet"),          # dictionary used by the lemmatizer
    ("corpora/omw-1.4", "omw-1.4"),          # extra WordNet data
]

# Words that change the meaning ("I can NOT log in"). NLTK's stopword list
# contains them, so we keep them.
NEGATION_WORDS = {"not", "no", "never", "nor", "cannot", "without"}

PLACEHOLDER_RE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")   # e.g. {{Order Number}}
NUMBER_RE = re.compile(r"\d+")                           # e.g. 12588
VALID_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_\-]*$")      # words and codes like "err-404"


class NLTKResourceError(RuntimeError):
    """Raised when NLTK data needed for preprocessing is not installed."""


def _is_installed(path: str) -> bool:
    # Some packages (e.g. wordnet) stay as .zip files, which NLTK can still read
    for candidate in (path, path + ".zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            pass
    return False


def ensure_nltk_resources():
    """Download any missing NLTK data; raise a clear error if that fails."""
    missing = []
    for path, package in REQUIRED_NLTK_DATA:
        if not _is_installed(path):
            nltk.download(package, quiet=True)
            if not _is_installed(path):
                missing.append(package)
    if missing:
        raise NLTKResourceError(
            f"Missing NLTK data: {missing}. Run this command once with an internet "
            "connection:\n  python -m src.preprocess --setup"
        )


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset:
    """NLTK English stopwords, minus the negation words we keep."""
    ensure_nltk_resources()
    return frozenset(stopwords.words("english")) - NEGATION_WORDS


@lru_cache(maxsize=1)
def get_lemmatizer() -> WordNetLemmatizer:
    ensure_nltk_resources()
    return WordNetLemmatizer()


def preprocess_text(text) -> str:
    """
    Turn a raw customer message into a clean string of lemmatized words.

      1. Placeholders such as {{Order Number}} become plain words ("order number").
      2. Numbers become the word "number" ("#12588" and "#99" mean the same here).
      3. Lowercase, and expand "can't" / "won't" so the negation is kept.
      4. Split into words with NLTK's word_tokenize.
      5. "n't" becomes "not".
      6. Remove punctuation and symbols (keep words and codes like "err-404").
      7. Remove stopwords ("the", "is", "to", ...) but keep negations.
      8. Lemmatize with WordNet ("cancelling" -> "cancel", "orders" -> "order").

    Missing or empty input returns an empty string instead of failing.
    """
    if not isinstance(text, str) or not text.strip():   # None, NaN or empty
        return ""
    stop_words, lemmatizer = get_stopwords(), get_lemmatizer()

    text = PLACEHOLDER_RE.sub(lambda m: f" {m.group(1)} ", text)            # 1
    text = NUMBER_RE.sub(" number ", text)                                  # 2
    text = text.lower().replace("’", "'")                                   # 3
    text = text.replace("can't", "can not").replace("won't", "will not")

    words = []
    for token in word_tokenize(text):                                       # 4
        if token in ("n't", "nt"):                                          # 5
            token = "not"
        if not VALID_TOKEN_RE.match(token) or token in stop_words:          # 6, 7
            continue
        lemma = lemmatizer.lemmatize(token, pos="v")                        # 8 (verb form)
        words.append(lemmatizer.lemmatize(lemma, pos="n"))                  #   (noun form)
    return " ".join(words)


def main():
    parser = argparse.ArgumentParser(description="NLTK preprocessing.")
    parser.add_argument("--setup", action="store_true", help="download the required NLTK data")
    if parser.parse_args().setup:
        ensure_nltk_resources()
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

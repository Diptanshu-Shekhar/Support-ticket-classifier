"""
Module 6: Live ticket classification.

Loads the saved model (NLTK preprocessing + TF-IDF + Linear SVM) and predicts
the category of new messages without retraining.

    python -m src.predict "I was charged twice for my order" "I forgot my password"
"""
import sys
from dataclasses import dataclass

import joblib

from src import config
from src.preprocess import get_stopwords

MAX_MESSAGE_LENGTH = 2000  # characters; protects against accidental huge pastes


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file does not exist yet."""


@dataclass
class Prediction:
    """Result of classifying one message."""
    text: str
    category: str
    confidence: float                 # probability of the predicted category
    probabilities: dict[str, float]   # every category -> probability, highest first
    known_terms: list[str]            # terms of the message found in the TF-IDF vocabulary
    review_reasons: list[str]         # why the ticket needs a human (empty = auto-route)

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)


def load_model() -> dict:
    """Load the model saved by train.py."""
    if not config.MODEL_PATH.exists():
        raise ModelNotFoundError(f"No trained model at {config.MODEL_PATH}.\nTrain it first with:  python -m src.train")
    return joblib.load(config.MODEL_PATH)


def validate_message(text) -> str:
    """Reject empty or overly long input with a clear message."""
    text = str(text or "").strip()
    if not text:
        raise ValueError("Please enter a customer message - the input is empty.")
    if len(text) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Message is too long ({len(text)} characters, max {MAX_MESSAGE_LENGTH}).")
    return text


def classify_message(text, model: dict, threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD) -> Prediction:
    """
    Predict the category of one message.

    The saved pipeline applies the same NLTK preprocessing and TF-IDF vectorizer
    as training. The ticket is flagged for manual review when none of its words
    are known to the model, when it only contains common words, or when the
    confidence is below the threshold. A high confidence does NOT guarantee a
    correct answer: the model always picks one of its 11 categories.
    """
    text = validate_message(text)
    pipeline = model["pipeline"]
    vectorizer = pipeline.named_steps["tfidf"]

    probs = pipeline.predict_proba([text])[0]
    probabilities = dict(sorted(zip(pipeline.classes_, probs.tolist()), key=lambda item: -item[1]))
    category, confidence = next(iter(probabilities.items()))

    # Which terms of the message does the model know, and do any of them carry meaning?
    known_terms = sorted({t for t in vectorizer.build_analyzer()(text) if t in vectorizer.vocabulary_})
    stop_words = get_stopwords()
    meaningful = [t for t in known_terms if any(w not in stop_words for w in t.split())]

    reasons = []
    if not known_terms:
        reasons.append("None of the words in this message appear in the training vocabulary.")
    elif not meaningful:
        reasons.append("The model only recognised common words (e.g. 'the', 'is', 'my') - "
                       "the message may be unrelated to customer support.")
    if confidence < threshold:
        reasons.append(f"Confidence {confidence:.1%} is below the review threshold of {threshold:.0%}.")

    return Prediction(text, str(category), confidence, probabilities, known_terms, reasons)


def print_prediction(result: Prediction):
    top3 = ", ".join(f"{c} {p:.1%}" for c, p in list(result.probabilities.items())[:3])
    print(f"\nMessage            : {result.text}")
    print(f"Predicted category : {result.category}")
    print(f"Confidence         : {result.confidence:.1%}")
    print(f"Top 3              : {top3}")
    print(f"Known terms        : {', '.join(result.known_terms) or '(none)'}")
    if result.needs_review:
        print("STATUS             : FLAGGED FOR MANUAL REVIEW")
        for reason in result.review_reasons:
            print(f"                     - {reason}")
    else:
        print("STATUS             : auto-routed")


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.predict "customer message" ["another message" ...]')
        sys.exit(1)
    try:
        model = load_model()
    except ModelNotFoundError as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    print(f"Loaded model: NLTK preprocessing + TF-IDF + {model['model_name']}")
    for message in sys.argv[1:]:
        try:
            print_prediction(classify_message(message, model))
        except ValueError as error:
            print(f"ERROR: {error}")


if __name__ == "__main__":
    main()

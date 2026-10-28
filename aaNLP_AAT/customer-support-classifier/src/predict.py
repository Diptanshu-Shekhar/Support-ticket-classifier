"""
Module 6: Live ticket classification.

Loads the saved pipeline (preprocessing + TF-IDF + classifier) and predicts
the category of new messages without retraining.

    python -m src.predict "I was charged twice for my order"
    python -m src.predict            # interactive mode, type messages one by one
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    confidence: float | None                 # None if the model has no probabilities
    probabilities: dict[str, float]          # every category -> probability
    known_terms: list[str] = field(default_factory=list)   # input terms found in the vocabulary
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)


def load_model(path: Path = config.MODEL_PATH) -> dict:
    """Load the saved artifact produced by train.py."""
    if not Path(path).exists():
        raise ModelNotFoundError(
            f"No trained model at {path}.\nTrain it first with:  python -m src.train"
        )
    return joblib.load(path)


def validate_message(text) -> str:
    """Reject empty or overly long input with a clear message."""
    if text is None or not str(text).strip():
        raise ValueError("Please enter a customer message - the input is empty.")
    text = str(text).strip()
    if len(text) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Message is too long ({len(text)} characters, max {MAX_MESSAGE_LENGTH}).")
    return text


def classify_message(
    text: str,
    artifact: dict,
    threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
) -> Prediction:
    """
    Predict the category for one message.

    The pipeline applies exactly the same preprocessing and the same fitted
    TF-IDF vectorizer as training. The ticket is flagged for manual review when:
      - no word of the message is in the model's vocabulary (the model has
        nothing to go on, e.g. gibberish or an unrelated topic), or
      - the top probability is below the threshold.
    Note: a high confidence does NOT guarantee the prediction is correct, and
    the model will always pick one of its 11 categories, even for off-topic text.
    """
    text = validate_message(text)
    pipeline = artifact["pipeline"]
    vectorizer = pipeline.named_steps["tfidf"]

    # Which terms of the message does the model actually know?
    analyzer = vectorizer.build_analyzer()
    known_terms = sorted({term for term in analyzer(text) if term in vectorizer.vocabulary_})

    category = str(pipeline.predict([text])[0])
    probabilities, confidence = {}, None
    if artifact.get("supports_probability") and hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([text])[0]
        probabilities = {str(c): float(p) for c, p in zip(pipeline.classes_, probs)}
        probabilities = dict(sorted(probabilities.items(), key=lambda item: -item[1]))
        confidence = probabilities[category]

    # Known terms that carry meaning (not only words like "the", "is", "my")
    stop_words = get_stopwords()
    meaningful_terms = [t for t in known_terms if any(w not in stop_words for w in t.split())]

    reasons = []
    if not known_terms:
        reasons.append("None of the words in this message appear in the training vocabulary.")
    elif not meaningful_terms:
        reasons.append(
            "The model only recognised common words (e.g. 'the', 'is', 'my') - "
            "the message may be unrelated to customer support."
        )
    if confidence is not None and confidence < threshold:
        reasons.append(f"Confidence {confidence:.1%} is below the review threshold of {threshold:.0%}.")

    return Prediction(
        text=text,
        category=category,
        confidence=confidence,
        probabilities=probabilities,
        known_terms=known_terms,
        needs_review=bool(reasons),
        review_reasons=reasons,
    )


def _print_prediction(result: Prediction):
    print(f"\nMessage            : {result.text}")
    print(f"Predicted category : {result.category}")
    if result.confidence is not None:
        print(f"Confidence         : {result.confidence:.1%}")
        top3 = list(result.probabilities.items())[:3]
        print("Top 3              : " + ", ".join(f"{c} {p:.1%}" for c, p in top3))
    print(f"Known terms        : {', '.join(result.known_terms) or '(none)'}")
    if result.needs_review:
        print("STATUS             : FLAGGED FOR MANUAL REVIEW")
        for reason in result.review_reasons:
            print(f"                     - {reason}")
    else:
        print("STATUS             : auto-routed")


def main():
    try:
        artifact = load_model()
    except ModelNotFoundError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
    print(f"Loaded model: NLTK preprocessing + TF-IDF + {artifact['model_name']}")

    messages = sys.argv[1:]
    if messages:
        for message in messages:
            try:
                _print_prediction(classify_message(message, artifact))
            except ValueError as error:
                print(f"ERROR: {error}")
        return

    print("Type a customer message and press Enter (empty line to quit).")
    while True:
        message = input("\n> ").strip()
        if not message:
            break
        _print_prediction(classify_message(message, artifact))


if __name__ == "__main__":
    main()

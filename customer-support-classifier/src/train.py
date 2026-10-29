"""
Modules 3 and 4: TF-IDF feature extraction and model training.

    python -m src.train

Model: NLTK preprocessing -> TF-IDF -> Linear SVM

Steps:
  1. Load and clean the dataset (remove empty rows and repeated messages).
  2. Split it into train (70%) / validation (15%) / test (15%), stratified.
  3. 5-fold cross-validation on the training set.
  4. Train on the training set and score on the validation set.
  5. Evaluate once on the untouched test set.
  6. Extra checks: do repeated messages inflate the score, and how does the
     model do on hand-written messages that are not in the dataset?
  7. Save the trained model and the results used by the web app.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config
from src.data_loader import DatasetError, clean_dataset, load_raw_dataset, normalise
from src.evaluate import METRICS_TO_PRINT, compute_metrics, evaluate_on_test
from src.preprocess import NLTKResourceError, ensure_nltk_resources, preprocess_text

MODEL_NAME = "Linear SVM"

# Messages written by us (not from the dataset) with the category we expect
HANDWRITTEN_EXAMPLES = [
    ("I cannot access my account after resetting my password.", "ACCOUNT"),
    ("I forgot my password", "ACCOUNT"),
    ("I'd like to delete my profile permanently", "ACCOUNT"),
    ("My parcel still hasn't arrived and it's been two weeks", "DELIVERY"),
    ("What shipping methods do you offer to Canada?", "DELIVERY"),
    ("I want my money back for the broken headphones", "REFUND"),
    ("Has my refund been processed yet?", "REFUND"),
    ("How do I unsubscribe from your marketing emails?", "SUBSCRIPTION"),
    ("Please change the address on order 55821 to 12 Baker Street", "SHIPPING"),
    ("Set up a new delivery address for my profile", "SHIPPING"),
    ("Your app keeps failing when I try to pay with PayPal", "PAYMENT"),
    ("Which credit cards do you accept?", "PAYMENT"),
    ("Can I talk to a real person please", "CONTACT"),
    ("How much does it cost to terminate my contract early?", "CANCEL"),
    ("Where can I download the receipt for last month's purchase?", "INVOICE"),
    ("I need to see my bill from March", "INVOICE"),
    ("I want to buy two more of these jackets", "ORDER"),
    ("I'd like to stop my order before it ships", "ORDER"),
    ("Your service was terrible and I want to file a complaint", "FEEDBACK"),
    ("Leave a review for the product I bought", "FEEDBACK"),
]


def build_pipeline() -> Pipeline:
    """
    NLTK preprocessing -> TF-IDF -> Linear SVM, all in one object.

    - The NLTK function is plugged into the TF-IDF vectorizer, so it is saved
      with the model and reused for every prediction.
    - The vectorizer is fitted on training data only (also inside every
      cross-validation fold), so no information leaks from validation/test data.
    - Linear SVM finds the widest-margin boundary between categories;
      class_weight="balanced" gives small categories more weight. It has no
      probabilities of its own, so a calibrator (3-fold, training data only)
      turns its scores into confidence values.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,   # NLTK: clean, tokenize, remove stopwords, lemmatize
        tokenizer=str.split,            # the text is already tokenized by NLTK
        token_pattern=None,
        lowercase=False,                # already lowercased by NLTK step
        **config.TFIDF_PARAMS,
    )
    classifier = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3)
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def split_data(data: pd.DataFrame):
    """Stratified 70 / 15 / 15 split with a fixed random seed."""
    X, y = data[config.TEXT_COLUMN], data[config.TARGET_COLUMN]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=config.VALIDATION_SIZE / (1 - config.TEST_SIZE),
        stratify=y_rest, random_state=config.RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test


def cross_validate_model(X_train, y_train) -> dict:
    """5-fold stratified cross-validation on the training set."""
    folds = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_validate(build_pipeline(), X_train, y_train, cv=folds, scoring=["accuracy", "f1_macro"])
    accuracy, macro_f1 = scores["test_accuracy"], scores["test_f1_macro"]
    return {
        "folds": config.CV_FOLDS,
        "accuracy_per_fold": accuracy.round(4).tolist(),
        "macro_f1_per_fold": macro_f1.round(4).tolist(),
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_std": float(accuracy.std()),
        "macro_f1_mean": float(macro_f1.mean()),
        "macro_f1_std": float(macro_f1.std()),
    }


def top_terms_per_category(pipeline: Pipeline, n_terms: int = 8) -> dict:
    """Terms with the largest SVM weights per category (what the model learned)."""
    calibrated = pipeline.named_steps["classifier"]
    # The calibrator holds 3 SVMs (one per internal fold); average their weights
    weights = np.mean([c.estimator.coef_ for c in calibrated.calibrated_classifiers_], axis=0)
    vocabulary = pipeline.named_steps["tfidf"].get_feature_names_out()
    return {category: vocabulary[np.argsort(row)[::-1][:n_terms]].tolist()
            for category, row in zip(calibrated.classes_, weights)}


def duplicate_check(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Do repeated messages inflate the test score? Train and test (80/20) three ways:
      A. duplicates kept, B. exact duplicates removed (our method),
      C. messages that are identical after NLTK preprocessing kept in the same split.
    """
    rows = []

    def run(name, data, train_idx, test_idx):
        X, y = data[config.TEXT_COLUMN], data[config.TARGET_COLUMN]
        processed = X.map(preprocess_text)
        pipeline = build_pipeline().fit(X.iloc[train_idx], y.iloc[train_idx])
        metrics = compute_metrics(y.iloc[test_idx], pipeline.predict(X.iloc[test_idx]))
        rows.append({
            "setting": name,
            "test_msgs_also_in_train_exact": int(normalise(X.iloc[test_idx]).isin(set(normalise(X.iloc[train_idx]))).sum()),
            "test_msgs_also_in_train_after_preprocessing": int(processed.iloc[test_idx].isin(set(processed.iloc[train_idx])).sum()),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        })

    def random_split(data):
        return train_test_split(np.arange(len(data)), test_size=0.2,
                                stratify=data[config.TARGET_COLUMN], random_state=config.RANDOM_STATE)

    with_duplicates, _ = clean_dataset(raw, drop_duplicates=False)
    run("A. Duplicates kept (random split)", with_duplicates, *random_split(with_duplicates))

    deduped, _ = clean_dataset(raw)
    run("B. Exact duplicates removed (final method)", deduped, *random_split(deduped))

    groups = deduped[config.TEXT_COLUMN].map(preprocess_text)
    grouped = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(grouped.split(deduped, deduped[config.TARGET_COLUMN], groups))
    run("C. Grouped by preprocessed text (strictest)", deduped, train_idx, test_idx)
    return pd.DataFrame(rows)


def main():
    try:
        ensure_nltk_resources()
        raw = load_raw_dataset()
    except (DatasetError, NLTKResourceError) as error:
        print(f"ERROR: {error}")
        sys.exit(1)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model: NLTK preprocessing -> TF-IDF -> {MODEL_NAME}\n")

    # 1. Clean
    data, cleaning = clean_dataset(raw)
    print(f"[1/7] Dataset: {cleaning['rows_before']} rows -> {cleaning['rows_after']} after cleaning "
          f"({cleaning['rows_removed_duplicates']} repeated messages removed)")

    # 2. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    labels = sorted(data[config.TARGET_COLUMN].unique())
    print(f"[2/7] Split: train={len(X_train)}  validation={len(X_val)}  test={len(X_test)}  "
          f"({len(labels)} categories)")

    # 3. Cross-validation on the training set
    print(f"[3/7] {config.CV_FOLDS}-fold cross-validation on the training set...")
    cv_results = cross_validate_model(X_train, y_train)
    print(f"      macro-F1 per fold: {cv_results['macro_f1_per_fold']}")
    print(f"      mean macro-F1: {cv_results['macro_f1_mean']:.4f} (+/- {cv_results['macro_f1_std']:.4f})")

    # 4. Train, then score on the validation set
    print("[4/7] Training on the training set and scoring on the validation set...")
    pipeline = build_pipeline().fit(X_train, y_train)
    train_metrics = compute_metrics(y_train, pipeline.predict(X_train))
    val_metrics = compute_metrics(y_val, pipeline.predict(X_val))
    print(f"      train accuracy: {train_metrics['accuracy']:.4f}   "
          f"validation accuracy: {val_metrics['accuracy']:.4f}   "
          f"validation macro-F1: {val_metrics['macro_f1']:.4f}")

    # 5. One final evaluation on the held-out test set
    print("[5/7] Evaluating on the held-out test set...")
    test_metrics = evaluate_on_test(pipeline, X_test, y_test, labels)
    for key in METRICS_TO_PRINT:
        print(f"      {key:<18}{test_metrics[key]:.4f}")
    print("      weakest categories:", test_metrics["weakest_categories"])

    # 6. Extra checks
    print("[6/7] Duplicate-leakage check...")
    print(duplicate_check(raw).round(4).to_string(index=False))
    texts = [text for text, _ in HANDWRITTEN_EXAMPLES]
    correct = sum(p == label for p, (_, label) in zip(pipeline.predict(texts), HANDWRITTEN_EXAMPLES))
    print(f"      Hand-written messages: {correct} / {len(texts)} correct")

    # 7. Save the model and the results shown in the web app
    joblib.dump({"pipeline": pipeline, "model_name": MODEL_NAME, "labels": labels}, config.MODEL_PATH)
    pd.DataFrame({config.TEXT_COLUMN: X_test, config.TARGET_COLUMN: y_test}).to_csv(config.TEST_SPLIT_PATH, index=False)
    results = {
        "model": f"NLTK + TF-IDF + {MODEL_NAME}",
        "cleaning": cleaning,
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "cross_validation": cv_results,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "top_terms_per_category": top_terms_per_category(pipeline),
    }
    (config.REPORTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[7/7] Saved model to {config.MODEL_PATH}")
    print(f"      Saved evaluation results to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()

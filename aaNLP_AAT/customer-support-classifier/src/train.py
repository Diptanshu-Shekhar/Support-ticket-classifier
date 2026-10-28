"""
Modules 3 and 4: TF-IDF feature extraction and model training.

    python -m src.train

Model: NLTK preprocessing -> TF-IDF -> Linear SVM

What this script does:
  1. Loads and cleans the dataset (removes empty rows and repeated messages).
  2. Splits it into train (70%) / validation (15%) / test (15%), stratified.
  3. Builds the pipeline: NLTK preprocessing -> TF-IDF -> Linear SVM.
  4. Validates it with 5-fold cross-validation on the training set, then
     trains it on the training set and scores it on the validation set.
  5. Evaluates it once on the untouched test set.
  6. Runs a duplicate-leakage check to confirm repeated messages do not
     inflate the score.
  7. Tests it on hand-written messages that are not in the dataset.
  8. Saves the trained pipeline, test split and all evaluation results.
"""
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config
from src.data_loader import DatasetError, clean_dataset, load_raw_dataset
from src.evaluate import compute_metrics, evaluate_on_test
from src.preprocess import NLTKResourceError, ensure_nltk_resources, preprocess_text

MODEL_NAME = "Linear SVM"

# Hand-written messages that do NOT come from the dataset, with the category we
# expect. They test whether the model copes with phrasing different from the
# dataset. Only 20 examples, so this is an extra sanity check, not a benchmark.
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

    - The NLTK function `preprocess_text` is plugged into the TF-IDF
      vectorizer, so it is saved with the model and reused for every prediction.
    - The vectorizer is fitted on training data only (also inside every
      cross-validation fold) and only reused to transform validation/test
      data, so no information leaks from them.
    - Linear SVM finds the boundary with the widest margin between categories.
      class_weight="balanced" gives small categories more weight. LinearSVC
      has no probabilities of its own, so it is wrapped in a calibrator
      (fitted with internal 3-fold CV on training data only) to produce
      confidence scores.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,   # NLTK cleaning, tokenizing, stopwords, lemmatizing
        tokenizer=str.split,            # text is already tokenized by NLTK
        token_pattern=None,
        lowercase=False,                # already lowercased by the NLTK step
        **config.TFIDF_PARAMS,
    )
    classifier = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3)
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def split_data(data: pd.DataFrame):
    """Stratified 70 / 15 / 15 split with a fixed random seed."""
    X, y = data[config.TEXT_COLUMN], data[config.TARGET_COLUMN]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    val_fraction = config.VALIDATION_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_fraction, stratify=y_rest, random_state=config.RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def cross_validate_model(X_train, y_train) -> dict:
    """5-fold stratified cross-validation on the training set."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_validate(build_pipeline(), X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1)
    return {
        "folds": config.CV_FOLDS,
        "accuracy_per_fold": scores["test_accuracy"].round(4).tolist(),
        "macro_f1_per_fold": scores["test_f1_macro"].round(4).tolist(),
        "accuracy_mean": float(scores["test_accuracy"].mean()),
        "accuracy_std": float(scores["test_accuracy"].std()),
        "macro_f1_mean": float(scores["test_f1_macro"].mean()),
        "macro_f1_std": float(scores["test_f1_macro"].std()),
    }


def top_terms_per_category(pipeline: Pipeline, n_terms: int = 8) -> dict:
    """Terms with the largest SVM weights per category (explains what the model learned)."""
    vectorizer = pipeline.named_steps["tfidf"]
    calibrated = pipeline.named_steps["classifier"]
    # The calibrator holds 3 SVMs (one per internal fold); average their weights
    weights = np.mean([c.estimator.coef_ for c in calibrated.calibrated_classifiers_], axis=0)
    vocabulary = np.array(vectorizer.get_feature_names_out())
    return {
        category: vocabulary[np.argsort(row)[::-1][:n_terms]].tolist()
        for category, row in zip(calibrated.classes_, weights)
    }


def duplicate_leakage_experiment(raw: pd.DataFrame) -> list[dict]:
    """
    Do repeated messages inflate the test score?

      A. Keep duplicates, random stratified 80/20 split.
      B. Remove exact duplicates first (the method used for the final model).
      C. Remove duplicates AND keep messages that become identical after NLTK
         preprocessing (near-duplicates) in the same split, using grouped splitting.
    """
    results = []

    def run(name, data, train_idx, test_idx):
        X, y = data[config.TEXT_COLUMN].reset_index(drop=True), data[config.TARGET_COLUMN].reset_index(drop=True)
        processed = X.map(preprocess_text)
        train_raw = set(X.iloc[train_idx].str.lower().str.split().str.join(" "))
        train_processed = set(processed.iloc[train_idx])
        test_raw_norm = X.iloc[test_idx].str.lower().str.split().str.join(" ")

        pipeline = build_pipeline().fit(X.iloc[train_idx], y.iloc[train_idx])
        metrics = compute_metrics(y.iloc[test_idx], pipeline.predict(X.iloc[test_idx]))
        results.append({
            "setting": name,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "test_msgs_also_in_train_exact": int(test_raw_norm.isin(train_raw).sum()),
            "test_msgs_also_in_train_after_preprocessing": int(processed.iloc[test_idx].isin(train_processed).sum()),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        })

    def stratified_split(data):
        indices = np.arange(len(data))
        return train_test_split(
            indices, test_size=0.2, stratify=data[config.TARGET_COLUMN], random_state=config.RANDOM_STATE
        )

    # A: duplicates kept
    with_dups, _ = clean_dataset(raw, drop_duplicates=False)
    run("A. Duplicates kept (random split)", with_dups, *stratified_split(with_dups))

    # B: exact duplicates removed
    deduped, _ = clean_dataset(raw, drop_duplicates=True)
    run("B. Exact duplicates removed (final method)", deduped, *stratified_split(deduped))

    # C: grouped split on preprocessed text
    groups = deduped[config.TEXT_COLUMN].map(preprocess_text)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(deduped, deduped[config.TARGET_COLUMN], groups))
    run("C. Grouped by preprocessed text (strictest)", deduped, train_idx, test_idx)
    return results


def main():
    try:
        ensure_nltk_resources()
        raw = load_raw_dataset()
    except (DatasetError, NLTKResourceError) as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model: NLTK preprocessing -> TF-IDF -> {MODEL_NAME}\n")

    # 1. Clean
    data, clean_report = clean_dataset(raw, drop_duplicates=True)
    print(f"[1/7] Dataset: {clean_report['rows_before']} rows -> {clean_report['rows_after']} after cleaning "
          f"({clean_report['rows_removed_duplicates']} repeated messages removed)")

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

    # 4. Train on the training set, score on the validation set
    print("[4/7] Training on the training set and scoring on the validation set...")
    pipeline = build_pipeline().fit(X_train, y_train)
    train_metrics = compute_metrics(y_train, pipeline.predict(X_train))
    val_metrics = compute_metrics(y_val, pipeline.predict(X_val))
    print(f"      train accuracy: {train_metrics['accuracy']:.4f}   "
          f"validation accuracy: {val_metrics['accuracy']:.4f}   "
          f"validation macro-F1: {val_metrics['macro_f1']:.4f}")

    # 5. Final, one-time evaluation on the held-out test set
    print("[5/7] Evaluating on the held-out test set...")
    test_metrics = evaluate_on_test(pipeline, X_test, y_test, labels)
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        print(f"      {key:<18}{test_metrics[key]:.4f}")
    print("      weakest categories:", test_metrics["weakest_categories"])

    # 6. Duplicate-leakage check
    print("[6/7] Duplicate-leakage check...")
    leakage = duplicate_leakage_experiment(raw)
    pd.DataFrame(leakage).round(4).to_csv(config.REPORTS_DIR / "duplicate_experiment.csv", index=False)
    print(pd.DataFrame(leakage)[["setting", "test_msgs_also_in_train_exact",
                                 "test_msgs_also_in_train_after_preprocessing", "accuracy", "macro_f1"]]
          .round(4).to_string(index=False))

    # Hand-written messages (extra check)
    texts = [text for text, _ in HANDWRITTEN_EXAMPLES]
    handwritten = pd.DataFrame({
        "message": texts,
        "expected": [label for _, label in HANDWRITTEN_EXAMPLES],
        "predicted": pipeline.predict(texts),
    })
    handwritten["correct"] = handwritten["expected"] == handwritten["predicted"]
    handwritten.to_csv(config.REPORTS_DIR / "handwritten_check.csv", index=False)
    print(f"      Hand-written messages: {int(handwritten['correct'].sum())} / {len(handwritten)} correct")

    # 7. Save everything needed for inference and for the web app
    artifact = {
        "pipeline": pipeline,
        "model_name": MODEL_NAME,
        "classifier": "linear_svm",
        "preprocessing": "NLTK",
        "labels": labels,
        "supports_probability": hasattr(pipeline, "predict_proba"),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, config.MODEL_PATH)
    pd.DataFrame({config.TEXT_COLUMN: X_test, config.TARGET_COLUMN: y_test}).to_csv(
        config.TEST_SPLIT_PATH, index=False
    )

    results = {
        "model": f"NLTK + TF-IDF + {MODEL_NAME}",
        "cleaning": clean_report,
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "split_distribution": {
            "train": y_train.value_counts().to_dict(),
            "validation": y_val.value_counts().to_dict(),
            "test": y_test.value_counts().to_dict(),
        },
        "tfidf_params": {k: list(v) if isinstance(v, tuple) else v for k, v in config.TFIDF_PARAMS.items()},
        "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "cross_validation": cv_results,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "handwritten_check_correct": int(handwritten["correct"].sum()),
        "handwritten_check_total": len(handwritten),
        "top_terms_per_category": top_terms_per_category(pipeline),
    }
    (config.REPORTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[7/7] Saved model to {config.MODEL_PATH}")
    print(f"      Saved evaluation results to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()

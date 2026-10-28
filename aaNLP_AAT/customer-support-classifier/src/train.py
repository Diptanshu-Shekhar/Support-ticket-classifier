"""
Modules 3 and 4: TF-IDF feature extraction and model training.

    python -m src.train

What this script does:
  1. Loads and cleans the dataset (removes empty rows and repeated messages).
  2. Splits it into train (70%) / validation (15%) / test (15%), stratified.
  3. Builds 6 candidate pipelines:
        {baseline lowercase only, full NLTK preprocessing}
      x {Multinomial Naive Bayes, Logistic Regression, Linear SVM}
     Each pipeline = preprocessing -> TF-IDF -> classifier.
  4. Scores each candidate with 5-fold cross-validation on the training set
     and on the validation set.
  5. Selects the deployed model by validation macro-F1 (default: from all 6
     candidates; --preprocessing can restrict the choice).
  6. Evaluates only that model, once, on the untouched test set.
  7. Runs a duplicate-leakage experiment to check whether repeated messages
     inflate scores.
  8. Runs a small robustness check on hand-written messages that are not in
     the dataset (qualitative only - not used for model selection).
  9. Saves the trained pipeline, test split and all reports.

Options:
    python -m src.train                        # deploy best of all 6 candidates (default)
    python -m src.train --preprocessing nltk   # deploy best NLTK-preprocessed candidate
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config
from src.data_loader import DatasetError, clean_dataset, load_raw_dataset
from src.evaluate import compute_metrics, evaluate_on_test, plot_model_comparison
from src.preprocess import NLTKResourceError, ensure_nltk_resources, preprocess_text

CLASSIFIER_NAMES = {
    "naive_bayes": "Multinomial Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
}
PREPROCESSING_NAMES = {"basic": "Baseline (lowercase only)", "nltk": "NLTK pipeline"}

# Hand-written messages that do NOT come from the dataset, with the category we
# expect. Written before looking at any model output. They test whether a
# model copes with phrasing different from the dataset's templates. Only 20
# examples, so this is a qualitative check, not a benchmark.
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


def make_classifier(name: str):
    """Create one of the three classifiers with fixed, explainable settings."""
    if name == "naive_bayes":
        # Probabilistic model based on word frequencies per category.
        # alpha = smoothing so unseen words don't get zero probability.
        return MultinomialNB(alpha=0.1)
    if name == "logistic_regression":
        # Linear model that outputs class probabilities (softmax over 11 classes).
        # class_weight="balanced" gives small categories more weight.
        return LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced")
    if name == "linear_svm":
        # Finds the boundary with the widest margin between categories.
        # LinearSVC has no probabilities, so it is wrapped in a calibrator
        # (fitted with internal 3-fold CV on training data only) to get them.
        return CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3)
    raise ValueError(f"Unknown classifier: {name}")


def build_pipeline(classifier_name: str, preprocessing: str = "nltk") -> Pipeline:
    """
    preprocessing -> TF-IDF -> classifier, all in one object.

    Because the vectorizer is inside the pipeline, it is fitted on training
    data only (also inside every cross-validation fold) and simply reused to
    transform validation/test data, so no information leaks from them.
    """
    if preprocessing == "nltk":
        # Text is cleaned by our NLTK function; TF-IDF splits on whitespace only.
        vectorizer = TfidfVectorizer(
            preprocessor=preprocess_text,
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            **config.TFIDF_PARAMS,
        )
    else:
        # Baseline: TF-IDF's own lowercasing and default word tokenizer only.
        vectorizer = TfidfVectorizer(lowercase=True, **config.TFIDF_PARAMS)
    return Pipeline([("tfidf", vectorizer), ("classifier", make_classifier(classifier_name))])


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


def compare_candidates(X_train, y_train, X_val, y_val) -> tuple[pd.DataFrame, dict]:
    """Cross-validate and validate every (preprocessing, classifier) combination."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rows, fitted = [], {}

    for preprocessing in ("basic", "nltk"):
        for clf_name in CLASSIFIER_NAMES:
            label = f"{CLASSIFIER_NAMES[clf_name]} | {PREPROCESSING_NAMES[preprocessing]}"
            print(f"  - {label}")
            start = time.time()

            pipeline = build_pipeline(clf_name, preprocessing)
            cv_scores = cross_validate(
                pipeline, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1
            )
            pipeline.fit(X_train, y_train)
            train_metrics = compute_metrics(y_train, pipeline.predict(X_train))
            val_metrics = compute_metrics(y_val, pipeline.predict(X_val))

            rows.append({
                "model": label,
                "classifier": clf_name,
                "preprocessing": preprocessing,
                "cv_accuracy_mean": cv_scores["test_accuracy"].mean(),
                "cv_macro_f1_mean": cv_scores["test_f1_macro"].mean(),
                "cv_macro_f1_std": cv_scores["test_f1_macro"].std(),
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
                "seconds": round(time.time() - start, 1),
            })
            fitted[(clf_name, preprocessing)] = pipeline

    return pd.DataFrame(rows), fitted


def top_terms_per_category(pipeline: Pipeline, n_terms: int = 8) -> dict:
    """Most positive Logistic Regression weights per category (explains what the model learned)."""
    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["classifier"]
    vocabulary = np.array(vectorizer.get_feature_names_out())
    return {
        category: vocabulary[np.argsort(weights)[::-1][:n_terms]].tolist()
        for category, weights in zip(model.classes_, model.coef_)
    }


def duplicate_leakage_experiment(raw: pd.DataFrame, clf_name: str, preprocessing: str) -> list[dict]:
    """
    How much do repeated messages inflate the test score?

      A. Keep duplicates, random stratified 80/20 split.
      B. Remove exact duplicates first (the method used for the final model).
      C. Remove duplicates AND keep messages that become identical after
         preprocessing (near-duplicates) in the same split, using grouped splitting.
    """
    results = []

    def run(name, data, train_idx, test_idx):
        X, y = data[config.TEXT_COLUMN].reset_index(drop=True), data[config.TARGET_COLUMN].reset_index(drop=True)
        processed = X.map(preprocess_text)
        train_raw = set(X.iloc[train_idx].str.lower().str.split().str.join(" "))
        train_processed = set(processed.iloc[train_idx])
        test_raw_norm = X.iloc[test_idx].str.lower().str.split().str.join(" ")

        pipeline = build_pipeline(clf_name, preprocessing).fit(X.iloc[train_idx], y.iloc[train_idx])
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


def handwritten_check(fitted: dict) -> pd.DataFrame:
    """Run every fitted candidate on the hand-written examples."""
    texts = [text for text, _ in HANDWRITTEN_EXAMPLES]
    expected = [label for _, label in HANDWRITTEN_EXAMPLES]
    table = pd.DataFrame({"message": texts, "expected": expected})
    for (clf_name, preprocessing), pipeline in fitted.items():
        column = f"{CLASSIFIER_NAMES[clf_name]} | {PREPROCESSING_NAMES[preprocessing]}"
        table[column] = pipeline.predict(texts)
    return table


def main():
    parser = argparse.ArgumentParser(description="Train and compare ticket classifiers.")
    parser.add_argument(
        "--preprocessing",
        choices=["nltk", "basic", "any"],
        default="any",
        help="which candidates the deployed model is chosen from (default: any)",
    )
    args = parser.parse_args()
    started = time.time()
    try:
        ensure_nltk_resources()
        raw = load_raw_dataset()
    except (DatasetError, NLTKResourceError) as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Clean
    data, clean_report = clean_dataset(raw, drop_duplicates=True)
    print(f"[1/7] Dataset: {clean_report['rows_before']} rows -> {clean_report['rows_after']} after cleaning "
          f"({clean_report['rows_removed_duplicates']} repeated messages removed)")

    # 2. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    labels = sorted(data[config.TARGET_COLUMN].unique())
    print(f"[2/7] Split: train={len(X_train)}  validation={len(X_val)}  test={len(X_test)}  "
          f"({len(labels)} categories)")

    # 3-4. Compare candidates
    print(f"[3/7] Training 6 candidates ({config.CV_FOLDS}-fold CV on train + validation set):")
    comparison, fitted = compare_candidates(X_train, y_train, X_val, y_val)
    comparison = comparison.sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    comparison.round(4).to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    plot_model_comparison(comparison, config.REPORTS_DIR / "model_comparison.png")
    print(comparison[["model", "cv_macro_f1_mean", "val_accuracy", "val_macro_f1", "train_accuracy"]]
          .round(4).to_string(index=False))

    # 5. Select by validation macro-F1 (never by training accuracy)
    best_overall = comparison.iloc[0]
    if args.preprocessing == "any":
        eligible = comparison
    else:
        eligible = comparison[comparison["preprocessing"] == args.preprocessing]
    best = eligible.iloc[0]
    best_key = (best["classifier"], best["preprocessing"])
    best_pipeline = fitted[best_key]
    print(f"\n[4/7] Best of all candidates (validation macro-F1): {best_overall['model']} "
          f"({best_overall['val_macro_f1']:.4f})")
    print(f"      Deployed model (preprocessing = {args.preprocessing}): {best['model']} "
          f"({best['val_macro_f1']:.4f})")

    # 6. Final, one-time evaluation on the held-out test set
    print("[5/7] Evaluating the selected model on the held-out test set...")
    test_metrics = evaluate_on_test(best_pipeline, X_test, y_test, labels)
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        print(f"      {key:<18}{test_metrics[key]:.4f}")
    print("      weakest categories:", test_metrics["weakest_categories"])

    # 7. Duplicate-leakage experiment
    print("[6/7] Duplicate-leakage experiment...")
    leakage = duplicate_leakage_experiment(raw, *best_key)
    pd.DataFrame(leakage).round(4).to_csv(config.REPORTS_DIR / "duplicate_experiment.csv", index=False)
    print(pd.DataFrame(leakage)[["setting", "test_msgs_also_in_train_exact",
                                 "test_msgs_also_in_train_after_preprocessing", "accuracy", "macro_f1"]]
          .round(4).to_string(index=False))

    # Robustness check on hand-written messages (reported, not used for selection)
    handwritten = handwritten_check(fitted)
    handwritten.to_csv(config.REPORTS_DIR / "handwritten_check.csv", index=False)
    model_columns = [c for c in handwritten.columns if c not in ("message", "expected")]
    handwritten_scores = {
        column: int((handwritten[column] == handwritten["expected"]).sum()) for column in model_columns
    }
    print(f"      Hand-written check (correct out of {len(handwritten)}):")
    for column, correct in sorted(handwritten_scores.items(), key=lambda item: -item[1]):
        print(f"        {correct:>2}  {column}")

    # 8. Save everything needed for inference and for the web app
    top_terms = top_terms_per_category(fitted[("logistic_regression", "nltk")])
    artifact = {
        "pipeline": best_pipeline,
        "model_name": best["model"],
        "classifier": best["classifier"],
        "preprocessing": best["preprocessing"],
        "labels": labels,
        "supports_probability": hasattr(best_pipeline, "predict_proba"),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, config.MODEL_PATH)
    pd.DataFrame({config.TEXT_COLUMN: X_test, config.TARGET_COLUMN: y_test}).to_csv(
        config.TEST_SPLIT_PATH, index=False
    )

    results = {
        "selected_model": best["model"],
        "selection_metric": "validation macro-F1",
        "selection_pool": args.preprocessing,
        "best_overall_model": best_overall["model"],
        "best_overall_val_macro_f1": float(best_overall["val_macro_f1"]),
        "selected_val_macro_f1": float(best["val_macro_f1"]),
        "handwritten_check_correct": handwritten_scores,
        "handwritten_check_total": len(handwritten),
        "cleaning": clean_report,
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "split_distribution": {
            "train": y_train.value_counts().to_dict(),
            "validation": y_val.value_counts().to_dict(),
            "test": y_test.value_counts().to_dict(),
        },
        "tfidf_params": {k: list(v) if isinstance(v, tuple) else v for k, v in config.TFIDF_PARAMS.items()},
        "test_metrics": test_metrics,
        "top_terms_logistic_regression": top_terms,
        "trained_at": artifact["trained_at"],
    }
    (config.REPORTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[7/7] Saved model to {config.MODEL_PATH}")
    print(f"      Saved reports to {config.REPORTS_DIR}")
    print(f"Done in {time.time() - started:.0f} s.")


if __name__ == "__main__":
    main()

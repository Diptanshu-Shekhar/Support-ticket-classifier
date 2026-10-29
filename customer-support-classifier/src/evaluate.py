"""
Module 5: Model evaluation.

Every number is calculated from real model predictions. train.py calls
evaluate_on_test() automatically; you can also re-evaluate the saved model on
the saved test set at any time:

    python -m src.evaluate
"""
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw charts to files without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from src import config

METRICS_TO_PRINT = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy plus macro- and weighted-averaged precision, recall and F1."""
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
    }


def classification_report_frame(y_true, y_pred, labels) -> pd.DataFrame:
    """Per-category precision / recall / F1 / support as a table."""
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    frame = pd.DataFrame(report).T
    frame["support"] = frame["support"].astype(int)
    # sklearn stores accuracy as one number; show it only in the f1 column, with the total support
    frame.loc["accuracy", ["precision", "recall"]] = float("nan")
    frame.loc["accuracy", "support"] = len(y_true)
    return frame.round(4)


def plot_confusion_matrix(y_true, y_pred, labels, save_path: Path, normalize: bool = False):
    """Heatmap: rows = true category, columns = predicted category."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("Confusion matrix (row-normalized)" if normalize else "Confusion matrix (counts)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def evaluate_on_test(pipeline, X_test, y_test, labels) -> dict:
    """Score the model on the held-out test set and save the reports used by the web app."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    report = classification_report_frame(y_test, y_pred, labels)
    report.to_csv(config.REPORTS_DIR / "classification_report.csv")
    plot_confusion_matrix(y_test, y_pred, labels, config.REPORTS_DIR / "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, labels, config.REPORTS_DIR / "confusion_matrix_normalized.png", normalize=True)

    # Every misclassified message, for error analysis
    results = pd.DataFrame({"message": list(X_test), "true": list(y_test), "predicted": list(y_pred)})
    mistakes = results[results["true"] != results["predicted"]]
    mistakes.to_csv(config.REPORTS_DIR / "misclassified_test.csv", index=False)

    weakest = report.loc[labels, "f1-score"].sort_values().head(3)
    confused = mistakes[["true", "predicted"]].value_counts().head(5)
    metrics["weakest_categories"] = [{"category": name, "f1": float(f1)} for name, f1 in weakest.items()]
    metrics["most_confused_pairs"] = [{"true": t, "predicted": p, "count": int(c)} for (t, p), c in confused.items()]
    metrics["num_test_samples"] = len(y_test)
    metrics["num_errors"] = len(mistakes)
    return metrics


def main():
    """Re-evaluate the saved model on the saved test set."""
    if not config.MODEL_PATH.exists() or not config.TEST_SPLIT_PATH.exists():
        print("No trained model found. Train first with:  python -m src.train")
        sys.exit(1)

    model = joblib.load(config.MODEL_PATH)
    test = pd.read_csv(config.TEST_SPLIT_PATH)
    metrics = evaluate_on_test(model["pipeline"], test[config.TEXT_COLUMN], test[config.TARGET_COLUMN], model["labels"])

    print(f"Model: {model['model_name']}")
    print(f"Test samples: {metrics['num_test_samples']}  (errors: {metrics['num_errors']})")
    for key in METRICS_TO_PRINT:
        print(f"  {key:<18}{metrics[key]:.4f}")
    print("\nPer-category report:")
    print(pd.read_csv(config.REPORTS_DIR / "classification_report.csv", index_col=0).to_string())
    print(f"\nCharts saved in: {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()

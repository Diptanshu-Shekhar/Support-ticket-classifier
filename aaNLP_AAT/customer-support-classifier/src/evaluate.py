"""
Module 5: Model evaluation and validation.

Every number here is calculated from real model predictions; nothing is hardcoded.
train.py calls these functions automatically. You can also re-evaluate the
saved model on the saved held-out test set at any time:

    python -m src.evaluate
"""
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src import config


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
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    frame = pd.DataFrame(report).T
    frame["support"] = frame["support"].astype(int)
    return frame.round(4)


def plot_confusion_matrix(y_true, y_pred, labels, save_path: Path, normalize: bool = False):
    """Heatmap: rows = true category, columns = predicted category."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("Confusion matrix (row-normalized)" if normalize else "Confusion matrix (counts)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return matrix


def plot_model_comparison(comparison: pd.DataFrame, save_path: Path):
    """Grouped bar chart of validation metrics for every candidate model."""
    metrics = ["val_accuracy", "val_macro_f1", "val_weighted_f1"]
    long = comparison.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
    long["metric"] = long["metric"].str.replace("val_", "").str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long, x="model", y="score", hue="metric", ax=ax)
    # Zoom the y-axis so small differences are visible (noted in the title)
    lowest = long["score"].min()
    margin = max(1.0 - lowest, 0.002)
    ax.set_ylim(max(0.0, lowest - margin), 1.0 + margin * 0.15)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=7, rotation=90, padding=2)
    ax.set_xlabel("")
    ax.set_ylabel("Validation score")
    ax.set_title("Model comparison on the validation set (y-axis zoomed in)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_per_class_f1(report: pd.DataFrame, labels, save_path: Path):
    """Horizontal bar chart of F1 per category, weakest at the top."""
    scores = report.loc[labels, "f1-score"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=scores.values, y=scores.index, ax=ax, color="#55A868")
    ax.set_xlim(max(0.0, scores.min() - 0.05), 1.0)
    ax.set_xlabel("F1-score on test set")
    ax.set_ylabel("")
    ax.set_title("Per-category F1 (weakest first)")
    for i, value in enumerate(scores.values):
        ax.text(value, i, f" {value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def most_confused_pairs(y_true, y_pred, top_n: int = 5) -> list[dict]:
    """The most frequent (true -> predicted) mistakes."""
    errors = pd.DataFrame({"true": list(y_true), "predicted": list(y_pred)})
    errors = errors[errors["true"] != errors["predicted"]]
    pairs = errors.value_counts().head(top_n)
    return [{"true": t, "predicted": p, "count": int(c)} for (t, p), c in pairs.items()]


def evaluate_on_test(pipeline, X_test, y_test, labels, reports_dir: Path = config.REPORTS_DIR) -> dict:
    """Score a fitted pipeline on the held-out test set and save all reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    y_pred = pipeline.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    report = classification_report_frame(y_test, y_pred, labels)
    report.to_csv(reports_dir / "classification_report.csv")

    matrix = plot_confusion_matrix(y_test, y_pred, labels, reports_dir / "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, labels, reports_dir / "confusion_matrix_normalized.png", normalize=True)
    plot_per_class_f1(report, labels, reports_dir / "per_class_f1.png")

    per_class_f1 = report.loc[labels, "f1-score"].sort_values()
    metrics["weakest_categories"] = [
        {"category": name, "f1": float(score)} for name, score in per_class_f1.head(3).items()
    ]
    metrics["most_confused_pairs"] = most_confused_pairs(y_test, y_pred)
    metrics["num_test_samples"] = int(len(y_test))
    metrics["num_errors"] = int((pd.Series(list(y_test)) != pd.Series(list(y_pred))).sum())

    # Save every misclassified test message for error analysis
    mistakes = pd.DataFrame({"message": list(X_test), "true": list(y_test), "predicted": list(y_pred)})
    mistakes[mistakes["true"] != mistakes["predicted"]].to_csv(reports_dir / "misclassified_test.csv", index=False)

    (reports_dir / "confusion_matrix.json").write_text(
        json.dumps({"labels": list(labels), "matrix": matrix.tolist()}, indent=2)
    )
    return metrics


def main():
    """Re-evaluate the saved model on the saved held-out test split."""
    import joblib

    if not config.MODEL_PATH.exists() or not config.TEST_SPLIT_PATH.exists():
        print("No trained model found. Train first with:  python -m src.train")
        sys.exit(1)

    artifact = joblib.load(config.MODEL_PATH)
    test = pd.read_csv(config.TEST_SPLIT_PATH)
    metrics = evaluate_on_test(
        artifact["pipeline"], test[config.TEXT_COLUMN], test[config.TARGET_COLUMN], artifact["labels"]
    )
    print(f"Model: {artifact['model_name']}")
    print(f"Test samples: {metrics['num_test_samples']}  (errors: {metrics['num_errors']})")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        print(f"  {key:<18}{metrics[key]:.4f}")
    print("\nPer-category report:")
    print(pd.read_csv(config.REPORTS_DIR / "classification_report.csv", index_col=0).to_string())
    print(f"\nCharts saved in: {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()

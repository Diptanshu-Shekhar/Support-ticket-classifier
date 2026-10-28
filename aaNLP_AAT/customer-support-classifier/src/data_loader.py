"""
Module 1: Dataset loading and exploration.

Run it on its own to print a dataset report and save the category chart:

    python -m src.data_loader            # explore the dataset
    python -m src.data_loader --download # download the CSV first if missing
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Allow running as "python src/data_loader.py" as well as "python -m src.data_loader"
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # draw charts to files without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config


class DatasetError(Exception):
    """Raised when the dataset file is missing or has an unexpected format."""


def download_dataset(destination: Path = config.DATASET_PATH) -> Path:
    """Download the original Bitext CSV from GitHub into the dataset folder."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from:\n  {config.DATASET_URL}")
    urllib.request.urlretrieve(config.DATASET_URL, destination)
    print(f"Saved to: {destination}")
    return destination


def load_raw_dataset(path: Path = config.DATASET_PATH) -> pd.DataFrame:
    """Read the CSV exactly as published and check that the expected columns exist."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"Dataset not found at:\n  {path}\n\n"
            "Download it with:  python -m src.data_loader --download\n"
            f"or download it manually from:\n  {config.DATASET_URL}\n"
            "and place it in the 'dataset' folder."
        )
    df = pd.read_csv(path)

    missing = [c for c in (config.TEXT_COLUMN, config.TARGET_COLUMN) if c not in df.columns]
    if missing:
        raise DatasetError(
            f"The dataset is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def clean_dataset(df: pd.DataFrame, drop_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Prepare the dataset for modelling and report what was removed.

    Steps:
      1. Keep only the input text and the target label (prevents leakage from
         the 'response' and 'intent' columns).
      2. Drop rows where the text or label is missing or empty.
      3. Optionally drop repeated messages. A message that appears twice could
         otherwise end up in both the training and test sets, which would make
         the test score look better than it really is.
    """
    report = {"rows_before": int(len(df))}
    data = df[[config.TEXT_COLUMN, config.TARGET_COLUMN]].copy()

    # Missing values
    data = data.dropna(subset=[config.TEXT_COLUMN, config.TARGET_COLUMN])

    # Empty / whitespace-only values
    data[config.TEXT_COLUMN] = data[config.TEXT_COLUMN].astype(str).str.strip()
    data[config.TARGET_COLUMN] = data[config.TARGET_COLUMN].astype(str).str.strip()
    data = data[(data[config.TEXT_COLUMN] != "") & (data[config.TARGET_COLUMN] != "")]
    report["rows_removed_missing_or_empty"] = report["rows_before"] - int(len(data))

    # Duplicates: compare case-insensitively with collapsed whitespace
    if drop_duplicates:
        normalised = data[config.TEXT_COLUMN].str.lower().str.split().str.join(" ")
        before = len(data)
        data = data.loc[~normalised.duplicated(keep="first")]
        report["rows_removed_duplicates"] = before - int(len(data))
    else:
        report["rows_removed_duplicates"] = 0

    report["rows_after"] = int(len(data))
    return data.reset_index(drop=True), report


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Collect the statistics shown in the exploration report and the web app."""
    text = df[config.TEXT_COLUMN].astype(str)
    counts = df[config.TARGET_COLUMN].value_counts()
    normalised = text.str.lower().str.split().str.join(" ")

    # Do repeated messages ever carry different categories? (label conflicts)
    labels_per_text = df.assign(_norm=normalised).groupby("_norm")[config.TARGET_COLUMN].nunique()

    return {
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows_all_columns": int(df.duplicated().sum()),
        "duplicate_messages": int(normalised.duplicated().sum()),
        "messages_with_conflicting_labels": int((labels_per_text > 1).sum()),
        "num_categories": int(counts.size),
        "category_counts": {k: int(v) for k, v in counts.items()},
        "imbalance_ratio_max_to_min": round(float(counts.max() / counts.min()), 2),
        "message_length_chars": {
            "mean": round(float(text.str.len().mean()), 1),
            "min": int(text.str.len().min()),
            "max": int(text.str.len().max()),
        },
        "message_length_words_mean": round(float(text.str.split().str.len().mean()), 1),
        "messages_with_placeholders": int(text.str.contains(r"\{\{", regex=True).sum()),
    }


def plot_category_distribution(df: pd.DataFrame, save_path: Path | None = None):
    """Bar chart of how many messages belong to each category."""
    counts = df[config.TARGET_COLUMN].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("Number of messages")
    ax.set_ylabel("Category")
    ax.set_title("Category distribution")
    for i, value in enumerate(counts.values):
        ax.text(value, i, f" {value}", va="center", fontsize=9)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Load and explore the Bitext dataset.")
    parser.add_argument("--download", action="store_true", help="download the CSV if it is missing")
    args = parser.parse_args()

    if args.download and not config.DATASET_PATH.exists():
        download_dataset()

    try:
        raw = load_raw_dataset()
    except DatasetError as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    print("=" * 70)
    print("DATASET EXPLORATION - Bitext Customer Support Dataset")
    print("=" * 70)
    print(f"File: {config.DATASET_PATH.name}")
    print(f"Records: {raw.shape[0]}   Columns: {raw.shape[1]}")
    print(f"Columns: {list(raw.columns)}\n")

    print("Sample records (input text -> target label):")
    sample = raw[[config.TEXT_COLUMN, config.TARGET_COLUMN]].sample(8, random_state=config.RANDOM_STATE)
    print(sample.to_string(index=False), "\n")

    summary = summarize_dataset(raw)
    print("Missing values per column:", summary["missing_values"])
    print("Fully duplicated rows:", summary["duplicate_rows_all_columns"])
    print("Repeated customer messages:", summary["duplicate_messages"])
    print("Repeated messages with conflicting labels:", summary["messages_with_conflicting_labels"])
    print("Messages containing {{placeholders}}:", summary["messages_with_placeholders"])
    print("Average message length:", summary["message_length_words_mean"], "words\n")

    print("Category distribution:")
    for name, count in summary["category_counts"].items():
        print(f"  {name:<14}{count:>6}  ({count / summary['num_rows']:.1%})")
    print(f"Imbalance ratio (largest / smallest class): {summary['imbalance_ratio_max_to_min']}\n")

    cleaned, clean_report = clean_dataset(raw)
    print("Cleaning report:", clean_report)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = config.REPORTS_DIR / "category_distribution.png"
    plot_category_distribution(raw, chart_path)
    summary["cleaning"] = clean_report
    (config.REPORTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved chart: {chart_path}")
    print(f"Saved summary: {config.REPORTS_DIR / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()

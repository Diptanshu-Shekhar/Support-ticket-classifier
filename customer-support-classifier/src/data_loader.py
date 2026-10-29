"""
Module 1: Dataset loading and exploration.

    python -m src.data_loader              # print a report about the dataset
    python -m src.data_loader --download   # download the CSV first if it is missing
"""
import argparse
import sys
import urllib.request

import pandas as pd

from src import config


class DatasetError(Exception):
    """Raised when the dataset file is missing or has an unexpected format."""


def download_dataset():
    """Download the original Bitext CSV from GitHub into the dataset folder."""
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from:\n  {config.DATASET_URL}")
    urllib.request.urlretrieve(config.DATASET_URL, config.DATASET_PATH)
    print(f"Saved to: {config.DATASET_PATH}")


def load_raw_dataset() -> pd.DataFrame:
    """Read the CSV as published and check that the columns we need exist."""
    if not config.DATASET_PATH.exists():
        raise DatasetError(
            f"Dataset not found at:\n  {config.DATASET_PATH}\n\n"
            "Download it with:  python -m src.data_loader --download\n"
            f"or download it manually from:\n  {config.DATASET_URL}\n"
            "and place it in the 'dataset' folder."
        )
    df = pd.read_csv(config.DATASET_PATH)

    missing = [c for c in (config.TEXT_COLUMN, config.TARGET_COLUMN) if c not in df.columns]
    if missing:
        raise DatasetError(f"The dataset is missing column(s) {missing}. Found: {list(df.columns)}")
    return df


def normalise(text: pd.Series) -> pd.Series:
    """Lowercase and collapse whitespace, so repeated messages can be compared."""
    return text.astype(str).str.lower().str.split().str.join(" ")


def clean_dataset(df: pd.DataFrame, drop_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Prepare the dataset for training and report what was removed.

      1. Keep only the input text and the target label (the 'response' and
         'intent' columns would give the answer away).
      2. Drop rows where the text or label is missing or empty.
      3. Drop repeated messages, so the same sentence can never be in both the
         training and the test set. (Only the duplicate check in train.py
         turns this off, to measure what difference it makes.)
    """
    report = {"rows_before": len(df)}
    data = df[[config.TEXT_COLUMN, config.TARGET_COLUMN]].dropna()
    data = data.apply(lambda column: column.astype(str).str.strip())
    data = data[(data[config.TEXT_COLUMN] != "") & (data[config.TARGET_COLUMN] != "")]
    report["rows_removed_missing_or_empty"] = report["rows_before"] - len(data)

    before = len(data)
    if drop_duplicates:
        data = data[~normalise(data[config.TEXT_COLUMN]).duplicated()]
    report["rows_removed_duplicates"] = before - len(data)
    report["rows_after"] = len(data)
    return data.reset_index(drop=True), report


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Statistics shown by the exploration report and the web app."""
    text = df[config.TEXT_COLUMN].astype(str)
    counts = df[config.TARGET_COLUMN].value_counts()
    # Repeated messages that carry different categories (label conflicts)
    labels_per_text = df.groupby(normalise(text))[config.TARGET_COLUMN].nunique()
    return {
        "num_rows": len(df),
        "num_columns": df.shape[1],
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_messages": int(normalise(text).duplicated().sum()),
        "messages_with_conflicting_labels": int((labels_per_text > 1).sum()),
        "num_categories": counts.size,
        "category_counts": {k: int(v) for k, v in counts.items()},
        "imbalance_ratio_max_to_min": round(float(counts.max() / counts.min()), 2),
        "message_length_words_mean": round(float(text.str.split().str.len().mean()), 1),
        "messages_with_placeholders": int(text.str.contains(r"\{\{").sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Load and explore the Bitext dataset.")
    parser.add_argument("--download", action="store_true", help="download the CSV if it is missing")
    if parser.parse_args().download and not config.DATASET_PATH.exists():
        download_dataset()

    try:
        raw = load_raw_dataset()
    except DatasetError as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    print("=" * 70)
    print("DATASET EXPLORATION - Bitext Customer Support Dataset")
    print("=" * 70)
    print(f"File: {config.DATASET_FILENAME}")
    print(f"Records: {raw.shape[0]}   Columns: {raw.shape[1]}")
    print(f"Columns: {list(raw.columns)}\n")

    print("Sample records (input text -> target label):")
    sample = raw[[config.TEXT_COLUMN, config.TARGET_COLUMN]].sample(8, random_state=config.RANDOM_STATE)
    print(sample.to_string(index=False), "\n")

    summary = summarize_dataset(raw)
    print("Missing values per column:", summary["missing_values"])
    print("Repeated customer messages:", summary["duplicate_messages"])
    print("Repeated messages with conflicting labels:", summary["messages_with_conflicting_labels"])
    print("Messages containing {{placeholders}}:", summary["messages_with_placeholders"])
    print("Average message length:", summary["message_length_words_mean"], "words\n")

    print("Category distribution:")
    for name, count in summary["category_counts"].items():
        print(f"  {name:<14}{count:>6}  ({count / summary['num_rows']:.1%})")
    print(f"Imbalance ratio (largest / smallest class): {summary['imbalance_ratio_max_to_min']}\n")

    _, clean_report = clean_dataset(raw)
    print("Cleaning report:", clean_report)


if __name__ == "__main__":
    main()

"""
Central configuration: file paths and experiment settings.

All paths are built from this file's location, so the project works no matter
which folder you run the commands from.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_FILENAME = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
DATASET_PATH = DATASET_DIR / DATASET_FILENAME
DATASET_URL = (
    "https://raw.githubusercontent.com/bitext/"
    "customer-support-llm-chatbot-training-dataset/main/data/" + DATASET_FILENAME
)

MODEL_PATH = MODELS_DIR / "ticket_classifier.joblib"
TEST_SPLIT_PATH = MODELS_DIR / "test_split.csv"

# ---------------------------------------------------------------------------
# Dataset schema (verified by inspecting the downloaded CSV)
# ---------------------------------------------------------------------------
TEXT_COLUMN = "instruction"      # the customer's message  -> model input
TARGET_COLUMN = "category"       # high-level ticket category -> model output
# The other columns are "flags", "intent" and "response".
# "response" is the assistant's example answer and "intent" is a finer label
# inside each category. Neither is used as input, to avoid data leakage.

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42          # fixed seed so results are reproducible
TEST_SIZE = 0.15           # 15% held-out test set (used once, at the very end)
VALIDATION_SIZE = 0.15     # 15% validation set (used to pick the best model)
CV_FOLDS = 5               # k-fold cross-validation on the training set

# TF-IDF settings (kept simple so they are easy to explain)
TFIDF_PARAMS = {
    "ngram_range": (1, 2),     # single words and two-word phrases
    "min_df": 2,               # ignore terms that appear in only one message
    "max_features": 20000,     # cap vocabulary size
    "sublinear_tf": True,      # use 1 + log(tf) to dampen repeated words
}

# Predictions below this probability are flagged for manual review
DEFAULT_CONFIDENCE_THRESHOLD = 0.50

"""
Module 7: Streamlit web application.

    streamlit run app.py

Loads the saved model and saved evaluation reports - it never retrains and
never shows hardcoded metrics.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so "src" imports work from any folder

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import DatasetError, load_raw_dataset, summarize_dataset
from src.predict import ModelNotFoundError, classify_message, load_model
from src.preprocess import NLTKResourceError, preprocess_text

st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫", layout="wide")

EXAMPLE_MESSAGES = [
    "I cannot access my account after resetting my password.",
    "My parcel still hasn't arrived and it's been two weeks",
    "I want my money back for the broken headphones",
    "Which credit cards do you accept?",
    "How much does it cost to terminate my contract early?",
    "What is the weather today?",
]

CATEGORY_DESCRIPTIONS = {
    "ACCOUNT": "create, delete, edit or switch account; recover password; registration problems",
    "CANCEL": "check the cancellation fee",
    "CONTACT": "contact customer service or a human agent",
    "DELIVERY": "delivery options and delivery period",
    "FEEDBACK": "complaints and reviews",
    "INVOICE": "check or get an invoice",
    "ORDER": "place, cancel, change or track an order",
    "PAYMENT": "payment methods and payment issues",
    "REFUND": "refund policy, get or track a refund",
    "SHIPPING": "change or set up a shipping address",
    "SUBSCRIPTION": "newsletter subscription",
}


# ---------------------------------------------------------------------------
# Cached loaders (run once per server session, not on every click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model...")
def get_model():
    return load_model()


@st.cache_data(show_spinner="Loading dataset...")
def get_dataset():
    return load_raw_dataset()


def read_json(name: str):
    path = config.REPORTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def read_csv(name: str, **kwargs):
    path = config.REPORTS_DIR / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


def full_height(frame: pd.DataFrame) -> int:
    """Table height that shows every row without an inner scroll bar."""
    return 35 * (len(frame) + 1) + 3


def show_image(name: str, caption: str | None = None):
    path = config.REPORTS_DIR / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"{name} not found - run `python -m src.train` first.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home():
    st.title("🎫 NLP-Based Customer Support Ticket Classification System")
    st.caption("Using NLTK, TF-IDF and supervised machine learning (Scikit-learn)")

    st.subheader("Problem")
    st.write(
        "Companies receive thousands of support messages every day. Reading each one and "
        "forwarding it to the right team by hand is slow and expensive. Because the message "
        "is free-form text, this is a natural language processing problem."
    )
    st.subheader("Objective")
    st.write(
        "Automatically classify a customer's message into one of **11 support categories** "
        "(ACCOUNT, ORDER, REFUND, ...) so it can be routed to the right team, and flag "
        "uncertain messages for a human to check."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Technologies")
        st.markdown(
            "- **Python** - main language\n"
            "- **Pandas** - loading and exploring the dataset\n"
            "- **NLTK** - tokenization, stopwords, lemmatization\n"
            "- **Scikit-learn** - TF-IDF, classifiers, evaluation\n"
            "- **Matplotlib / Seaborn** - charts and confusion matrices\n"
            "- **Joblib** - saving the trained model\n"
            "- **Streamlit** - this web app"
        )
    with col2:
        st.subheader("NLP pipeline")
        st.code(
            "Customer message\n"
            "      |\n"
            "NLTK preprocessing\n"
            "  (tokenize, remove stopwords, lemmatize)\n"
            "      |\n"
            "TF-IDF feature vectors (words + 2-word phrases)\n"
            "      |\n"
            "Linear SVM classifier\n"
            "      |\n"
            "Predicted category + confidence -> route or manual review",
            language=None,
        )

    results = read_json("results.json")
    if results:
        st.subheader("Model at a glance")
        m = results["test_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", "Linear SVM")
        c2.metric("Test accuracy", f"{m['accuracy']:.2%}")
        c3.metric("Test macro-F1", f"{m['macro_f1']:.4f}")
        c4.metric("Test messages", f"{m['num_test_samples']:,}")


def page_dataset():
    st.title("📊 Dataset Explorer")
    try:
        raw = get_dataset()
    except DatasetError as error:
        st.error(str(error))
        return

    summary = summarize_dataset(raw)
    st.markdown(
        "**Source:** [Bitext Customer Support LLM Chatbot Training Dataset]"
        "(https://github.com/bitext/customer-support-llm-chatbot-training-dataset)  \n"
        f"**File:** `{config.DATASET_FILENAME}`"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{summary['num_rows']:,}")
    c2.metric("Columns", summary["num_columns"])
    c3.metric("Categories", summary["num_categories"])
    c4.metric("Avg. words / message", summary["message_length_words_mean"])

    st.subheader("Columns and how they are used")
    st.dataframe(
        pd.DataFrame([
            ("flags", "Language-variation tags (e.g. typos, colloquial)", "Not used"),
            ("instruction", "The customer's message", "Model INPUT"),
            ("category", "High-level ticket category (11 classes)", "Model TARGET"),
            ("intent", "Fine-grained intent (27 classes) inside a category", "Not used (would leak the answer)"),
            ("response", "Example reply from an agent", "Not used (would leak the answer)"),
        ], columns=["Column", "Meaning", "Use in this project"]),
        hide_index=True,
    )

    st.subheader("Sample records")
    n = st.slider("Number of samples", 5, 30, 10)
    st.dataframe(
        raw[[config.TEXT_COLUMN, config.TARGET_COLUMN, "intent"]].sample(n, random_state=config.RANDOM_STATE),
        hide_index=True,
    )

    st.subheader("Category distribution")
    counts = pd.DataFrame({
        "category": list(summary["category_counts"].keys()),
        "messages": list(summary["category_counts"].values()),
    })
    counts["share"] = (counts["messages"] / summary["num_rows"]).map("{:.1%}".format)
    counts["covers"] = counts["category"].map(CATEGORY_DESCRIPTIONS)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.bar_chart(counts.set_index("category")["messages"], horizontal=True)
    with col2:
        st.dataframe(counts, hide_index=True, height=full_height(counts))


def page_classify(threshold: float):
    st.title("🔎 Classify a Ticket")
    try:
        artifact = get_model()
    except ModelNotFoundError as error:
        st.error(str(error))
        return

    st.caption(f"Model: **NLTK preprocessing + TF-IDF + {artifact['model_name']}**")

    st.write("Try an example:")
    cols = st.columns(3)
    for i, example in enumerate(EXAMPLE_MESSAGES):
        if cols[i % 3].button(example, key=f"example_{i}"):
            st.session_state["message_input"] = example

    message = st.text_area(
        "Customer message",
        key="message_input",
        height=110,
        placeholder="e.g. I was charged twice for my order and need help",
    )

    if st.button("Classify Ticket", type="primary"):
        try:
            result = classify_message(message, artifact, threshold=threshold)
        except ValueError as error:
            st.warning(str(error))
            return
        except NLTKResourceError as error:
            st.error(str(error))
            return

        st.session_state.setdefault("history", []).append({
            "message": result.text,
            "category": result.category,
            "confidence": round(result.confidence, 3),
            "status": "manual review" if result.needs_review else "auto-routed",
        })

        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Predicted category", result.category)
            st.caption(CATEGORY_DESCRIPTIONS.get(result.category, ""))
            st.metric("Confidence", f"{result.confidence:.1%}")
            st.progress(min(result.confidence, 1.0))
            if result.needs_review:
                st.warning("⚠️ **Flagged for manual review**\n\n" + "\n".join(f"- {r}" for r in result.review_reasons))
            else:
                st.success("✅ Confident enough to route automatically.")
        with col2:
            st.write("Probability for each category")
            st.bar_chart(pd.Series(result.probabilities, name="probability"), horizontal=True)

        with st.expander("How was this prediction made?", expanded=True):
            st.markdown(
                "1. **NLTK preprocessing** cleans the message exactly as during training "
                "(lowercase, tokenize, remove stopwords, lemmatize).\n"
                "2. The saved **TF-IDF** vectorizer turns it into a vector of word / phrase weights.\n"
                "3. The **Linear SVM** scores each of the 11 categories; the highest wins.\n"
                "4. The confidence comes from a calibration step fitted on the training data."
            )
            st.write("**Step 1 output - message after NLTK preprocessing:**")
            st.code(preprocess_text(result.text) or "(empty)", language=None)
            st.write("**Step 2 - terms found in the TF-IDF vocabulary:**")
            st.code(", ".join(result.known_terms) or "(none)", language=None)
            st.caption(
                "Note: a high confidence does not guarantee a correct answer. The model always picks one "
                "of its 11 categories, even for unrelated text - that is why unknown or low-confidence "
                "messages are flagged."
            )


def page_performance():
    st.title("📈 Model Performance")
    results = read_json("results.json")
    if results is None:
        st.info("No evaluation results yet. Run `python -m src.train` first.")
        return

    m = results["test_metrics"]
    st.subheader("Linear SVM - results on the held-out test set")
    row1, row2 = st.columns(3), st.columns(3)
    row1[0].metric("Accuracy", f"{m['accuracy']:.4f}")
    row1[1].metric("Macro-F1", f"{m['macro_f1']:.4f}")
    row1[2].metric("Weighted-F1", f"{m['weighted_f1']:.4f}")
    row2[0].metric("Macro precision", f"{m['macro_precision']:.4f}")
    row2[1].metric("Macro recall", f"{m['macro_recall']:.4f}")
    row2[2].metric("Errors", f"{m['num_errors']} / {m['num_test_samples']}")

    sizes = results["split_sizes"]
    st.caption(
        f"Data after cleaning: {results['cleaning']['rows_after']:,} messages, split (stratified, seed "
        f"{config.RANDOM_STATE}) into train {sizes['train']:,} / validation {sizes['validation']:,} / "
        f"test {sizes['test']:,}. The test set was used only once, for this final evaluation."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Validation", "Per-category report", "Confusion matrix", "Error analysis"]
    )

    with tab1:
        cv = results["cross_validation"]
        st.write(
            f"**{cv['folds']}-fold cross-validation** on the training set: the training data is split into "
            f"{cv['folds']} parts; the model is trained on {cv['folds'] - 1} and tested on the remaining one, "
            f"{cv['folds']} times."
        )
        st.dataframe(
            pd.DataFrame({
                "fold": list(range(1, cv["folds"] + 1)),
                "accuracy": cv["accuracy_per_fold"],
                "macro-F1": cv["macro_f1_per_fold"],
            }),
            hide_index=True,
        )
        st.write(
            f"Mean accuracy **{cv['accuracy_mean']:.4f}** (± {cv['accuracy_std']:.4f}), "
            f"mean macro-F1 **{cv['macro_f1_mean']:.4f}** (± {cv['macro_f1_std']:.4f})."
        )
        st.write("**Training vs validation vs test** (a small gap means the model is not overfitting):")
        rows = []
        for name, key in (("Training set", "train_metrics"), ("Validation set", "validation_metrics"),
                          ("Test set", "test_metrics")):
            s = results[key]
            rows.append({"data": name, "accuracy": s["accuracy"], "macro precision": s["macro_precision"],
                         "macro recall": s["macro_recall"], "macro-F1": s["macro_f1"],
                         "weighted-F1": s["weighted_f1"]})
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True)

    with tab2:
        report = read_csv("classification_report.csv", index_col=0)
        report = report.drop(index="accuracy", errors="ignore")  # accuracy is shown at the top of the page
        report = report.rename_axis("category").reset_index()
        st.dataframe(report, hide_index=True, height=full_height(report))
        weakest = ", ".join(f"{w['category']} ({w['f1']:.4f})" for w in m["weakest_categories"])
        st.write(f"**Weakest categories:** {weakest}")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            show_image("confusion_matrix.png", "Counts")
        with col2:
            show_image("confusion_matrix_normalized.png", "Row-normalized (recall per category)")

    with tab4:
        mistakes = read_csv("misclassified_test.csv")
        st.write("Every misclassified message in the test set:")
        st.dataframe(mistakes, hide_index=True)
        if m["most_confused_pairs"]:
            st.write("Most frequent confusions:")
            st.dataframe(pd.DataFrame(m["most_confused_pairs"]), hide_index=True)

    with st.expander("What do the metrics mean?"):
        st.markdown(
            "- **Accuracy** - share of all messages classified correctly.\n"
            "- **Precision** (per category) - of the messages the model *put* in a category, how many "
            "really belong there.\n"
            "- **Recall** (per category) - of the messages that *belong* to a category, how many the "
            "model found.\n"
            "- **F1-score** - harmonic mean of precision and recall; high only if both are high.\n"
            "- **Macro average** - plain average over categories (every category counts equally, so "
            "small ones matter).\n"
            "- **Weighted average** - average weighted by category size."
        )


def page_session_analytics():
    st.title("🗂️ Session Analytics")
    history = st.session_state.get("history", [])
    if not history:
        st.info("No tickets classified yet in this session. Use **Classify a Ticket** first.")
        return
    frame = pd.DataFrame(history)
    c1, c2, c3 = st.columns(3)
    c1.metric("Tickets classified", len(frame))
    c2.metric("Auto-routed", int((frame["status"] == "auto-routed").sum()))
    c3.metric("Manual review", int((frame["status"] == "manual review").sum()))
    st.bar_chart(frame["category"].value_counts())
    st.dataframe(frame, hide_index=True, height=full_height(frame))
    if st.button("Clear history"):
        st.session_state["history"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "Home / Overview": page_home,
    "Dataset Explorer": page_dataset,
    "Classify a Ticket": page_classify,
    "Model Performance": page_performance,
    "Session Analytics": page_session_analytics,
}

with st.sidebar:
    st.header("Navigation")
    choice = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.divider()
    threshold = st.slider(
        "Manual-review threshold",
        min_value=0.0, max_value=1.0, value=config.DEFAULT_CONFIDENCE_THRESHOLD, step=0.05,
        help="Predictions with a confidence below this value are flagged for a human to check.",
    )
    st.caption("NLP Assessment Project")

if choice == "Classify a Ticket":
    page_classify(threshold)
else:
    PAGES[choice]()

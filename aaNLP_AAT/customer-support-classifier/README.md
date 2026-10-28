# NLP-Based Customer Support Ticket Classification System Using Machine Learning

An NLP application that reads a customer's support message and predicts which support team should handle it (ACCOUNT, ORDER, REFUND, ...). It uses NLTK, TF-IDF and Scikit-learn classifiers, and comes with a Streamlit web app.

---

## 1. Abstract

Support teams receive large numbers of free-text messages, and routing each one by hand is slow. This project builds a supervised text-classification system that assigns each message to one of **11 categories**. It is trained on the **Bitext Customer Support dataset** (26,872 messages).

Three classifiers are compared: Multinomial Naive Bayes, Logistic Regression and Linear SVM. Each is tested with two preprocessing options: simple lowercasing and a full NLTK pipeline. The models are compared with 5-fold cross-validation and a separate validation set, and the chosen model is tested once on a held-out test set.

The deployed model (**Linear SVM + TF-IDF**) reaches **99.89% accuracy and 0.9990 macro-F1** on 3,642 unseen test messages. We show that this very high score comes from the synthetic, template-based dataset, not from data leakage, and we discuss what that means for real tickets.

## 2. Problem statement

Given the text of a customer support message, predict its support category automatically. Messages that the model is unsure about should be flagged for a human to check.

**Why NLP?** The input is unstructured natural language with typos, abbreviations and many ways of saying the same thing ("I want my money back", "reimbursement", "get a refund"). Turning that text into features a model can learn from is exactly what NLP techniques do.

## 3. Objectives

1. Load, explore and clean the original Bitext dataset, and document its quality and limitations.
2. Build a reusable NLP preprocessing pipeline with NLTK.
3. Turn text into features with TF-IDF.
4. Train and fairly compare three classifiers, without data leakage.
5. Evaluate with accuracy, precision, recall, F1 (macro and weighted), per-class reports and confusion matrices.
6. Provide live classification, with confidence scores and manual-review flagging, in a web app.

## 4. Methodology (workflow)

```
 Bitext CSV (26,872 rows)
        |
 [Module 1] Load + validate columns + explore + clean
        |   - keep only `instruction` (input) and `category` (target)
        |   - remove 2,598 repeated messages  -> 24,274 rows
        |
 Stratified split (seed 42): train 70% | validation 15% | test 15%
        |
 [Module 2] Preprocessing  (option A: lowercase only | option B: NLTK pipeline)
        |
 [Module 3] TF-IDF (unigrams + bigrams, fitted on training data only)
        |
 [Module 4] Classifier: Naive Bayes | Logistic Regression | Linear SVM
        |   -> 6 candidates, each scored by 5-fold CV on train + validation set
        |   -> best validation macro-F1 is selected
        |
 [Module 5] One final evaluation on the held-out test set
        |   + duplicate-leakage experiment + hand-written robustness check
        |
 Saved pipeline (models/ticket_classifier.joblib)
        |
 [Module 6] predict.py  ->  [Module 7] Streamlit web app
```

## 5. Dataset

- **Name:** Bitext - Customer Service Tagged Training Dataset for LLM-based Virtual Assistants
- **Source:** https://github.com/bitext/customer-support-llm-chatbot-training-dataset
- **File:** `data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv` (19.2 MB)
- **License:** CDLA-Sharing-1.0
- **Size:** 26,872 rows x 5 columns, 0 missing values

### Fields and how they are used

| Column | Meaning | Use |
|---|---|---|
| `flags` | Language-variation tags (e.g. `Z` = typos, `Q` = colloquial) | not used |
| `instruction` | The customer's message | **model input** |
| `category` | High-level category (11 classes) | **model target** |
| `intent` | Fine-grained intent (27 classes) | not used: it maps directly onto the category, so it would give the answer away |
| `response` | Example agent reply | not used: written *after* knowing the category, so it would leak the answer |

### Target labels (verified from the file)

| Category | Messages | Share | Intents it covers |
|---|---:|---:|---|
| ACCOUNT | 5,986 | 22.3% | create / delete / edit / switch account, recover password, registration problems |
| ORDER | 3,988 | 14.8% | place / cancel / change / track order |
| REFUND | 2,992 | 11.1% | check refund policy, get refund, track refund |
| INVOICE | 1,999 | 7.4% | check / get invoice |
| CONTACT | 1,999 | 7.4% | contact customer service / human agent |
| PAYMENT | 1,998 | 7.4% | check payment methods, payment issue |
| FEEDBACK | 1,997 | 7.4% | complaint, review |
| DELIVERY | 1,994 | 7.4% | delivery options, delivery period |
| SHIPPING | 1,970 | 7.3% | change / set up shipping address |
| SUBSCRIPTION | 999 | 3.7% | newsletter subscription |
| CANCEL | 950 | 3.5% | check cancellation fee |

The repository README calls two categories `CANCELLATION_FEE` and `SHIPPING_ADDRESS`, but the CSV actually uses `CANCEL` and `SHIPPING`. The project uses the names from the file.

### Data quality findings

- **Missing values:** none. Empty rows would be removed safely if present.
- **Class imbalance:** ACCOUNT is **6.3x** larger than CANCEL. We handle this with a stratified split, `class_weight="balanced"` for Logistic Regression and SVM, and macro-F1 as the selection metric.
- **Repeated messages:** 2,598 messages appear more than once, with different example responses. None of them has conflicting labels. They are removed **before** splitting, so no sentence can be in both train and test.
- **Placeholders:** 6,670 messages contain entity placeholders such as `{{Order Number}}`.
- **Noise:** many messages have deliberate typos ("hw can I modify purchase", "nmewsletter"), added by Bitext to imitate real users.
- Average message length: **8.7 words**.

### Dataset limitations

- **Synthetic text.** Bitext generated the messages from seed templates and then added variations. Messages in the same category are very similar, which is why every model scores above 99.6%. Real tickets are longer, messier, can mention several issues at once, and would score lower.
- **SUBSCRIPTION means newsletter only.** For example, "I was charged twice for my monthly subscription" is predicted as SUBSCRIPTION, although a real team would treat it as a billing/PAYMENT issue.
- English only, general e-commerce domain, and no priority, urgency or sentiment labels.

## 6. NLP preprocessing (NLTK) - `src/preprocess.py`

| Step | Operation | Why |
|---|---|---|
| 1 | `{{Order Number}}` -> `order number` | Turns dataset placeholders into words a real user would type |
| 2 | Digits -> `number` | `#12588` and `#99` mean the same thing for routing |
| 3 | Lowercasing, `can't` -> `can not` | "Refund" = "refund"; keeps negation |
| 4 | **Tokenization** with `nltk.word_tokenize` | Splits text into words and handles punctuation and contractions |
| 5 | `n't` -> `not` | Keeps negation ("hasn't arrived") |
| 6 | Remove punctuation/symbols; keep words and codes such as `err-404` | Removes noise without dropping technical terms |
| 7 | **Stopword removal** with NLTK's English list, **keeping** `not, no, never, nor, cannot, without` | Removes filler words but not words that flip the meaning |
| 8 | **Lemmatization** with `WordNetLemmatizer` (verb, then noun) | "cancelling" -> "cancel", "orders" -> "order" |

Example: `"I never received a refund for the cancelled orders"` becomes `"never receive refund cancel order"`.

Missing, `NaN` or empty input returns an empty string instead of failing. The preprocessing function is plugged into the TF-IDF vectorizer, so it is **saved inside the model**, and training and the app always clean text in exactly the same way.

**Baseline comparison.** As the brief asks, we did not assume NLTK preprocessing helps. Every classifier was also trained with a baseline that only lowercases the text, using TF-IDF's built-in word tokenizer. See the results below.

## 7. Feature extraction: TF-IDF

**TF-IDF** (Term Frequency - Inverse Document Frequency) gives each word or phrase a weight:

- **TF:** how often the term appears in this message. We use `1 + log(tf)` (`sublinear_tf=True`), so repeated words don't dominate.
- **IDF:** how rare the term is across all messages. Common words like "I" get a low weight, and distinctive words like "refund" get a high weight.

Settings (`src/config.py`):

| Parameter | Value | Reason |
|---|---|---|
| `ngram_range` | (1, 2) | Single words plus two-word phrases such as "order number" and "not arrive" |
| `min_df` | 2 | Ignores terms that appear in only one message (mostly typos) |
| `max_features` | 20,000 | Upper limit; the real vocabulary is smaller (4,705 terms baseline, 3,785 NLTK) |
| `sublinear_tf` | True | Dampens repeated words |

**Why TF-IDF?** It is simple, fast on a normal laptop without a GPU, and works very well for short texts with distinctive keywords. It is also **interpretable**: we can list which words drive each category (see `top_terms_logistic_regression` in `reports/results.json`, e.g. REFUND -> refund, reimbursement, rebate, compensation, money).

**No leakage:** the vectorizer is part of a Scikit-learn `Pipeline`. It is fitted only on training data, including inside each cross-validation fold, and validation and test data are only transformed with it.

## 8. The three classifiers

| Classifier | How it works | Settings |
|---|---|---|
| **Multinomial Naive Bayes** | Probabilistic: learns how often each word appears per category and applies Bayes' theorem, assuming words are independent. Very fast and a strong text baseline. | `alpha=0.1` (smoothing for unseen words) |
| **Logistic Regression** | Linear model that learns one weight per word per category and turns the scores into probabilities with softmax. | `C=5`, `class_weight="balanced"`, `max_iter=2000` |
| **Linear SVM** | Finds the boundary with the **widest margin** between categories. Usually the strongest linear model for TF-IDF text. It has no probabilities of its own, so it is wrapped in `CalibratedClassifierCV` (3-fold, trained on training data only) to produce confidence scores. | `C=1`, `class_weight="balanced"` |

## 9. Training and testing methodology - `src/train.py`

1. **Clean**: 26,872 -> 24,274 rows (repeated messages removed).
2. **Stratified split**, fixed seed `42`: **train 16,991 / validation 3,641 / test 3,642**. Every split keeps the same category proportions.
3. **6 candidates** = 3 classifiers x 2 preprocessing options, each a full `Pipeline` (TF-IDF -> classifier).
4. Each candidate gets **5-fold stratified cross-validation** on the training set (mean ± std macro-F1), is then fitted on the full training set and scored on the **validation set**.
5. **Selection:** the highest **validation macro-F1**, never training accuracy.
6. **Test:** only the selected model is evaluated, once, on the test set.
7. **Extra checks:** a duplicate-leakage experiment and a hand-written robustness check (section 11).

## 10. Evaluation metrics

- **Accuracy**: share of all messages classified correctly. It can look good even if a small category is always wrong.
- **Precision** (per category): of the messages the model *put* in this category, how many truly belong there. Low precision means a team gets tickets that aren't theirs.
- **Recall** (per category): of the messages that truly belong to this category, how many the model found. Low recall means tickets get lost to other teams.
- **F1-score**: harmonic mean of precision and recall, high only if both are high.
- **Macro average**: plain mean over the 11 categories. Every category counts equally, so small classes like CANCEL matter. This is our main metric.
- **Weighted average**: mean weighted by category size.
- **Confusion matrix**: rows = true category, columns = predicted category. Off-diagonal cells are mistakes.

## 11. Results (real output of `python -m src.train`, 2026-10-28)

All numbers below are calculated from model predictions and saved in `reports/`. Re-running training reproduces them, since the seed is fixed.

### Model comparison (`reports/model_comparison.csv`)

| Model | Preprocessing | CV macro-F1 (5-fold) | Train acc. | **Val. macro-F1** | Val. acc. |
|---|---|---:|---:|---:|---:|
| **Linear SVM** | Baseline (lowercase) | 0.9989 ± 0.0007 | 0.9998 | **0.9989** | 0.9989 |
| Logistic Regression | Baseline (lowercase) | 0.9982 ± 0.0004 | 0.9998 | 0.9982 | 0.9984 |
| Multinomial NB | Baseline (lowercase) | 0.9976 ± 0.0007 | 0.9985 | 0.9979 | 0.9981 |
| Multinomial NB | NLTK pipeline | 0.9973 ± 0.0003 | 0.9985 | 0.9975 | 0.9975 |
| Linear SVM | NLTK pipeline | 0.9985 ± 0.0004 | 0.9998 | 0.9973 | 0.9975 |
| Logistic Regression | NLTK pipeline | 0.9982 ± 0.0002 | 0.9998 | 0.9969 | 0.9973 |

**Selected model:** Linear SVM with lowercase-only preprocessing.

**Did NLTK preprocessing help?** Not on this dataset. The best NLTK model reached 0.9975 against 0.9989 for the best baseline, a difference of about 5 of 3,641 validation messages, roughly the size of the CV standard deviation. A likely reason: the messages are short and templated, so words that NLTK removes or merges (stopwords, word endings) still carry signal. For example, the NLTK models turn the number in "change the address on **order 55821**" into "order number", a phrase strongly tied to ORDER, so they predict ORDER instead of SHIPPING. On longer, noisier real-world text, normalization usually matters more. This is an honest negative result, not a failure of the module.

### Final test-set performance (selected model, 3,642 unseen messages)

| Metric | Score |
|---|---:|
| Accuracy | **0.9989** |
| Macro precision | 0.9995 |
| Macro recall | 0.9985 |
| **Macro-F1** | **0.9990** |
| Weighted-F1 | 0.9989 |
| Errors | 4 / 3,642 |

Per-category report (`reports/classification_report.csv`):

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ACCOUNT | 0.9963 | 1.0000 | 0.9981 | 804 |
| CANCEL | 1.0000 | 1.0000 | 1.0000 | 141 |
| CONTACT | 1.0000 | 0.9966 | 0.9983 | 295 |
| DELIVERY | 1.0000 | 1.0000 | 1.0000 | 247 |
| FEEDBACK | 1.0000 | 1.0000 | 1.0000 | 298 |
| INVOICE | 1.0000 | 1.0000 | 1.0000 | 270 |
| ORDER | 0.9978 | 1.0000 | 0.9989 | 460 |
| **PAYMENT** | 1.0000 | **0.9933** | **0.9966** | 298 |
| **SUBSCRIPTION** | 1.0000 | **0.9933** | **0.9966** | 149 |
| SHIPPING | 1.0000 | 1.0000 | 1.0000 | 293 |
| REFUND | 1.0000 | 1.0000 | 1.0000 | 387 |

**Weakest categories:** PAYMENT and SUBSCRIPTION (2 and 1 messages missed). All 4 test errors (`reports/misclassified_test.csv`) are messages where **two words are joined without a space**, which turns them into words the model has never seen:

| Message | True | Predicted |
|---|---|---|
| assistance informing of an error with **onlinepayment** | PAYMENT | ACCOUNT |
| i want assistance canceling the **newslettersubscription** | SUBSCRIPTION | ORDER |
| **contactingassistant** | CONTACT | ACCOUNT |
| help me inform of a problem **withpayment** | PAYMENT | ACCOUNT |

Charts: `reports/confusion_matrix.png`, `confusion_matrix_normalized.png`, `per_class_f1.png`, `model_comparison.png`, `category_distribution.png`.

### Is the score inflated by duplicates? (`reports/duplicate_experiment.csv`)

The selected configuration was retrained with three different 80/20 splitting rules:

| Setting | Test msgs also in train (exact) | ...identical after NLTK preprocessing | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| A. Duplicates kept, random split | 682 | 2,506 | 0.9983 | 0.9984 |
| B. Exact duplicates removed (**our method**) | 0 | 1,949 | 0.9990 | 0.9990 |
| C. Grouped split on preprocessed text (strictest) | 0 | 0 | 0.9981 | 0.9985 |

Even when no test message (or near-duplicate of one) appears in training (C), the score barely moves. So the high accuracy is **not** caused by leakage. It reflects how consistent the synthetic dataset is.

### Hand-written robustness check (`reports/handwritten_check.csv`)

We wrote 20 realistic messages that are **not** in the dataset and labelled them before running any model. This is a qualitative check only; 20 messages are too few to compare models statistically, and it was not used for model selection.

| Model | Correct / 20 |
|---|---:|
| **Linear SVM, Baseline (deployed)** | **20** |
| NB / LR, Baseline; LR / SVM, NLTK | 19 |
| NB, NLTK | 18 |

### Validation of live predictions

The app and `src/predict.py` flag a message for **manual review** when:
- its confidence (calibrated probability) is below the threshold (default 50%, adjustable in the app), **or**
- none of its words are in the model's vocabulary (e.g. `asdkjh qwe`), **or**
- it only contains common words such as "what", "is" and "the" (e.g. `What is the weather today?`).

A high confidence **does not guarantee** a correct answer. The model must always pick one of its 11 categories, even for unrelated text, so these checks catch many but not all bad cases.

## 12. System architecture

| Module | File | Responsibility |
|---|---|---|
| Config | `src/config.py` | Paths (relative to the project, so it works from any folder), seed, split sizes, TF-IDF settings |
| 1. Data loading | `src/data_loader.py` | Download, load, validate columns, explore, clean, plot distribution |
| 2. Preprocessing | `src/preprocess.py` | NLTK pipeline, NLTK resource setup with clear error messages |
| 3+4. Training | `src/train.py` | TF-IDF pipelines, CV, validation, selection, experiments, saving |
| 5. Evaluation | `src/evaluate.py` | Metrics, reports, confusion matrices, charts; re-evaluates the saved model |
| 6. Prediction | `src/predict.py` | Input validation, prediction, confidence, manual-review flags; command-line interface |
| 7. Web app | `app.py` | Streamlit interface; loads the saved model and reports, never retrains |

The saved file `models/ticket_classifier.joblib` contains the whole fitted pipeline (preprocessing + TF-IDF + classifier), plus labels and metadata.

## 13. Project structure

```
customer-support-classifier/
├── app.py                    # Streamlit web application
├── requirements.txt
├── README.md
├── PRESENTATION_GUIDE.md     # what to show and say, module by module
├── .gitignore
├── .streamlit/config.toml    # turns off Streamlit usage statistics
├── dataset/
│   ├── README.md             # how to obtain the dataset
│   └── Bitext_..._v11.csv    # (downloaded, not committed)
├── models/                   # created by training (not committed)
│   ├── ticket_classifier.joblib
│   └── test_split.csv
├── reports/                  # created by training: metrics, charts, logs
└── src/
    ├── config.py
    ├── data_loader.py
    ├── preprocess.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

## 14. Installation (Windows PowerShell)

Requires **Python 3.10 or newer** (tested with 3.12). No GPU is needed.

```powershell
cd D:\aaNLP_AAT\customer-support-classifier

# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Download the NLTK data (one time, needs internet)
python -m src.preprocess --setup
```

If PowerShell refuses to run `Activate.ps1`, skip activation and call the environment's Python directly instead, e.g. `.\.venv\Scripts\python.exe -m src.train`.

## 15. Running the project

```powershell
# Step 3 - get the dataset (skip if the CSV is already in dataset\)
python -m src.data_loader --download

# Step 4 - inspect the dataset (prints statistics, saves reports\category_distribution.png)
python -m src.data_loader

# Steps 5-7 - train, compare, select, test and save (about 1.5-4 minutes on a laptop)
python -m src.train
#   optional: deploy the best NLTK-preprocessed model instead of the overall best
python -m src.train --preprocessing nltk

# Re-evaluate the saved model on the saved test set
python -m src.evaluate

# Classify messages from the terminal
python -m src.predict "I want my money back for the broken headphones"
python -m src.predict          # interactive mode

# Step 8 - launch the web app (opens http://localhost:8501)
streamlit run app.py
```

## 16. Example inputs and outputs (real output of the deployed model)

| Message | Predicted | Confidence | Status |
|---|---|---:|---|
| I cannot access my account after resetting my password. | ACCOUNT | 99.7% | auto-routed |
| My parcel still hasn't arrived and it's been two weeks | DELIVERY | 47.2% | manual review (low confidence) |
| I was charged twice for my monthly subscription. | SUBSCRIPTION | 87.2% | auto-routed (see limitation: SUBSCRIPTION = newsletter) |
| What is the weather today? | CONTACT | 61.1% | manual review (only common words recognised) |
| asdkjh qwe | ACCOUNT | 41.6% | manual review (no known words) |
| *(empty)* | - | - | rejected: "Please enter a customer message" |

## 17. Limitations and future work

**Limitations**
- The dataset is synthetic and templated, so the 99.9% test score is an upper bound. Expect lower accuracy on real tickets.
- Bag-of-words TF-IDF ignores word order beyond two-word phrases and doesn't know synonyms it hasn't seen ("parcel" only gets 47% confidence for DELIVERY).
- Words joined without spaces become unknown words (all 4 test errors).
- One label per message. A ticket about both a refund and an address change gets only one category.
- Category definitions follow Bitext (e.g. SUBSCRIPTION = newsletter only).
- The confidence threshold reduces risk but cannot guarantee correctness.

**Future improvements**
- Collect and label real tickets from the target company, and evaluate on them.
- Add character n-grams to TF-IDF to handle typos and joined words.
- Try word embeddings or a small fine-tuned transformer (e.g. DistilBERT) for better handling of paraphrases.
- Support multiple labels per ticket, and add the 27-intent level as a second stage.
- Log the manual-review decisions and use them to retrain the model (active learning).

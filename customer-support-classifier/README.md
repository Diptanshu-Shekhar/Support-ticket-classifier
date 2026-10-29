# NLP-Based Customer Support Ticket Classification System Using Machine Learning

An NLP application that reads a customer's support message and predicts which support team should handle it (ACCOUNT, ORDER, REFUND, ...). It uses **NLTK** for text preprocessing, **TF-IDF** for features, a **Linear SVM** classifier (Scikit-learn), and comes with a **Streamlit** web app.

---

## 1. Abstract

Support teams receive large numbers of free-text messages, and routing each one by hand is slow. This project builds a supervised text-classification system that assigns each message to one of **11 categories**. It is trained on the **Bitext Customer Support dataset** (26,872 messages).

Every message passes through an **NLTK preprocessing pipeline** (tokenization, stopword removal that keeps negations, WordNet lemmatization) and is converted to **TF-IDF** features. A **Linear SVM** classifier is trained on these features and validated with 5-fold cross-validation (mean macro-F1 0.9985) and a separate validation set. On a held-out test set of 3,642 unseen messages it reaches **99.78% accuracy and 0.9978 macro-F1**.

## 2. Problem statement

Given the text of a customer support message, predict its support category automatically. Messages that the model is unsure about should be flagged for a human to check.

**Why NLP?** The input is unstructured natural language with typos, abbreviations and many ways of saying the same thing ("I want my money back", "reimbursement", "get a refund"). Turning that text into features a model can learn from is exactly what NLP techniques do.

## 3. Objectives

1. Load, explore and clean the original Bitext dataset, and document its quality and limitations.
2. Build a reusable NLP preprocessing pipeline with NLTK.
3. Turn text into features with TF-IDF.
4. Train and validate a Linear SVM classifier, without data leakage.
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
 [Module 2] NLTK preprocessing: tokenize -> remove stopwords -> lemmatize
        |
 [Module 3] TF-IDF (unigrams + bigrams, fitted on training data only)
        |
 [Module 4] Linear SVM classifier
        |   -> 5-fold cross-validation on the training set
        |   -> trained on the training set, scored on the validation set
        |
 [Module 5] One final evaluation on the held-out test set
        |   + duplicate-leakage check + hand-written message check
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
- **Class imbalance:** ACCOUNT is **6.3x** larger than CANCEL. We handle this with a stratified split, `class_weight="balanced"` in the Linear SVM, and macro-F1 as the main metric.
- **Repeated messages:** 2,598 messages appear more than once, with different example responses. None of them has conflicting labels. They are removed **before** splitting, so no sentence can be in both train and test.
- **Placeholders:** 6,670 messages contain entity placeholders such as `{{Order Number}}`.
- **Noise:** many messages have deliberate typos ("hw can I modify purchase", "nmewsletter"), added by Bitext to imitate real users.
- Average message length: **8.7 words**.

### Dataset limitations

- **Synthetic text.** Bitext generated the messages from seed templates and then added variations. Messages in the same category are very similar, which makes the task easier than real-world ticket routing. Real tickets are longer, messier, and can mention several issues at once.
- **SUBSCRIPTION means newsletter only.** A message about a paid subscription doesn't fit any category cleanly.
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
| 8 | **Lemmatization** with `WordNetLemmatizer` (verb, then noun) | "cancelling" -> "cancel", "orders" -> "order", "arrived" -> "arrive" |

Examples (real output of `python -m src.preprocess`):

| Raw message | After NLTK preprocessing |
|---|---|
| I was charged twice for my monthly subscription!! | `charge twice monthly subscription` |
| I can't access my account after resetting my password | `not access account reset password` |
| hw can I modify purchase {{Order Number}}? | `hw modify purchase order number` |
| how can I check bill  #12588? | `check bill number` |
| I never received a refund for the cancelled orders | `never receive refund cancel order` |

Missing, `NaN` or empty input returns an empty string instead of failing. The NLTK function is plugged into the TF-IDF vectorizer, so it is **saved inside the model**, and training and the app always clean text in exactly the same way. Preprocessing also shrinks the vocabulary: lemmatization merges word forms, so the final TF-IDF vocabulary has **3,785 terms**.

## 7. Feature extraction: TF-IDF

**TF-IDF** (Term Frequency - Inverse Document Frequency) gives each word or phrase a weight:

- **TF:** how often the term appears in this message. We use `1 + log(tf)` (`sublinear_tf=True`), so repeated words don't dominate.
- **IDF:** how rare the term is across all messages. Common words get a low weight, and distinctive words like "refund" get a high weight.

Settings (`src/config.py`):

| Parameter | Value | Reason |
|---|---|---|
| `ngram_range` | (1, 2) | Single words plus two-word phrases such as "money back" and "reset password" |
| `min_df` | 2 | Ignores terms that appear in only one message (mostly typos) |
| `max_features` | 20,000 | Upper limit; the real vocabulary is 3,785 terms |
| `sublinear_tf` | True | Dampens repeated words |

**Why TF-IDF?** It is simple, fast on a normal laptop without a GPU, and works very well for short texts with distinctive keywords. It is also **interpretable**: we can list which words drive each category (see `top_terms_per_category` in `reports/results.json`: e.g. REFUND -> refund, reimbursement, rebate, compensation, money; DELIVERY -> arrive, shipment, period, ship, delivery).

**No leakage:** the vectorizer is part of a Scikit-learn `Pipeline`. It is fitted only on training data, including inside each cross-validation fold, and validation and test data are only transformed with it.

## 8. The classifier: Linear SVM

A **Support Vector Machine (SVM)** treats every message as a point in a high-dimensional space (one dimension per TF-IDF term). For each category it learns a **straight boundary (hyperplane)** that separates that category's messages from the others, placed so that the **margin** (the gap to the nearest messages on either side) is as **wide as possible**. A wide margin makes the model more robust on new messages.

- **Linear:** the boundary is a weighted sum of the TF-IDF features. This works very well for text, because there are thousands of features and the categories are usually separable by keywords.
- **Multi-class:** with 11 categories, it trains one boundary per category (one-vs-rest) and picks the category with the highest score.
- **Interpretable:** each term gets a weight per category; large positive weights show which words point to that category.

| Setting | Value | Meaning |
|---|---|---|
| `C` | 1.0 | Trade-off between a wide margin and fitting every training example |
| `class_weight` | `"balanced"` | Gives small categories (CANCEL, SUBSCRIPTION) more weight |
| Calibration | `CalibratedClassifierCV`, 3-fold, on training data only | SVMs output scores, not probabilities; calibration turns the scores into confidence percentages used for the manual-review flag |

## 9. Training and testing methodology - `src/train.py`

1. **Clean**: 26,872 -> 24,274 rows (repeated messages removed).
2. **Stratified split**, fixed seed `42`: **train 16,991 / validation 3,641 / test 3,642**. Every split keeps the same category proportions.
3. **5-fold stratified cross-validation** on the training set: the training data is split into 5 parts; the model is trained on 4 and tested on the 5th, 5 times.
4. The model is trained on the full training set and scored on the **validation set**.
5. **Test:** the model is evaluated once on the test set, which was never used before.
6. **Extra checks:** a duplicate-leakage check and a check on hand-written messages (section 11).

## 10. Evaluation metrics

- **Accuracy**: share of all messages classified correctly. It can look good even if a small category is always wrong.
- **Precision** (per category): of the messages the model *put* in this category, how many truly belong there. Low precision means a team gets tickets that aren't theirs.
- **Recall** (per category): of the messages that truly belong to this category, how many the model found. Low recall means tickets get lost to other teams.
- **F1-score**: harmonic mean of precision and recall, high only if both are high.
- **Macro average**: plain mean over the 11 categories. Every category counts equally, so small classes like CANCEL matter. This is our main metric.
- **Weighted average**: mean weighted by category size.
- **Confusion matrix**: rows = true category, columns = predicted category. Off-diagonal cells are mistakes.

## 11. Results (real output of `python -m src.train`)

All numbers below are calculated from model predictions and saved in `reports/`. Re-running training reproduces them, since the seed is fixed.

### 5-fold cross-validation (training set)

| Fold | 1 | 2 | 3 | 4 | 5 | **Mean ± std** |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.9979 | 0.9988 | 0.9982 | 0.9985 | 0.9985 | **0.9984 ± 0.0003** |
| Macro-F1 | 0.9981 | 0.9989 | 0.9980 | 0.9989 | 0.9985 | **0.9985 ± 0.0004** |

The scores are almost identical across folds, so the model is stable.

### Training vs validation vs test

| Data | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Training set (16,991) | 0.9998 | 0.9999 | 0.9998 | 0.9999 | 0.9998 |
| Validation set (3,641) | 0.9975 | 0.9975 | 0.9971 | 0.9973 | 0.9975 |
| **Test set (3,642)** | **0.9978** | **0.9986** | **0.9971** | **0.9978** | **0.9978** |

Validation and test scores are very close to the training score, so the model is **not overfitting**. Test errors: **8 out of 3,642**.

### Per-category report on the test set (`reports/classification_report.csv`)

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ACCOUNT | 0.9950 | 0.9988 | 0.9969 | 804 |
| CANCEL | 1.0000 | 1.0000 | 1.0000 | 141 |
| CONTACT | 1.0000 | 0.9966 | 0.9983 | 295 |
| DELIVERY | 1.0000 | 0.9960 | 0.9980 | 247 |
| FEEDBACK | 1.0000 | 0.9966 | 0.9983 | 298 |
| INVOICE | 1.0000 | 1.0000 | 1.0000 | 270 |
| ORDER | 0.9957 | 1.0000 | 0.9978 | 460 |
| **PAYMENT** | 0.9966 | 0.9933 | **0.9950** | 298 |
| REFUND | 0.9974 | 1.0000 | 0.9987 | 387 |
| SHIPPING | 1.0000 | 1.0000 | 1.0000 | 293 |
| **SUBSCRIPTION** | 1.0000 | 0.9866 | **0.9932** | 149 |

**Weakest categories:** SUBSCRIPTION (0.9932), PAYMENT (0.9950) and ACCOUNT (0.9969). The confusion matrix is saved as `reports/confusion_matrix.png` (counts) and `reports/confusion_matrix_normalized.png` (per-category recall).

### Error analysis (`reports/misclassified_test.csv`)

All 8 test errors are messages with a **misspelled keyword** or **two words joined without a space**. The key word is then unknown to the model, so it has to rely on the remaining words:

| Message | True | Predicted |
|---|---|---|
| where do I **leavefeedback** about your products? | FEEDBACK | ORDER |
| I want assistance to see when will my purchase **arrtive** | DELIVERY | ORDER |
| help to receive your **newslettyer** | SUBSCRIPTION | REFUND |
| assistance informing of an error with **onlinepayment** | PAYMENT | ACCOUNT |
| need help reporting errors with **registratons** | ACCOUNT | PAYMENT |
| i want assistance canceling the **newslettersubscription** | SUBSCRIPTION | ACCOUNT |
| **contactingassistant** | CONTACT | ACCOUNT |
| help me inform of a problem **withpayment** | PAYMENT | ACCOUNT |

### Duplicate-leakage check (printed by `python -m src.train`, step 6)

The model was retrained with three different 80/20 splitting rules:

| Setting | Test msgs also in train (exact) | ...identical after NLTK preprocessing | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| A. Duplicates kept, random split | 682 | 2,506 | 0.9983 | 0.9982 |
| B. Exact duplicates removed (**our method**) | 0 | 1,949 | 0.9979 | 0.9980 |
| C. Grouped split on preprocessed text (strictest) | 0 | 0 | 0.9986 | 0.9987 |

Even when no test message, or near-duplicate of one, appears in training (C), the score stays the same. So the high accuracy is **not** caused by data leakage.

### Hand-written messages (printed by `python -m src.train`, step 6)

We wrote 20 realistic messages that are **not** in the dataset (e.g. "My parcel still hasn't arrived and it's been two weeks"). The model classified **19 / 20** correctly. The miss: "Please change the address on order 55821 to 12 Baker Street" (SHIPPING, predicted ORDER), because "order number" is a strong ORDER phrase.

### Validation of live predictions

The app and `src/predict.py` flag a message for **manual review** when:
- its confidence is below the threshold (default 50%, adjustable in the app), **or**
- none of its words are in the model's vocabulary (e.g. `asdkjh qwe`, `What is the weather today?`), **or**
- it only contains very common words.

A high confidence **does not guarantee** a correct answer. The model must always pick one of its 11 categories, even for unrelated text, so these checks catch many but not all bad cases.

## 12. System architecture

| Module | File | Responsibility |
|---|---|---|
| Config | `src/config.py` | Paths (relative to the project, so it works from any folder), seed, split sizes, TF-IDF settings |
| 1. Data loading | `src/data_loader.py` | Download, load, validate columns, explore, clean, plot distribution |
| 2. Preprocessing | `src/preprocess.py` | NLTK pipeline, NLTK resource setup with clear error messages |
| 3+4. Training | `src/train.py` | NLTK + TF-IDF + Linear SVM pipeline, cross-validation, validation, checks, saving |
| 5. Evaluation | `src/evaluate.py` | Metrics, reports, confusion matrices, charts; re-evaluates the saved model |
| 6. Prediction | `src/predict.py` | Input validation, prediction, confidence, manual-review flags; command-line interface |
| 7. Web app | `app.py` | Streamlit interface; loads the saved model and results, never retrains |

The saved file `models/ticket_classifier.joblib` contains the whole fitted pipeline (NLTK preprocessing + TF-IDF + Linear SVM), plus the labels.

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
├── reports/                  # created by training: results.json, classification report, confusion matrices, errors
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

# Step 4 - inspect the dataset (prints statistics about the dataset)
python -m src.data_loader

# Show the NLTK preprocessing on example messages
python -m src.preprocess

# Steps 5-7 - train, validate, test and save
python -m src.train

# Re-evaluate the saved model on the saved test set
python -m src.evaluate

# Classify messages from the terminal (one or more messages in quotes)
python -m src.predict "I want my money back for the broken headphones" "I forgot my password"

# Step 8 - launch the web app (opens http://localhost:8501)
streamlit run app.py
```

## 16. Example inputs and outputs (real output of the model)

| Message | After NLTK | Predicted | Confidence | Status |
|---|---|---|---:|---|
| I cannot access my account after resetting my password. | `not access account reset password` | ACCOUNT | 99.7% | auto-routed |
| My parcel still hasn't arrived and it's been two weeks | `parcel still not arrive two week` | DELIVERY | 99.4% | auto-routed |
| I want my money back for the broken headphones | `want money back break headphone` | REFUND | 99.2% | auto-routed |
| Which credit cards do you accept? | `credit card accept` | PAYMENT | 98.4% | auto-routed |
| How much does it cost to terminate my contract early? | `much cost terminate contract early` | CANCEL | 98.8% | auto-routed |
| I was charged twice for my monthly subscription. | `charge twice monthly subscription` | SUBSCRIPTION | 50.0% | manual review (low confidence) |
| What is the weather today? | `weather today` | - | - | manual review (no known words) |
| asdkjh qwe | `asdkjh qwe` | - | - | manual review (no known words) |
| *(empty)* | - | - | - | rejected: "Please enter a customer message" |

## 17. Limitations and future work

**Limitations**
- The dataset is synthetic and templated, so real tickets would likely score lower.
- Bag-of-words TF-IDF ignores word order beyond two-word phrases.
- Misspelled or joined keywords become unknown words, which causes most errors.
- One label per message. A ticket about both a refund and an address change gets only one category.
- Category definitions follow Bitext (e.g. SUBSCRIPTION = newsletter only), so "charged twice for my subscription" has no clear category.
- The confidence threshold reduces risk but cannot guarantee correctness.

**Future improvements**
- Collect and label real tickets from the target company, and evaluate on them.
- Add spelling correction or character n-grams to handle typos and joined words.
- Try word embeddings or a small fine-tuned transformer (e.g. DistilBERT) for better handling of paraphrases.
- Support multiple labels per ticket, and add the 27-intent level as a second stage.
- Log the manual-review decisions and use them to retrain the model (active learning).

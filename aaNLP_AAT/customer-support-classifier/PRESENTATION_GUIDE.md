# Presentation Guide

This guide follows the five assessment requirements, in order. Commands assume PowerShell in the project folder with the virtual environment activated (`.\.venv\Scripts\Activate.ps1`).

**Before the presentation:** run `python -m src.train` once, so the model and reports exist, and start the app with `streamlit run app.py`. Keep one terminal window and the browser open.

---

## 1. Introduction to the topic (about 2 min)

**Show:** the app's **Home / Overview** page.

**Say:**
- Companies get thousands of support messages a day, and reading and routing them by hand is slow.
- **Why NLP:** the input is free text with typos and many ways to say the same thing. We need NLP to turn it into features a model can learn from.
- **Goal:** predict 1 of 11 categories (ACCOUNT, ORDER, REFUND, ...) and flag uncertain messages for a human.
- **Approach:** NLTK preprocessing -> TF-IDF features -> 3 machine-learning classifiers compared -> best one deployed in a web app.

## 2. Significant modules (about 5 min)

Open each file briefly and run its command.

| Module | File | Command to run live | Key point to say |
|---|---|---|---|
| 1. Data loading | `src/data_loader.py` | `python -m src.data_loader` | Checks columns, missing values and duplicates; keeps only `instruction` -> `category`; `response` and `intent` would leak the answer |
| 2. NLTK preprocessing | `src/preprocess.py` | `python -m src.preprocess` | Tokenize -> stopwords (but keep "not", "never") -> lemmatize. Shows before/after examples |
| 3. TF-IDF | `src/train.py` (`build_pipeline`) | - | Words + 2-word phrases, `min_df=2`; fitted on training data only, inside a Pipeline |
| 4. Training | `src/train.py` | (run in step 5) | 3 classifiers x 2 preprocessing options = 6 candidates |
| 5. Evaluation | `src/evaluate.py` | `python -m src.evaluate` | All metrics computed from predictions; saves the confusion matrix |
| 6. Prediction | `src/predict.py` | `python -m src.predict "I forgot my password"` | Same preprocessing as training, confidence score, manual-review flag |
| 7. Web app | `app.py` | `streamlit run app.py` | Loads the saved model; never retrains |

**Why NLTK?** It is the standard teaching library for classic NLP and gives us a tokenizer, a stopword list and the WordNet lemmatizer.

**Why TF-IDF?** It is fast, needs no GPU, works well for short keyword-rich messages, and is explainable. You can show the top words per category in `reports/results.json` (REFUND -> refund, reimbursement, rebate, money).

**How do the classifiers differ?**
- **Naive Bayes:** word-probability counting with Bayes' rule; fastest; assumes words are independent.
- **Logistic Regression:** learns a weight per word and gives probabilities.
- **Linear SVM:** finds the widest-margin boundary between categories; usually the strongest for text. It is calibrated to give probabilities.

## 3. Dataset details (about 2 min)

**Show:** the app's **Dataset Explorer** page.

**Say:**
- Bitext Customer Support dataset, from GitHub (CDLA-Sharing license), 26,872 messages, 5 columns.
- Input = `instruction`, target = `category` (11 classes). The CSV uses `CANCEL` and `SHIPPING`, even though the repository README names them differently.
- Imbalanced: ACCOUNT is 6.3x larger than CANCEL, so we use stratified splits, balanced class weights and macro-F1.
- 2,598 repeated messages are removed **before** splitting, to avoid leakage.
- **Limitation:** the text is synthetic and templated, which is why scores are so high. SUBSCRIPTION means newsletter only.

## 4. Validation and other functionalities (about 4 min)

**Show:** the app's **Model Performance** page, one tab at a time.

1. **Split:** 70 / 15 / 15 stratified, seed 42. The test set is touched only once, after the model is chosen.
2. **Model comparison tab:** 5-fold CV + validation scores for all 6 candidates. Selection uses validation macro-F1, never training accuracy.
   - Honest finding: NLTK preprocessing did *not* beat plain lowercasing here (a difference of about 5 messages). Short templated text keeps useful signal in stopwords and word endings.
3. **Test results:** accuracy 0.9989, macro-F1 0.9990, 4 errors out of 3,642.
4. **Per-category report + confusion matrix:** weakest are PAYMENT and SUBSCRIPTION.
5. **Error analysis tab:** all 4 errors are joined words ("onlinepayment", "newslettersubscription").
6. **Leakage tab:** even a strict grouped split gives about the same score, so the high accuracy is not leakage.
7. **Hand-written check:** 20 of our own messages; the deployed model gets 20/20.
8. **Metrics explained** (expander at the bottom): accuracy, precision, recall, F1, macro vs weighted.

**Other functionality:**
- Manual-review flag: low confidence, unknown words, or only common words.
- Adjustable threshold slider in the sidebar.
- Session Analytics page: counts and chart of the tickets classified during the demo.

## 5. Execution of the complete code (about 4 min)

Run in the terminal:

```powershell
python -m src.data_loader     # dataset report
python -m src.train           # full training (takes about 1.5-4 min; start it, then talk over it)
python -m src.evaluate        # re-evaluate the saved model
streamlit run app.py          # web app
```

Then in the app, open **Classify a Ticket** and try:

| Type this | Expect |
|---|---|
| I cannot access my account after resetting my password. | ACCOUNT, very high confidence |
| I want my money back for the broken headphones | REFUND |
| How much does it cost to terminate my contract early? | CANCEL |
| My parcel still hasn't arrived and it's been two weeks | DELIVERY, but low confidence -> manual review |
| What is the weather today? | flagged: unrelated message |
| asdkjh qwe | flagged: no known words |
| *(empty box)* | warning: input is empty |

Then show **Session Analytics**.

## Likely questions and short answers

- **Why is accuracy so high? Is it overfitting?** The test data was never used for training or selection; duplicates were removed and a strict grouped split gives the same score. The dataset is synthetic and very consistent; real tickets would score lower.
- **Why not deep learning?** TF-IDF + linear models already reach about 99.9% here, train in seconds on a laptop CPU, and are explainable. Deep learning is listed as future work for real, messier data.
- **Why macro-F1?** The classes are imbalanced. Macro-F1 gives the small categories (CANCEL, SUBSCRIPTION) equal weight.
- **Why isn't `response` used as input?** It is the agent's reply, written knowing the category, so it would leak the answer.
- **Does a high confidence mean it is correct?** No. It is a calibrated estimate, and the model always picks one of the 11 categories. That's why we also check for unknown or only-common words.
- **Why did the baseline beat NLTK?** Short templated messages keep useful information in the words NLTK removes or merges (e.g. "order 55821" -> "order number" pushes the prediction toward ORDER). On real, longer text, normalization usually helps more.

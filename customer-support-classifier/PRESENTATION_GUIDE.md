# Presentation Guide

This guide follows the five assessment requirements, in order. Commands assume PowerShell in the project folder with the virtual environment activated (`.\.venv\Scripts\Activate.ps1`).

**Before the presentation:** run `python -m src.train` once, so the model and results exist, and start the app with `streamlit run app.py`. Keep one terminal window and the browser open.

**Model used:** NLTK preprocessing -> TF-IDF -> Linear SVM

---

## 1. Introduction to the topic

**Show:** the app's **Home / Overview** page.

**Say:**
- Companies get thousands of support messages a day, and reading and routing them by hand is slow.
- **Why NLP:** the input is free text with typos and many ways to say the same thing. We need NLP to turn it into features a model can learn from.
- **Goal:** predict 1 of 11 categories (ACCOUNT, ORDER, REFUND, ...) and flag uncertain messages for a human.
- **Approach:** NLTK preprocessing -> TF-IDF features -> Linear SVM classifier -> web app.

## 2. Significant modules

Open each file briefly and run its command.

| Module | File | Command to run live | Key point to say |
|---|---|---|---|
| 1. Data loading | `src/data_loader.py` | `python -m src.data_loader` | Checks columns, missing values and duplicates; keeps only `instruction` -> `category`; `response` and `intent` would leak the answer |
| 2. NLTK preprocessing | `src/preprocess.py` | `python -m src.preprocess` | Tokenize -> stopwords (but keep "not", "never") -> lemmatize. Shows before/after examples |
| 3. TF-IDF | `src/train.py` (`build_pipeline`) | - | Words + 2-word phrases, `min_df=2`; fitted on training data only, inside a Pipeline |
| 4. Training (Linear SVM) | `src/train.py` | (run in step 5) | Widest-margin linear classifier, balanced class weights, calibrated confidence |
| 5. Evaluation | `src/evaluate.py` | `python -m src.evaluate` | All metrics computed from predictions; saves the confusion matrix |
| 6. Prediction | `src/predict.py` | `python -m src.predict "I forgot my password"` | Same NLTK preprocessing as training, confidence score, manual-review flag |
| 7. Web app | `app.py` | `streamlit run app.py` | Loads the saved model; never retrains |

**Why NLTK?** It is the standard library for classic NLP and gives us a tokenizer (`word_tokenize`), an English stopword list and the WordNet lemmatizer. Example: "I never received a refund for the cancelled orders" -> `never receive refund cancel order`.

**Why TF-IDF?** It is fast, needs no GPU, works well for short keyword-rich messages, and is explainable. Top words per category are in `reports/results.json` (REFUND -> refund, reimbursement, rebate, compensation, money).

**What is a Linear SVM?** It treats each message as a point in a space with one dimension per TF-IDF term and learns, for each category, a straight boundary that separates it from the others with the **widest possible margin**. A wide margin makes it robust on new messages. Because SVMs give scores rather than probabilities, a calibration step turns the scores into confidence percentages.

## 3. Dataset details

**Show:** the app's **Dataset Explorer** page.

**Say:**
- Bitext Customer Support dataset, from GitHub (CDLA-Sharing license), 26,872 messages, 5 columns.
- Input = `instruction`, target = `category` (11 classes). The CSV uses `CANCEL` and `SHIPPING`, even though the repository README names them differently.
- Imbalanced: ACCOUNT is 6.3x larger than CANCEL, so we use stratified splits, balanced class weights and macro-F1.
- 2,598 repeated messages are removed **before** splitting, to avoid leakage.

## 4. Validation and other functionalities

**Show:** the app's **Model Performance** page, one tab at a time.

1. **Split:** 70 / 15 / 15 stratified, seed 42 (train 16,991 / validation 3,641 / test 3,642). The test set is used only once, at the end.
2. **Validation tab:**
   - 5-fold cross-validation: macro-F1 per fold 0.9981 / 0.9989 / 0.9980 / 0.9989 / 0.9985, mean **0.9985 ± 0.0004** (stable).
   - Training 0.9998, validation 0.9975, test 0.9978 accuracy: a small gap, so no overfitting.
3. **Test results:** accuracy **0.9978**, macro-F1 **0.9978**, 8 errors out of 3,642.
4. **Per-category report + confusion matrix:** weakest are SUBSCRIPTION (0.9932) and PAYMENT (0.9950).
5. **Error analysis tab:** all 8 errors are misspelled or joined keywords ("newslettyer", "onlinepayment", "contactingassistant").
6. **Metrics explained** (expander at the bottom): accuracy, precision, recall, F1, macro vs weighted.

**Other functionality:**
- Each prediction shows the message **after NLTK preprocessing** and the terms TF-IDF recognised.
- Manual-review flag: low confidence or unknown words.
- Adjustable threshold slider in the sidebar.
- Session Analytics page: counts and chart of the tickets classified during the demo.

## 5. Execution of the complete code

Run in the terminal:

```powershell
python -m src.data_loader     # dataset report
python -m src.preprocess      # NLTK preprocessing examples
python -m src.train           # training, cross-validation, validation and testing
python -m src.evaluate        # re-evaluate the saved model
streamlit run app.py          # web app
```

Then in the app, open **Classify a Ticket** and try:

| Type this | Expect |
|---|---|
| I cannot access my account after resetting my password. | ACCOUNT, 99.7% |
| My parcel still hasn't arrived and it's been two weeks | DELIVERY, 99.4% |
| I want my money back for the broken headphones | REFUND, 99.2% |
| Which credit cards do you accept? | PAYMENT, 98.4% |
| How much does it cost to terminate my contract early? | CANCEL, 98.8% |
| What is the weather today? | flagged: no known words |
| asdkjh qwe | flagged: no known words |
| *(empty box)* | warning: input is empty |

Then show **Session Analytics**.

## Likely questions and short answers

- **Why is accuracy so high? Is it overfitting?** The test data was never used for training; duplicates were removed, and a strict grouped split gives the same score. Cross-validation, validation and test scores are all about 0.998. The dataset is synthetic and very consistent; real tickets would score lower.
- **Why Linear SVM?** It is one of the strongest classic models for TF-IDF text: it handles thousands of features well, the widest-margin boundary generalises to new messages, and it trains quickly on a laptop CPU.
- **Why not deep learning?** NLTK + TF-IDF + Linear SVM already reaches 99.78% here and is explainable. Deep learning is listed as future work for real, messier data.
- **Why macro-F1?** The classes are imbalanced. Macro-F1 gives the small categories (CANCEL, SUBSCRIPTION) equal weight.
- **Why isn't `response` used as input?** It is the agent's reply, written knowing the category, so it would leak the answer.
- **Why keep "not" and "never"?** They change the meaning ("has not arrived"), so they are removed from NLTK's stopword list.
- **How do you get a confidence from an SVM?** The SVM scores are calibrated on the training data (`CalibratedClassifierCV`) to turn them into probabilities.
- **Does a high confidence mean it is correct?** No. The model always picks one of the 11 categories. That's why we also flag messages with no known words or low confidence.

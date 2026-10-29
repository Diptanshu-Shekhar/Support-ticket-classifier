# Dataset

This project uses the original **Bitext Customer Support LLM Chatbot Training Dataset**.

- Repository: https://github.com/bitext/customer-support-llm-chatbot-training-dataset
- File: `data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv` (about 19 MB)
- License: CDLA-Sharing-1.0 (see the repository's `LICENSE.txt`)

## How to get it

**Option 1 - automatic** (from the project folder, with the virtual environment active):

```powershell
python -m src.data_loader --download
```

**Option 2 - manual**

1. Open https://github.com/bitext/customer-support-llm-chatbot-training-dataset/tree/main/data
2. Download `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`
3. Put it in this folder, so the path is:
   `customer-support-classifier/dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`

**Option 3 - PowerShell download**

```powershell
curl.exe -L -o dataset\Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv https://raw.githubusercontent.com/bitext/customer-support-llm-chatbot-training-dataset/main/data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
```

## Verified schema (26,872 rows x 5 columns)

| Column | Meaning | Used? |
|---|---|---|
| `flags` | Language-variation tags (B, L, Q, Z, ...) | No |
| `instruction` | Customer message | **Input** |
| `category` | 11 high-level categories | **Target** |
| `intent` | 27 fine-grained intents | No (would leak the target) |
| `response` | Example agent reply | No (would leak the target) |

Note: the category names in the CSV are `CANCEL` and `SHIPPING`. The repository README calls them
`CANCELLATION_FEE` and `SHIPPING_ADDRESS`. The project uses the names exactly as they appear in the file.

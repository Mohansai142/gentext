import os
import pandas as pd

# ---------------- CONFIG ----------------
INPUT_FOLDER = "../reports"       # Folder containing your text files
OUTPUT_CSV = "../data/auto_labeled.csv"

# Keywords to identify Business Analytics content
BUSINESS_KEYWORDS = [
    "analytics", "business", "dashboard", "profit", "revenue",
    "sales", "kpi", "forecast", "data", "report", "strategy"
]

def label_text(text):
    """Return 1 if text seems business-related, else 0"""
    text_lower = text.lower()
    score = sum(k in text_lower for k in BUSINESS_KEYWORDS)
    return 1 if score >= 1 else 0

# Collect all text files from reports folder
all_texts = []
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".txt"):
        with open(os.path.join(INPUT_FOLDER, filename), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_texts.append(line)

# Label data
labeled_data = [{"text": t, "label": label_text(t)} for t in all_texts]

# Save as CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df = pd.DataFrame(labeled_data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Auto-labeled dataset saved to {OUTPUT_CSV}")
print(df.head())

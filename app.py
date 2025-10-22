# app.py
from transformers import pipeline
import os

# --- Step 1: Paths ---
input_path = "reports/sample_report.txt"
output_path = "output/summary.txt"

# --- Step 2: Load text ---
if not os.path.exists(input_path):
    raise FileNotFoundError(f"{input_path} not found. Please add a report file!")

with open(input_path, "r", encoding="utf-8") as file:
    report_text = file.read()

# --- Step 3: Load pre-trained summarization model ---
print("Loading summarization model... (this may take a few seconds)")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# --- Step 4: Generate summary ---
print("Generating summary...")
summary = summarizer(report_text, max_length=130, min_length=30, do_sample=False)

# --- Step 5: Save summary ---
summary_text = summary[0]['summary_text']

os.makedirs("output", exist_ok=True)
with open(output_path, "w", encoding="utf-8") as out:
    out.write(summary_text)

print("\n✅ Summary generated successfully!")
print(f"Saved to: {output_path}")
print("\n--- Summary Preview ---\n")
print(summary_text)

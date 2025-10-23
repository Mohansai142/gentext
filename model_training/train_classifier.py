import os
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# ---------------- CONFIG ----------------
DATA_PATH = os.path.join(os.getcwd(), "data", "auto_labeled.csv")   # Your labeled CSV
MODEL_DIR = os.path.join(os.getcwd(), "business_classifier")  # Absolute path
MODEL_NAME = "distilbert-base-uncased"
NUM_EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 2e-5

# ---------------- Ensure folder exists ----------------
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"✅ Model folder ensured at: {MODEL_DIR}")

# ---------------- Load Dataset ----------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded. Total samples: {len(df)}")

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ---------------- Tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(batch): 
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Rename label column
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

# Set format for PyTorch
train_dataset.set_format("torch")
val_dataset.set_format("torch")

# ---------------- Model ----------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
print("✅ Model loaded.")

# ---------------- Training Arguments ----------------
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",        # ✅ match evaluation strategy
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_total_limit=1,
    logging_dir=os.path.join(MODEL_DIR, "logs"),
    logging_steps=50,
    load_best_model_at_end=True,  # works now
)

# ---------------- Trainer ----------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# ---------------- Train ----------------
print("🚀 Starting training...")
trainer.train()
print("✅ Training completed.")

# ---------------- Save Model & Tokenizer ----------------
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"✅ Model and tokenizer saved to {MODEL_DIR}")

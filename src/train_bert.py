import os
import json

import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

# If running in Google Colab with Google Drive:
PROJECT_PATH = "/content/drive/MyDrive/MI1"

# If running locally instead, use:
# PROJECT_PATH = "."

DATA_PATH = os.path.join(
    PROJECT_PATH,
    "data",
    "ami_dialogue_dataset.csv"
)

MODEL_SAVE_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "bert_dialogue_act"
)

MODEL_NAME = "bert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 1

RANDOM_STATE = 42


# ============================================================
# 2. CHECK DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE")
print("=" * 60)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "CUDA version:",
        torch.version.cuda
    )


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Dataset size:", len(df))

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 4. CREATE LABEL MAPPING
# ============================================================

labels = sorted(
    df["label"].unique()
)

num_labels = len(labels)

label2id = {
    label: i
    for i, label in enumerate(labels)
}

id2label = {
    i: label
    for label, i in label2id.items()
}

df["label_id"] = df["label"].map(label2id)

print("\nNumber of classes:", num_labels)

print("\nLabel mapping:")

for i in range(num_labels):
    print(
        f"{i}: {id2label[i]}"
    )


# ============================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("CREATING DATA SPLITS")
print("=" * 60)

X = df["text"]
y = df["label_id"]


# First:
# 80% training
# 20% temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# Then:
# 10% validation
# 10% test

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)


print("Training examples:", len(X_train))
print("Validation examples:", len(X_val))
print("Test examples:", len(X_test))


# ============================================================
# 6. LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded.")


# ============================================================
# 7. DATASET CLASS
# ============================================================

class AMIDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length=128
    ):

        # Tokenize once when dataset is created.
        #
        # This is much faster than tokenizing every
        # time __getitem__() is called.

        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

        self.labels = labels.tolist()


    def __len__(self):

        return len(self.labels)


    def __getitem__(self, idx):

        item = {
            "input_ids": torch.tensor(
                self.encodings["input_ids"][idx],
                dtype=torch.long
            ),

            "attention_mask": torch.tensor(
                self.encodings["attention_mask"][idx],
                dtype=torch.long
            ),

            "labels": torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )
        }

        return item


# ============================================================
# 8. CREATE DATASETS
# ============================================================

print("\n" + "=" * 60)
print("TOKENIZING DATA")
print("=" * 60)

train_dataset = AMIDataset(
    X_train,
    y_train,
    tokenizer,
    MAX_LENGTH
)

val_dataset = AMIDataset(
    X_val,
    y_val,
    tokenizer,
    MAX_LENGTH
)

test_dataset = AMIDataset(
    X_test,
    y_test,
    tokenizer,
    MAX_LENGTH
)

print("Tokenization completed.")


# ============================================================
# 9. CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTrain batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))


# ============================================================
# 10. LOAD BERT MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING BERT")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

model.to(device)

print("BERT loaded.")
print("Number of classes:", num_labels)


# ============================================================
# 11. OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 12. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0

    for step, batch in enumerate(dataloader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)


        # Reset gradients from previous batch
        optimizer.zero_grad()


        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )


        # CrossEntropyLoss calculated internally
        loss = outputs.loss


        # Backpropagation
        loss.backward()


        # Update model parameters
        optimizer.step()


        total_loss += loss.item()


        # Print progress every 100 batches
        if (step + 1) % 100 == 0:

            print(
                f"Step {step + 1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f}"
            )


    average_loss = (
        total_loss / len(dataloader)
    )

    return average_loss


# ============================================================
# 13. EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    dataloader,
    device
):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0


    # No gradients during evaluation
    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)


            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )


            loss = outputs.loss
            logits = outputs.logits


            predictions = torch.argmax(
                logits,
                dim=1
            )


            total_loss += loss.item()


            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )


    average_loss = (
        total_loss / len(dataloader)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )


    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted"
    )


    return (
        average_loss,
        accuracy,
        macro_f1,
        weighted_f1,
        all_labels,
        all_predictions
    )


# ============================================================
# 14. TRAIN BERT
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

best_macro_f1 = -1

for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print("-" * 60)


    # ----------------------------
    # Training
    # ----------------------------

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device
    )


    # ----------------------------
    # Validation
    # ----------------------------

    (
        val_loss,
        val_accuracy,
        val_macro_f1,
        val_weighted_f1,
        val_labels,
        val_predictions
    ) = evaluate(
        model,
        val_loader,
        device
    )


    print("\nEpoch results:")

    print(
        f"Train Loss:      {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Validation Macro-F1: "
        f"{val_macro_f1:.4f}"
    )

    print(
        f"Validation Weighted-F1: "
        f"{val_weighted_f1:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_macro_f1 > best_macro_f1:

        best_macro_f1 = val_macro_f1

        os.makedirs(
            MODEL_SAVE_PATH,
            exist_ok=True
        )

        model.save_pretrained(
            MODEL_SAVE_PATH
        )

        tokenizer.save_pretrained(
            MODEL_SAVE_PATH
        )

        print(
            "\nBest model saved!"
        )


# ============================================================
# 15. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)


# Load the best model from disk
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_SAVE_PATH
)

model.to(device)


(
    test_loss,
    test_accuracy,
    test_macro_f1,
    test_weighted_f1,
    test_labels,
    test_predictions
) = evaluate(
    model,
    test_loader,
    device
)


print("\nFINAL TEST RESULTS")
print("------------------")

print(
    f"Test Loss:      {test_loss:.4f}"
)

print(
    f"Test Accuracy:  {test_accuracy:.4f}"
)

print(
    f"Test Macro-F1:  {test_macro_f1:.4f}"
)

print(
    f"Test Weighted-F1: {test_weighted_f1:.4f}"
)


# ============================================================
# 16. PER-CLASS PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("PER-CLASS RESULTS")
print("=" * 60)

report = classification_report(
    test_labels,
    test_predictions,
    target_names=[
        id2label[i]
        for i in range(num_labels)
    ],
    zero_division=0
)

print(report)


# ============================================================
# 17. SAVE LABEL MAPPINGS
# ============================================================

mapping_path = os.path.join(
    MODEL_SAVE_PATH,
    "label_mapping.json"
)

with open(
    mapping_path,
    "w"
) as f:

    json.dump(
        {
            "label2id": label2id,
            "id2label": {
                str(k): v
                for k, v in id2label.items()
            }
        },
        f,
        indent=4
    )


print("\nLabel mapping saved to:")
print(mapping_path)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best Validation Macro-F1: "
    f"{best_macro_f1:.4f}"
)

print(
    f"Final Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Final Test Macro-F1: "
    f"{test_macro_f1:.4f}"
)

print(
    f"Final Test Weighted-F1: "
    f"{test_weighted_f1:.4f}"
)

print("\nModel saved at:")
print(MODEL_SAVE_PATH)
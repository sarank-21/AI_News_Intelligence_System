# ============================================================
# classifier_utils.py
# ============================================================

import os
import joblib
import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# LOAD MODEL DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "classifier"
)

# ============================================================
# DEBUG / VALIDATION
# ============================================================

print("============================================")
print("CLASSIFIER MODEL LOADING")
print("============================================")

print("BASE_DIR:")
print(BASE_DIR)

print("\nMODEL_DIR:")
print(MODEL_DIR)

print("\nMODEL DIRECTORY EXISTS:")
print(os.path.isdir(MODEL_DIR))

if not os.path.isdir(MODEL_DIR):

    raise FileNotFoundError(
        f"""
Classifier model directory was not found.

Expected location:
{MODEL_DIR}

Please make sure the following folder exists
in your GitHub repository:

models/classifier/
"""
    )

print("\nMODEL DIRECTORY CONTENTS:")

for filename in os.listdir(MODEL_DIR):
    print(" -", filename)

# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    "config.json",
    "mlb.pkl",
    "thresholds.pkl",
    "max_length.pkl"
]

missing_files = []

for filename in required_files:

    file_path = os.path.join(
        MODEL_DIR,
        filename
    )

    if not os.path.isfile(file_path):
        missing_files.append(filename)

if missing_files:

    raise FileNotFoundError(
        "Missing classifier model files: "
        + ", ".join(missing_files)
    )

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

print("Tokenizer loaded successfully.")

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading classification model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.to(DEVICE)

model.eval()

print("Model loaded successfully.")

# ============================================================
# LOAD MULTI LABEL BINARIZER
# ============================================================

mlb = joblib.load(
    os.path.join(
        MODEL_DIR,
        "mlb.pkl"
    )
)

# ============================================================
# LOAD THRESHOLDS
# ============================================================

thresholds = joblib.load(
    os.path.join(
        MODEL_DIR,
        "thresholds.pkl"
    )
)

# ============================================================
# LOAD MAX LENGTH
# ============================================================

MAX_LENGTH = joblib.load(
    os.path.join(
        MODEL_DIR,
        "max_length.pkl"
    )
)

print("\nClassifier configuration loaded.")

print("Number of classes:", len(mlb.classes_))
print("MAX_LENGTH:", MAX_LENGTH)

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_categories(text):

    text = str(text)

    # ========================================================
    # TOKENIZATION
    # ========================================================

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    # ========================================================
    # MOVE TO DEVICE
    # ========================================================

    input_ids = encoding[
        "input_ids"
    ].to(DEVICE)

    attention_mask = encoding[
        "attention_mask"
    ].to(DEVICE)

    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        probs = torch.sigmoid(
            logits
        ).cpu().numpy()[0]

    # ========================================================
    # MULTI-LABEL PREDICTION
    # ========================================================

    CUSTOM_THRESHOLD = 0.20

    predicted = (
        probs >= CUSTOM_THRESHOLD
    ).astype(int)

    # ========================================================
    # FALLBACK
    # ========================================================

    if predicted.sum() == 0:

        max_index = np.argmax(probs)

        predicted[max_index] = 1

    # ========================================================
    # RESHAPE FOR MLB
    # ========================================================

    predicted = predicted.reshape(
        1,
        -1
    )

    # ========================================================
    # DECODE LABELS
    # ========================================================

    labels = mlb.inverse_transform(
        predicted
    )[0]

    # ========================================================
    # PROBABILITY DICTIONARY
    # ========================================================

    probability_dict = {

        label: round(
            float(prob),
            4
        )

        for label, prob in zip(
            mlb.classes_,
            probs
        )
    }

    # ========================================================
    # SORT PROBABILITIES
    # ========================================================

    probability_dict = dict(
        sorted(
            probability_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "labels": list(labels),

        "probabilities": probability_dict
    }


# ============================================================
# SAMPLE TEST
# ============================================================

if __name__ == "__main__":

    sample_text = """
    Apple announced new AI investments in India.
    Government officials welcomed the partnership.
    """

    result = predict_categories(
        sample_text
    )

    print("\nPrediction Result:\n")

    print(result)
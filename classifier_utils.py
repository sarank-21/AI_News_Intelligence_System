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
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

# ============================================================
# LOAD MODEL
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)

model.eval()

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
    # MULTI-LABEL PREDICTION LOGIC
    # ========================================================

    # More flexible threshold

    CUSTOM_THRESHOLD = 0.20

    predicted = (
        probs >= CUSTOM_THRESHOLD
    ).astype(int)

    # Fallback:
    # Ensure at least one category

    if predicted.sum() == 0:

        max_index = np.argmax(probs)

        predicted[max_index] = 1

    # ========================================================
    # RESHAPE FOR MLB
    # ========================================================

    predicted = predicted.reshape(1, -1)

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

        label: round(float(prob), 4)

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
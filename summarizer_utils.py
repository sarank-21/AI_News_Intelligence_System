# ============================================================
# SUMMARIZER UTILS
# ============================================================

import torch

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# ============================================================
# DEVICE
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else -1

# ============================================================
# LOAD MODEL
# ============================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SUMMARIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "summarizer"
)

print("Using:", SUMMARIZER_PATH)

tokenizer = AutoTokenizer.from_pretrained(
    SUMMARIZER_PATH
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    SUMMARIZER_PATH
)

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=DEVICE
)

print("✅ Summarizer Loaded Successfully")

# ============================================================
# GENERATE SUMMARY
# ============================================================

def generate_summary(
    text,
    max_length=120,
    min_length=40
):

    text = str(text)

    if len(text.strip()) == 0:
        return ""

    try:

        result = summarizer(

            text,

            max_length=max_length,
            min_length=min_length,

            truncation=True,

            do_sample=False
        )

        return result[0]["summary_text"]

    except Exception as e:

        print("Summarization Error:", e)

        return ""

# ============================================================
# STREAMLIT READY FUNCTION
# ============================================================

def summarize_article(text):

    summary = generate_summary(text)

    return {
        "summary": summary
    }

# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_text = """
    Apple CEO Tim Cook visited India to discuss
    new manufacturing investments and AI partnerships.
    Government officials welcomed the expansion plans.
    Analysts believe the move could strengthen Apple's
    presence in Asian markets.
    """

    result = summarize_article(sample_text)

    print("\nSUMMARY:\n")
    print(result["summary"])
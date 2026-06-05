# ============================================================
# MISINFORMATION DETECTION UTILS
# ============================================================

import re
import ast
import joblib
import numpy as np

# ============================================================
# LOAD CONFIG
# ============================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "models",
    "misinformation",
    "misinfo_config.pkl"
)

config = joblib.load(CONFIG_PATH)

max_density = config["max_density"]

trusted_sources = config["trusted_sources"]

emotion_words = config["emotion_words"]

clickbait_words = config["clickbait_words"]

reporting_verbs = config["reporting_verbs"]

print("✅ Misinformation Config Loaded")

# ============================================================
# SAFE ENTITY PARSER
# ============================================================

def safe_parse_entities(x):

    if isinstance(x, list):
        return x

    if isinstance(x, str):

        try:
            parsed = ast.literal_eval(x)

            if isinstance(parsed, list):
                return parsed

        except:
            return []

    return []

# ============================================================
# EMOTIONAL LANGUAGE FEATURE
# ============================================================

def emotional_language_ratio(text):

    text = str(text).lower()

    words = text.split()

    if len(words) == 0:
        return 0

    emotion_count = sum(

        1 for word in words
        if word in emotion_words
    )

    return emotion_count / len(words)

# ============================================================
# SOURCE CREDIBILITY
# ============================================================

def source_credibility_score(domain):

    domain = str(domain).lower()

    if any(
        source in domain
        for source in trusted_sources
    ):
        return 0.0

    return 0.5

# ============================================================
# FACTUAL DENSITY
# ============================================================

def factual_density(text, entities):

    words = str(text).split()

    word_count = len(words)

    entity_count = len(entities)

    if word_count == 0:
        return 0

    return (entity_count / word_count) * 100

# ============================================================
# QUOTE AUTHENTICITY
# ============================================================

def quote_authenticity(text):

    text = str(text).lower()

    score = 0

    quote_count = text.count('"')

    score += quote_count

    for verb in reporting_verbs:

        score += text.count(verb)

    return min(score / 20, 1.0)

# ============================================================
# CLICKBAIT SCORE
# ============================================================

def clickbait_score(headline):

    headline = str(headline).lower()

    score = 0

    for word in clickbait_words:

        if word in headline:
            score += 1

    score += headline.count("!")

    return min(score / 5, 1.0)

# ============================================================
# FINAL RISK LABEL
# ============================================================

def risk_label(score):

    if score < 0.25:
        return "Low"

    elif score < 0.40:
        return "Medium"

    return "High"

# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_misinformation_risk(

    headline,
    body_text,
    source_domain="unknown",
    entities=None
):

    if entities is None:
        entities = []

    emotion = emotional_language_ratio(body_text)

    credibility = source_credibility_score(
        source_domain
    )

    density = factual_density(
        body_text,
        entities
    )

    factual_risk = 1 - (density / max_density)

    factual_risk = max(
        0,
        min(factual_risk, 1)
    )

    quote_score = quote_authenticity(body_text)

    quote_risk = 1 - quote_score

    clickbait = clickbait_score(headline)

    final_score = (

        0.25 * clickbait +

        0.20 * emotion +

        0.20 * credibility +

        0.20 * factual_risk +

        0.15 * quote_risk
    )

    final_score = round(final_score, 3)

    label = risk_label(final_score)

    return {

        "misinformation_score": final_score,

        "risk_label": label,

        "features": {

            "emotion_ratio":
                round(emotion, 3),

            "clickbait_score":
                round(clickbait, 3),

            "source_credibility_score":
                round(credibility, 3),

            "factual_density":
                round(density, 3),

            "quote_authenticity":
                round(quote_score, 3)
        }
    }

# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample = predict_misinformation_risk(

        headline="URGENT shocking cyber attack threatens global banking systems!",

        body_text="""
        Officials warned about a devastating cyber attack affecting banks worldwide.
        Experts reported widespread chaos and panic among customers.
        Several emergency meetings were announced by government agencies.
        """,

        source_domain="unknownnews.com",

        entities=[
            {"entity": "banks", "label": "ORG"},
            {"entity": "government agencies", "label": "ORG"}
        ]
    )

    print("\nRESULT:\n")

    print(sample)
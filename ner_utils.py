# ============================================================
# ner_utils.py
# ============================================================

import spacy

# ============================================================
# LOAD SPACY MODEL
# ============================================================

nlp = spacy.load(
    "en_core_web_sm"
)

# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(text):

    text = str(text)

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        entities.append({

            "entity": ent.text,

            "label": ent.label_
        })

    return entities

# ============================================================
# STREAMLIT FUNCTION
# ============================================================

def predict_entities(text):

    entities = extract_entities(text)

    return {

        "entities": entities
    }

# ============================================================
# SAMPLE TEST
# ============================================================

if __name__ == "__main__":

    sample_text = """
    Apple CEO Tim Cook visited India
    and met Narendra Modi in New Delhi.
    """

    result = predict_entities(
        sample_text
    )

    print(result)
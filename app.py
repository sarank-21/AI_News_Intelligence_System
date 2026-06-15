import streamlit as st
import pandas as pd

from classifier_utils import predict_categories
from ner_utils import predict_entities
from summarizer_utils import summarize_article
from misinformation_utils import predict_misinformation_risk

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="AI News Intelligence",

    page_icon="📰",

    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================================================
BACKGROUND
========================================================= */

.stApp {
    background-color: #F5F7FB;
}

/* =========================================================
TITLE
========================================================= */

.main-title {

    font-size: 42px;

    font-weight: 800;

    color: #1F2A44;

    margin-bottom: 10px;
}

.sub-text {

    font-size: 18px;

    color: #555;

    margin-bottom: 35px;
}

/* =========================================================
SECTION TITLE
========================================================= */

.section-title {

    font-size: 30px;

    font-weight: 700;

    color: #1F2A44;

    margin-top: 40px;

    margin-bottom: 20px;
}

/* =========================================================
PREDICTION CARDS
========================================================= */

.prediction-card {

    background: linear-gradient(
        90deg,
        #5B6EE1,
        #764BA2
    );

    color: white;

    padding: 18px;

    border-radius: 18px;

    text-align: center;

    font-size: 22px;

    font-weight: bold;

    box-shadow: 0 4px 12px rgba(0,0,0,0.12);

    margin-bottom: 15px;
}

/* =========================================================
SUMMARY CARD
========================================================= */

.summary-card {

    background-color: white;

    padding: 28px;

    border-radius: 20px;

    box-shadow: 0 4px 14px rgba(0,0,0,0.08);

    font-size: 18px;

    line-height: 1.9;

    color: #1e293b;

    margin-bottom: 30px;
}

/* =========================================================
DATAFRAME
========================================================= */

[data-testid="stDataFrame"] {

    background-color: white;

    border-radius: 18px;

    padding: 10px;

    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

/* =========================================================
METRIC TEXT
========================================================= */

.metric-label {

    color: #64748b;

    font-size: 18px;

    margin-bottom: 0px;
}

.metric-value {

    font-size: 52px;

    font-weight: 700;

    margin-top: 0px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📰 AI News Intelligence System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-text">

    AI-powered platform for:

    • Multi-Label News Classification  
    • Named Entity Recognition  
    • AI News Summarization  
    • Misinformation Risk Detection

    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("📥 Enter News Article")

headline = st.text_input(
    "Headline"
)

body_text = st.text_area(
    "Article Content",
    height=250
)

source_domain = st.text_input(
    "Source Domain",
    value="unknown.com"
)

# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button("🚀 Analyze News"):

    if len(body_text.strip()) == 0:

        st.warning("Please enter article content.")

    else:

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        st.markdown(
            '<div class="section-title">🏷️ Predicted Categories</div>',
            unsafe_allow_html=True
        )

        try:

            classification_result = predict_categories(
                body_text
            )

            labels = classification_result["labels"]

            probs = classification_result["probabilities"]

            # ====================================================
            # CATEGORY CARDS
            # ====================================================

            if len(labels) == 0:

                st.warning("No categories predicted.")

            else:

                cols = st.columns(len(labels))

                for idx, label in enumerate(labels):

                    with cols[idx]:

                        st.markdown(
                            f"""
                            <div class="prediction-card">
                                {label}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # ====================================================
            # PROBABILITY TABLE
            # ====================================================

            st.markdown(
                '<div class="section-title">📊 Category Probabilities</div>',
                unsafe_allow_html=True
            )

            probs_df = pd.DataFrame({

                "Category":
                    list(probs.keys()),

                "Probability":
                    list(probs.values())
            })

            probs_df = probs_df.sort_values(

                by="Probability",

                ascending=False
            )

            st.dataframe(
                probs_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(f"Classification Error: {e}")

        # ====================================================
        # NER
        # ====================================================

        st.markdown(
            '<div class="section-title">🏷️ Named Entity Recognition</div>',
            unsafe_allow_html=True
        )

        try:

            ner_result = predict_entities(
                body_text
            )

            entities = ner_result["entities"]

            if len(entities) == 0:

                st.warning("No entities found.")

            else:

                entity_df = pd.DataFrame(entities)

                entity_df = entity_df.sort_values(
                    by="label"
                )

                st.dataframe(

                    entity_df,

                    use_container_width=True,

                    hide_index=True
                )

        except Exception as e:

            st.error(f"NER Error: {e}")

            entities = []

        # ====================================================
        # SUMMARIZATION
        # ====================================================

        st.markdown(
            '<div class="section-title">📝 AI Summary</div>',
            unsafe_allow_html=True
        )

        try:

            summary_result = summarize_article(
                body_text
            )

            summary_text = summary_result["summary"]

            st.markdown(
                f"""
                <div class="summary-card">
                    {summary_text}
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(f"Summarization Error: {e}")

        # ====================================================
        # MISINFORMATION DETECTION
        # ====================================================

        st.markdown(
            '<div class="section-title">⚠️ Misinformation Risk</div>',
            unsafe_allow_html=True
        )

        try:

            misinfo_result = predict_misinformation_risk(

                headline=headline,

                body_text=body_text,

                source_domain=source_domain,

                entities=entities
            )

            risk_score = round(
                float(
                    misinfo_result["misinformation_score"]
                ),
                3
            )

            risk_label = misinfo_result["risk_label"]

            # ====================================================
            # RISK COLOR
            # ====================================================

            if risk_label == "Low":

                risk_color = "#16a34a"

            elif risk_label == "Medium":

                risk_color = "#f59e0b"

            else:

                risk_color = "#dc2626"

            # ====================================================
            # METRICS
            # ====================================================

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    f"""
                    <div class="metric-label">
                        Risk Level
                    </div>

                    <div class="metric-value"
                         style="color:{risk_color};">
                        {risk_label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    f"""
                    <div class="metric-label">
                        Risk Score
                    </div>

                    <div class="metric-value"
                         style="color:{risk_color};">
                        {risk_score}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ====================================================
            # FEATURE TABLE
            # ====================================================

            st.markdown(
                '<div class="section-title">📌 Risk Features</div>',
                unsafe_allow_html=True
            )

            feature_df = pd.DataFrame({

                "Feature":
                    list(
                        misinfo_result["features"].keys()
                    ),

                "Value":
                    list(
                        misinfo_result["features"].values()
                    )
            })

            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(
                f"Misinformation Detection Error: {e}"
            )

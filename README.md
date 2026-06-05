# 📰 AI News Intelligence System
### Multilabel News Article Classification & Entity-Aware Summarisation Engine

---

## 🧩 About the Project

The **AI News Intelligence System** is an end-to-end NLP pipeline that automatically analyzes news articles across four dimensions: multi-label topic classification, named entity recognition, entity-aware abstractive summarization, and misinformation risk scoring. It fine-tunes **RoBERTa** for multilabel classification with per-label threshold tuning, uses **spaCy** for NER, fine-tunes **T5/BART** for entity-grounded summarization, and engineers five composite misinformation signals into a single `Mis-Risk Score [0–1]`. The system is served through a live **Streamlit** application, providing media platforms, PR teams, and trading desks a real-time article intelligence dashboard.

---

## 🛠️ Development Process

### 1. 📥 Data Collection & Audit
- Ingested a multilabel news dataset with columns: `article_id`, `headline`, `body_text`, `source_domain`, `published_at`, `language`, `labels`, `summary_ref`, `entities_ref`, `mis_risk_label`, `word_count`, `scrape_noise`
- Profiled data quality issues: HTML tags in ~8% of headlines, encoding noise in ~5% of bodies, mixed-case source domains, ~10% null language codes, ~35% partial `mis_risk_label`s
- Flagged evaluation-only columns (`summary_ref`, `entities_ref`, `mis_risk_label`) as strictly no-input to prevent data leakage

### 2. 🧹 Text Preprocessing
- Stripped HTML tags, normalized Unicode, removed scrape boilerplate (nav text, ad fragments) using custom `preprocess.py`
- Programmatic language detection via `langdetect` — language column not trusted directly due to ~10% nulls and wrong labels
- Lowercased, removed excess whitespace, handled encoding errors
- Profiled article-length distribution and applied 512-token truncation strategy for BERT compatibility (headline + first 3 sentences used as input for efficiency)

### 3. 📊 EDA & Label Analysis
- Plotted label co-occurrence heatmap across 10 topic categories to identify overlapping label pairs
- Computed label density (avg labels per article) and identified rare label combinations requiring special handling
- Analyzed headline vs. body length correlation; visualized entity type distribution per topic category
- Performed noise audit: quantified HTML tag rate, encoding error rate, and scrape fragment presence

### 4. ⚙️ Feature Engineering
- Constructed `model_text` column combining headline + body for transformer input
- Applied `safe_parse_labels()` to robustly parse multi-hot label lists from string representations
- Engineered five misinformation signals: clickbait score (headline sentiment intensity), emotional language ratio (NRC lexicon), source credibility (domain whitelist lookup), factual density (entity count per 100 words), quote authenticity (direct vs. indirect quote ratio)
- Extracted entity-level features: `entity_label_counts` per article, `entity_distribution` per topic label

### 5. ⚖️ Imbalanced Label Handling
- Applied label-weighted `BCEWithLogitsLoss` in RoBERTa fine-tuning to address label imbalance
- Searched for F1-optimal threshold per label on the validation set (not global 0.5) — mandatory for imbalanced multilabel settings
- Used `OneVsRestClassifier` wrappers in baseline models to handle multi-hot targets

### 6. 🤖 Baseline Model Building
- Built three baselines: TF-IDF + Logistic Regression (One-vs-Rest), TF-IDF + Linear SVM, Word2Vec averaged embeddings + MLP
- Evaluated with Micro-F1, Macro-F1, Hamming Loss, and Jaccard Similarity Score
- Compared results in a side-by-side table to justify moving to transformer fine-tuning

### 7. 🧠 RoBERTa Multilabel Fine-Tuning
- Fine-tuned `roberta-base` with a sigmoid output head (one sigmoid per label) using `BCEWithLogitsLoss`
- Applied label-weighted loss; used headline + first 3 sentences as input for efficiency within 512-token limit
- Tuned per-label decision threshold using F1-optimal search on validation set
- Generated SHAP token attribution plots for model interpretability
- Achieved **>0.81 Micro-F1** on the test set

### 8. 🏷️ Named Entity Recognition (NER)
- Loaded `spaCy en_core_web_sm` as NER backbone; extracted entities with `text`, `label_`, `start_char`, `end_char`
- Built `extract_entities()` and `predict_entities()` functions; computed `entity_label_counts` per article
- Analyzed entity type distribution per topic label using `topic_entity_counts` (defaultdict of Counters)
- Saved spaCy model to disk (`spacy_ner_model/`) and entity labels as `entity_labels.pkl` for Streamlit inference
- Achieved **>0.79 NER F1**

### 9. 📝 Entity-Aware Abstractive Summarization
- Fine-tuned T5/BART on (article, summary) pairs with entity-grounding constraint: summaries must include at least one Person, Organisation, or Location entity from the source article
- Evaluated with ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore (F1) — ROUGE alone insufficient
- Generated sample article → summary gallery for qualitative evaluation

### 10. ⚠️ Misinformation Signal Scoring
- Engineered five rule-based + model-based features into a composite `Mis-Risk Score [0–1]`
- Calibrated score vs. human-annotated `mis_risk_label`; computed Brier Score; analyzed top-10 highest-risk articles
- Risk labels: **Low** (green), **Medium** (amber), **High** (red) rendered with color-coded metrics in Streamlit

### 11. 🚀 Streamlit Application Development
- Built multi-section UI: headline + body + source domain input → Classification → NER → Summarization → Misinformation Risk
- Custom CSS: gradient prediction cards, white summary cards, styled metric labels, themed background
- Error-isolated try/except blocks per component; probability table sortable by score; entity dataframe sortable by label

### 12. ⚡ Performance Optimization
- Shared tokenizer across classifier, NER, and summarizer components for parameter efficiency
- Modularized inference into four utility files: `classifier_utils.py`, `ner_utils.py`, `summarizer_utils.py`, `misinformation_utils.py`
- All model artifacts versioned (`model_roberta_v1.pt`, etc.); hyperparameters stored in `config.yaml`; seeds set for all random operations

---

## 🔎 Key Features

### 📌 Multilabel Topic Classification
Fine-tuned RoBERTa with per-label sigmoid heads and F1-optimal threshold tuning — an article can be simultaneously tagged as Politics, Economy, and Health.

### 🏷️ Named Entity Recognition
spaCy-powered NER extracting Person, Organisation, Location, Event, and Law entities with character-level span positions.

### 📝 Entity-Aware Summarization
T5/BART abstractive summarizer constrained to ground every summary in at least one named entity from the source article.

### ⚠️ Five-Signal Misinformation Scoring
Composite Mis-Risk Score combining clickbait detection, emotional language ratio, source credibility lookup, factual density, and quote authenticity into a calibrated [0–1] score.

### 📊 Category Probability Table
Full per-label probability breakdown displayed as a sortable dataframe, not just top predicted labels.

### 🎨 Styled Streamlit Dashboard
Gradient prediction cards, custom metric typography, white summary cards, and a themed background — production-grade UI built entirely in CSS within Streamlit.

### 🧩 Modular Inference Architecture
Four independent utility modules (`classifier_utils`, `ner_utils`, `summarizer_utils`, `misinformation_utils`) with isolated error handling — one component failure doesn't crash the app.

### ⚖️ Label Imbalance Handling
Label-weighted BCE loss + per-label F1-optimal threshold search replacing naive global 0.5 threshold — mandatory for real-world imbalanced multilabel datasets.

### 🔬 Baseline vs. Transformer Comparison
TF-IDF + LR, TF-IDF + SVM, and Word2Vec + MLP baselines benchmarked with Hamming Loss, Micro/Macro F1, and Jaccard Score before transformer fine-tuning.

### 🌐 Language-Aware Preprocessing
`langdetect`-based programmatic language filtering — the dataset's `language` column has ~10% nulls and wrong labels and cannot be trusted directly.

---

## ✨ Features (Detailed)

### 🏷️ Classification Module
- RoBERTa fine-tuned with multilabel sigmoid head and `BCEWithLogitsLoss`
- Per-label threshold tuning via F1-optimal search on validation set
- SHAP token attribution plots for interpretability
- Micro-F1 > 0.81 on held-out test set
- Gradient prediction cards rendered for each predicted label in Streamlit

### 🔍 NER Module
- `spaCy en_core_web_sm` with entity types: PERSON, ORG, GPE, DATE, EVENT, LAW, MONEY, etc.
- Entity results displayed as sortable dataframe with entity text, label, start/end character offsets
- Entity label distribution analyzed per topic category using `defaultdict(Counter)`
- Model saved to disk as `spacy_ner_model/` for fast Streamlit loading

### 📄 Summarization Module
- Entity-grounding constraint: at least one Person/Org/Location entity must appear in the 3-sentence summary
- ROUGE-1, ROUGE-2, ROUGE-L + BERTScore (F1) evaluation
- Summary rendered in white card with large line-height for readability

### 🚨 Misinformation Risk Module
- **Clickbait score**: headline sentiment intensity analysis
- **Emotional language ratio**: NRC lexicon-based emotion word density
- **Source credibility**: domain whitelist lookup from `source_domain` field
- **Factual density**: named entity count per 100 words
- **Quote authenticity**: direct vs. indirect quote ratio
- Risk Level (Low/Medium/High) color-coded with composite score displayed as metric
- Full feature value table rendered below risk metrics

---

## 🧰 Tech Stack

### 🖥️ Frontend / UI
| Library | Role |
|---|---|
| `streamlit` | App framework, layout, widgets, custom CSS |

### 🧠 Machine Learning & NLP
| Library | Role |
|---|---|
| `transformers` | RoBERTa, BERT-NER, T5/BART fine-tuning |
| `torch` | Model training, BCE loss, sigmoid heads |
| `spacy` | NER pipeline (`en_core_web_sm`) |
| `scikit-learn` | TF-IDF, Logistic Regression, SVM, MLP, metrics |

### 📊 Data Processing
| Library | Role |
|---|---|
| `pandas` | DataFrame operations, label parsing, feature tables |
| `numpy` | Numerical ops, threshold tuning |
| `ast` | Safe label list parsing from string representations |
| `langdetect` | Programmatic language detection & filtering |
| `joblib` | Model artifact serialization (`entity_labels.pkl`) |

### 📈 Evaluation
| Library | Role |
|---|---|
| `rouge_score` | ROUGE-1/2/L for summarization |
| `bert_score` | BERTScore (F1) for semantic similarity |
| `shap` | Token attribution plots for classifier interpretability |

### 📦 Utilities
| Library | Role |
|---|---|
| `collections` | `Counter`, `defaultdict` for entity distribution analysis |
| `os`, `shutil` | Model artifact path management |

### 🚀 Deployment
| Tool | Role |
|---|---|
| Streamlit | Live demo app |
| Amazon EC2 | Cloud hosting |
| Git | Version control (one branch per component) |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-news-intelligence.git
cd ai-news-intelligence
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Key libraries:
```
streamlit transformers torch spacy scikit-learn pandas numpy
joblib langdetect rouge-score bert-score shap
```

Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

### 4. Prepare the Dataset
Place the dataset CSV files in the `NoteBooks/` directory:
```
news_train_model.csv
news_test_model.csv
```

Run notebooks in order:
```
01_Data_Preprocessing.ipynb
02_EDA_Label_Analysis.ipynb
03_Baseline_ML_Models.ipynb
04_Roberta_multilabel_news.ipynb
05_NER_Entity_Extraction.ipynb
06_Entity_Aware_Summarization.ipynb
07_Misinformation_Scoring.ipynb
```

### 5. Verify Saved Model Artifacts
After running notebooks, confirm these exist:
```
models/
├── roberta/
│   └── model_roberta_v1.pt
├── ner/
│   ├── spacy_ner_model/
│   ├── entity_labels.pkl
│   └── entity_distribution.csv
├── summarizer/
│   └── model_t5_v1.pt
└── misinformation/
    └── misinfo_model_v1.pkl
```

### 6. Run the Application
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## 💼 Use Cases

1. **📡 News Aggregator Auto-Tagging** — Automatically assign multiple topic labels to every incoming article before it hits personalized feeds. Handles the reality that a government healthcare budget article is simultaneously Politics, Economy, and Health.

2. **🏢 Brand & Reputation Monitoring** — Companies monitor thousands of daily news mentions. The NER layer flags when an organization appears alongside lawsuits, product launches, or regulatory actions; multilabel classification routes the alert to the right internal team.

3. **📈 Financial News Intelligence** — Trading desks ingest news as signals. An article tagged Economy + International + Politics simultaneously signals a different market impact than one tagged Economy alone. The summarizer compresses a 900-word article to 3 sentences in milliseconds.

4. **🛡️ Misinformation Detection Pipeline** — Media platforms and fact-checkers can triage articles by Mis-Risk Score, prioritizing human review for high-risk articles flagged for clickbait headlines, emotional language spikes, or low source credibility.

5. **📰 Editorial Workflow Automation** — News desks use predicted labels + entity-aware summaries as a first draft of article metadata, reducing manual tagging time and surfacing the key named entities for sub-editors.

6. **🎓 NLP Research Benchmark** — Provides a replicable multilabel classification + NER + summarization pipeline with per-label threshold tuning, BERTScore evaluation, and SHAP interpretability — a strong research baseline.

---

## 🔮 Future Enhancements

1. **🌍 Multilingual Support** — Extend classification and NER to non-English articles using multilingual BERT (`bert-base-multilingual-cased`)
2. **🔄 Real-Time News Ingestion** — RSS feed integration for automatic article ingestion and live scoring without manual input
3. **📊 SHAP Dashboard Integration** — Embed token attribution heatmaps directly in the Streamlit UI for end-user interpretability
4. **🤝 Custom NER Fine-Tuning** — Fine-tune `dslim/bert-base-NER` on domain-specific entity types (Law, Event, Financial Instrument)
5. **📉 Automated Calibration Reports** — Scheduled Brier Score recalibration as the misinformation signal weights drift over time
6. **⚡ FastAPI Backend** — Decouple inference from Streamlit UI with a FastAPI REST endpoint for production-scale throughput
7. **🧪 Active Learning Loop** — Flag low-confidence predictions for human review and feed corrections back into periodic model retraining
8. **📦 Containerized Deployment** — Dockerize the full stack (model artifacts + Streamlit app) for reproducible EC2 / ECS deployment

---

## 🏗️ How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT UI (app.py)                       │
│  [Headline Input] [Article Body Input] [Source Domain Input]    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    🚀 Analyze News Button
                             │
          ┌──────────────────┼──────────────────────┐
          ▼                  ▼                       ▼
┌─────────────────┐ ┌────────────────┐   ┌──────────────────────┐
│ classifier_utils│ │  ner_utils.py  │   │  summarizer_utils.py │
│ predict_categories│ │predict_entities│   │ summarize_article()  │
│ (RoBERTa)       │ │ (spaCy NER)    │   │ (T5/BART)            │
└────────┬────────┘ └───────┬────────┘   └──────────┬───────────┘
         │                  │                        │
         ▼                  ▼                        ▼
   labels + probs     entity list              summary text
   (per-label         (text, label,            (3 sentences,
    threshold)         start, end)              entity-grounded)
         │                  │                        │
         └──────────────────┼────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  misinformation_utils   │
              │  predict_misinformation │
              │  _risk()                │
              │  (5-signal composite)   │
              └────────────┬────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  mis_risk_score [0-1]  │
              │  risk_label (L/M/H)    │
              │  feature breakdown     │
              └────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │    STREAMLIT OUTPUT     │
              │  🏷️ Predicted Cards      │
              │  📊 Probability Table   │
              │  🏷️ Entity DataFrame    │
              │  📝 Summary Card        │
              │  ⚠️ Risk Metrics + Table│
              └─────────────────────────┘

MODEL ARTIFACTS
├── model_roberta_v1.pt       (classifier)
├── spacy_ner_model/          (NER)
├── entity_labels.pkl         (NER labels)
├── model_t5_v1.pt            (summarizer)
└── misinfo_model_v1.pkl      (risk scorer)
```

---

## 📋 Project Overview

The AI News Intelligence System is a production-grade multilabel NLP platform built to address the limitations of single-label news classification at scale. By fine-tuning `roberta-base` with a sigmoid output head and `BCEWithLogitsLoss`, the classifier assigns overlapping topic labels (Politics, Economy, Health, Crime, etc.) to each article, achieving over 0.81 Micro-F1 through per-label F1-optimal threshold tuning rather than a naive global 0.5 cutoff. A `spaCy en_core_web_sm` NER pipeline extracts named entities with character-level span positions, achieving over 0.79 NER F1, while an entity-grounding constraint on the T5/BART summarizer ensures every generated 3-sentence summary is anchored to at least one Person, Organisation, or Location entity from the source text. The misinformation scoring module engineers five signals — clickbait headline intensity, NRC-lexicon emotional language ratio, source domain credibility, factual entity density, and quote authenticity — into a calibrated composite Mis-Risk Score [0–1]. All four components are modularized into isolated utility files and served through a custom-styled Streamlit dashboard with gradient prediction cards, sortable probability tables, and color-coded risk metrics. The system targets media platforms, PR monitoring teams, and financial trading desks requiring real-time article intelligence at scale.

---

⭐ **If you find this project useful, give it a star on GitHub and share your feedback!**

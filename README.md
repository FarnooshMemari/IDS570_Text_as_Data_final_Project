# IDS 570: Text as Data — Final Project

## "Equity" in Public Health Discourse

**Course:** IDS 570: Text as Data  
**Due:** April 22, 2026  
**Team:** Farnoosh Memari & Sebine Scaria

---

## Research Question

> *How does the meaning of "equity" in public health vary across academic, policy, NGO, and news document types?*

Each document type represents a distinct institutional setting in which "equity" is used. This project treats these settings as parallel categories and examines how the concept differs across them using Named Entity Recognition, contextual BERT embeddings, and supervised classification.

---

## Hypotheses

**H1:** The meaning of "equity" differs systematically across document types. Each category emphasizes different aspects of the concept — structural, distributional, programmatic, or rhetorical — reflecting the priorities of its institutional context.

**H2:** These differences are detectable through quantitative methods: contextual embeddings will form distinguishable clusters by document type, named entities co-occurring with "equity" will vary across categories, and classification output will show different distributions across the four groups.

---

## Project Structure

```
IDS570_TEXT_AS_DATA_FINAL_PROJECT/
│
├── 01_data/                          # All data files
│   ├── raw_data/                     # Raw extraction outputs and logs
│   ├── equity_classified.csv         # Full corpus with classification labels
│   ├── equity_embeddings.npy         # PubMedBERT embeddings (4,040 × 768)
│   ├── equity_embeddings_meta.csv    # Metadata for embedded sentences
│   ├── full_sentence_corpus.csv      # Main corpus (27,365 sentences)
│   └── validated_dataset.csv         # Source URLs and document metadata
│
├── 02_data_preparation/              # Data collection and EDA
│   ├── data_extraction.ipynb         # Corpus builder 
│   └── eda.ipynb                     # Exploratory data analysis 
│
├── 03_main_analysis/                 # Core NLP analysis notebooks
│   ├── BERT_embeddings_and_Clustering.ipynb   # PubMedBERT embeddings + K-Means + UMAP
│   ├── bert_similarity_heatmap.png            # Cosine similarity between document types
│   ├── classification.ipynb                   # Logistic regression classifier
│   ├── confusion_matrix.png                   # Classification confusion matrix
│   ├── framing_by_doctype.png                 # Equity framing proportions by category
│   └── NER_analysis.ipynb                     # spaCy NER analysis 
│
├── 04_visualizations/                # Tableau visualization files
│   ├── equity_classification_tableau.csv      # Classification data for Tableau
│   ├── equity_dashboard.twbx                  # Tableau workbook (dashboard)
│   └── equity_tableau.csv                     # BERT/UMAP data for Tableau
│
├── Literature_journals/              # Literature review sources
│   ├── 2019-wiedemannetal-konvens-bert.pdf    # Wiedemann et al. 2019 (BERT)
│   ├── Braveman-MonitoringEquityHealth-2003.pdf  # Braveman 2003 (health equity)
│   ├── Defining equity in health.pdf          # Health equity definition source
│   └── N19-1423.pdf                           # Devlin et al. 2019 (BERT paper)
│
└── README.md                         # This file
```

---

## Corpus

The corpus consists of approximately 200 English-language public health documents distributed across four parallel categories:

| Category | Documents | Equity Sentences |
|----------|-----------|-----------------|
| State/Local Health Dept. | 25 | 2,202 |
| Federal Policy | 24 | 1,036 |
| NGO / Nonprofit | 29 | 630 |
| Academic Abstracts | 55 | 140 |
| News Commentary | 23 | 52 |
| **Total** | **156** | **4,060** |

All four categories are treated as equal analytical units. No category is privileged as a reference point against which others are measured.

---

## Methods

### Step 1 — Corpus Construction (`02_data_preparation/data_extraction.ipynb`)
- Reads source URLs from `validated_dataset.csv`
- Fetches content from PDF and HTML sources using requests, BeautifulSoup, and pdfplumber
- Segments text into sentences using spaCy (`en_core_web_sm`)
- Flags equity-related sentences using regex patterns
- Outputs `full_sentence_corpus.csv` with 27,365 sentences and 21 feature columns

### Step 2 — Exploratory Data Analysis (`02_data_preparation/eda.ipynb`)
- Frequency analysis of equity occurrences by document type
- Definition rate analysis — which categories explicitly define equity
- Length confounding analysis — does document length drive equity rate differences
- Quality checks and modeling readiness diagnostics

### Step 3 — Named Entity Recognition (`03_main_analysis/NER_analysis.ipynb`)
- Loads spaCy `en_core_web_sm` model
- Extracts ORG, GPE, and PERSON entities from 4,406 equity-related contexts
- Reports both raw counts and normalized rates per 1,000 equity contexts
- Produces frequency tables and co-occurrence analyses for each document type
- **Key finding:** ORG entities dominate (62%); CMS leads policy documents; Families USA leads NGO documents; academic texts have sparse, generic entity co-occurrence

### Step 4 — BERT Embeddings and Clustering (`03_main_analysis/BERT_embeddings_and_Clustering.ipynb`)
- Model: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
- Extracts embeddings from last 4 hidden layers averaged at equity token position
- Generates 768-dimensional embedding for each of 4,040 equity sentences
- K-Means clustering with silhouette score selection (best k=2, score=0.1840)
- Cosine similarity comparison between document types
- UMAP dimensionality reduction for Tableau visualization
- **Key finding:** Two clusters separate substantive equity discourse from fragmentary equity references; academic sentences concentrate 91% in Cluster 0

### Step 5 — Supervised Classification (`03_main_analysis/classification.ipynb`)
- Weak labeling using keyword heuristics (structural vs. distributional keywords)
- 2,762 labeled examples: 337 Class A (structural) and 2,425 Class B (distributional)
- Logistic regression with `class_weight="balanced"` and 5-fold cross-validation
- TF-IDF features (unigrams + bigrams, 3,000 max features)
- Feature comparison: TF-IDF only (F1=0.957) vs BERT only (F1=0.866) vs Combined (F1=0.890)
- **Key finding:** News commentary has highest structural framing (32.7%); federal policy and NGO most distributional (~90%); TF-IDF outperforms BERT — distinction is primarily lexical

---

## Key Results

| Method | Key Finding |
|--------|-------------|
| NER | CMS dominates policy equity discourse; Families USA dominates NGO; academic texts sparse |
| BERT | 2 clusters: substantive vs. fragmentary equity discourse; silhouette = 0.1840 |
| Classification | News most structural (32.7% Class A); federal policy most distributional (89.7% Class B) |

---

## Visualizations

Interactive Tableau dashboard available at: [link to be added after publishing]

The dashboard includes:
1. **Semantic Landscape of 'Equity'** — UMAP scatter plot of PubMedBERT embeddings colored by cluster and document type
2. **Equity Framing by Document Type** — Proportional stacked bar chart showing structural vs. distributional framing across five categories

---

## Setup and Reproduction

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn transformers torch \
            umap-learn spacy tqdm pdfplumber beautifulsoup4 requests
python -m spacy download en_core_web_sm
```

### Running Order

1. `02_data_preparation/data_extraction.ipynb` — builds the corpus
2. `02_data_preparation/eda.ipynb` — exploratory analysis
3. `03_main_analysis/NER_analysis.ipynb` — named entity recognition
4. `03_main_analysis/BERT_embeddings_and_Clustering.ipynb` — embeddings and clustering
5. `03_main_analysis/classification.ipynb` — supervised classification

**Note:** Step 4 (BERT embeddings) requires approximately 6 minutes on CPU or 2 minutes on GPU. Saved embeddings in `01_data/equity_embeddings.npy` can be reloaded to skip regeneration.

---

## Literature Review Sources

- Braveman, P. & Gruskin, S. (2003). “Defining equity in health.” Journal of Epidemiology & Community Health, 57(4), 254–258. 
- Devlin et al. (2019). “BERT: Pre-training of Deep Bidirectional Transformers.” NAACL. 
- Ehrmann et al. (2023). “NER and Classification in Historical Documents.
- Marmot, M. (2005). “Social Determinants of Health Inequalities.” The Lancet, 365(9464), 1099–1104. 
- Wiedemann et al. (2019). “Does BERT Make Any Sense? Interpretable Word Sense Disambiguation with Contextualized Embeddings.” arXiv. 

---

## Division of Labor

| Task | Sebine | Farnoosh |
|------|--------|----------|
| Corpus construction | | ✓ |
| EDA | | ✓ |
| NER analysis | | ✓ |
| BERT embeddings | ✓ | |
| K-Means clustering | ✓ | |
| Logistic regression | ✓ | |
| Tableau visualizations | ✓ | ✓ |
| GitHub organization | ✓ | |
| Report writing | ✓ | ✓ |

---

## Submission

- **GitHub:** [repo link]
- **Canvas:** Report posted in Discussion section (Data Exploration)
- **Tableau:** https://public.tableau.com/app/profile/sebine.scaria/viz/UsingNaturalLanguageProcessingtoseehowthewordEquityisdescribedacrossDocumentTypes/Dashboard1

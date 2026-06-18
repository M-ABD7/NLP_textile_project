# Textile Problem Diagnosis Assistant — Classical NLP Edition

A web assistant that takes a free-text description of a fabric
processing problem and returns likely causes, a recommended action,
a chemical category, and a specific product using rule-based and
statistical NLP (NLTK, spaCy, scikit-learn, RapidFuzz). No external
AI/LLM API is required to run this project.

## Setup

```bash
pip install -r requirements.txt
python setup_nltk.py        # one-time NLTK data download
streamlit run app.py
```

## Project structure

```
app.py                       Streamlit frontend, ties everything together
nlp_engine.py                Rule-based entity extraction (Fabric/Process/Problem)
kb_engine.py                 Knowledge base lookup + TF-IDF similarity fallback
classifier.py                Optional TF-IDF + Logistic Regression backup classifier
setup_nltk.py                One-time NLTK data download
data/knowledge_base.csv      Structured KB (built from the team's TDS dataset)
data/training_sentences.csv  Labeled example sentences for the classifier
requirements.txt
```

## How the pipeline works

1. **User input** — a free-text sentence describing the problem.
2. **Preprocessing (NLTK)** — lowercase, tokenize, remove stopwords, lemmatize.
3. **Entity extraction (spaCy `PhraseMatcher`)** — Fabric, Process, and
   Problem are matched against dictionaries the team builds from
   domain knowledge (see `FABRIC_TERMS`, `PROCESS_TERMS`,
   `PROBLEM_TERMS` in `nlp_engine.py`).
4. **Fuzzy fallback (RapidFuzz)** — if no exact phrase match is found
   (typos, reordered words, unseen synonyms), a per-word fuzzy score
   catches close matches.
5. **Knowledge base lookup (`kb_engine.lookup`)** — once Problem (and
   ideally Fabric/Process) are known, the structured KB is filtered
   for the matching row(s).
6. **Statistical backups** — if the dictionary-based extractor finds
   no Problem keyword at all, the system tries, in order:
   - the trained classifier (`classifier.py`) — TF-IDF + Logistic
     Regression trained on labeled example sentences;
   - TF-IDF + cosine similarity against the whole KB
     (`kb_engine.tfidf_similarity_search`) as a last resort, so a
     reasonable answer still comes back instead of nothing.
7. **Output** — causes, recommended action, chemical category, and
   product are displayed.
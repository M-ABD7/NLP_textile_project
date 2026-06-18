
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = "knowledge_base.csv"
_STEMMER = PorterStemmer()
_STOPWORDS = set(stopwords.words("english"))


def _stem_tokenize(text: str) -> list:
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    tokens = word_tokenize(text)
    return [_STEMMER.stem(t) for t in tokens if t not in _STOPWORDS and len(t) > 1]


def load_kb(path: str = KB_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def lookup(kb: pd.DataFrame, fabric: str | None, process: str | None,
           problem: str | None) -> pd.DataFrame:
    result = kb.copy()

    if problem:
        result = result[result["Problem"].str.lower() == problem.lower()]
    if result.empty:
        return result
    if fabric:
        fabric_match = result[result["Fabric"].str.lower() == fabric.lower()]
        if not fabric_match.empty:
            result = fabric_match
    if process:
        process_match = result[result["Process"].str.lower() == process.lower()]
        if not process_match.empty:
            result = process_match

    return result


def tfidf_similarity_search(kb: pd.DataFrame, user_text: str, top_n: int = 3,
                              min_similarity: float = 0.10) -> pd.DataFrame:
    corpus = ((kb["Problem"] + " ") * 3 + kb["Causes"] + " " +
              kb["Recommended_Action"] + " " + kb["Fabric"]).tolist()
    corpus.append(user_text)

    vectorizer = TfidfVectorizer(analyzer=_stem_tokenize)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    user_vector = tfidf_matrix[-1]
    kb_vectors = tfidf_matrix[:-1]
    scores = cosine_similarity(user_vector, kb_vectors).flatten()

    ranked = kb.copy()
    ranked["similarity"] = scores
    ranked = ranked.sort_values("similarity", ascending=False)
    return ranked[ranked["similarity"] > min_similarity].head(top_n)


if __name__ == "__main__":
    kb = load_kb()

    print("--- Exact lookup: Problem=Dull Shade, Fabric=Cotton Knit ---")
    print(lookup(kb, "Cotton Knit", "Dyeing", "Dull Shade")[
        ["Fabric", "Problem", "Chemical_Category", "Product"]])

    print("\n--- TF-IDF fallback for an unmatched free-text description ---")
    query = "fabric edges are turning slightly yellow after the bleach bath"
    print(tfidf_similarity_search(kb, query)[
        ["Fabric", "Problem", "Chemical_Category", "Product", "similarity"]])

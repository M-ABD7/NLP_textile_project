
import re
import spacy
from spacy.matcher import PhraseMatcher
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from rapidfuzz import fuzz

# canonical_label -> list of surface forms / synonyms

FABRIC_TERMS = {
    "Cotton Knit": ["cotton knit", "knit cotton", "cotton knitted fabric"],
    "Cotton Woven": ["cotton woven", "woven cotton", "cotton fabric"],
    "Polyester": ["polyester", "poly fabric", "pet fabric"],
    "Polyester Blend": ["polyester blend", "poly blend", "polyester mix"],
    "Tencel / Rayon": ["tencel", "rayon", "viscose"],
    "Synthetic Blend": ["synthetic blend", "synthetic fabric", "synthetics"],
    "Spandex Blend": ["spandex blend", "lycra blend", "elastane blend"],
    "Mixed Fiber": ["mixed fiber", "mixed fibre", "blended fabric"],
    "Cotton": ["cotton"],
}

PROCESS_TERMS = {
    "Dyeing": ["dyeing", "dyed", "dye bath", "dyebath", "reactive dyeing"],
    "Finishing": ["finishing", "finish", "finished"],
    "Scouring": ["scouring", "scoured"],
    "Bleaching": ["bleaching", "bleached", "bleach bath"],
    "Pre-treatment": ["pre-treatment", "pretreatment", "preparation"],
    "Wet Processing": ["wet processing", "wet process"],
}

PROBLEM_TERMS = {
    "Dull Shade": ["dull shade", "dull shades", "dull color", "flat color",
                   "lacks brightness", "washed out", "low brightness"],
    "Crease Marks": ["crease marks", "crease mark", "crease lines",
                      "friction marks", "friction mark", "creasing"],
    "Uneven Dyeing": ["uneven dyeing", "patchy dyeing", "patchy", "blotchy",
                       "irregular dye", "dyeing irregularity", "uneven color"],
    "Low Softness": ["low softness", "harsh handle", "stiff fabric",
                      "rough handle", "not soft", "feels hard", "feels stiff"],
    "Poor Absorbency": ["poor absorbency", "low absorbency", "water repellent",
                         "does not absorb water", "poor wetting", "beads up"],
    "Shade Variation": ["shade variation", "shade mismatch", "batch to batch",
                         "batch variation", "inconsistent shade"],
    "Yellowing": ["yellowing", "yellow staining", "turning yellow",
                  "yellowish", "yellow spots"],
}

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

_nlp = spacy.blank("en")

def _build_matcher(term_dict):
    """Build a spaCy PhraseMatcher from a {canonical: [synonyms]} dictionary."""
    matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")
    for canonical, synonyms in term_dict.items():
        patterns = [_nlp.make_doc(s) for s in synonyms]
        matcher.add(canonical, patterns)
    return matcher

_FABRIC_MATCHER = _build_matcher(FABRIC_TERMS)
_PROCESS_MATCHER = _build_matcher(PROCESS_TERMS)
_PROBLEM_MATCHER = _build_matcher(PROBLEM_TERMS)


def preprocess(text: str) -> str:
    """NLTK preprocessing: lowercase, tokenize, remove stopwords, lemmatize.
    Returned mainly for transparency/display; matching itself works on
    the raw (lightly cleaned) text so multi-word phrases stay intact."""
    text = re.sub(r"[^a-zA-Z0-9\s/+-]", " ", text.lower())
    tokens = word_tokenize(text)
    cleaned = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS]
    return " ".join(cleaned)

def _match_canonical(text: str, matcher: PhraseMatcher) -> str | None:
    doc = _nlp.make_doc(text.lower())
    matches = matcher(doc)
    if not matches:
        return None
    # If multiple matches, prefer the longest phrase match (most specific)
    best = max(matches, key=lambda m: m[2] - m[1])
    match_id = best[0]
    return _nlp.vocab.strings[match_id]

def _phrase_token_score(phrase: str, tokens: list) -> float:
    words = phrase.split()
    word_scores = [max(fuzz.ratio(w, t) for t in tokens) for w in words]
    return min(word_scores)


def _fuzzy_fallback(text: str, term_dict: dict, score_cutoff: int = 80):
    tokens = preprocess(text).split()
    if not tokens:
        return None

    best_score, best_canonical = 0, None
    for canonical, synonyms in term_dict.items():
        for phrase in synonyms:
            score = _phrase_token_score(phrase, tokens)
            if score > best_score:
                best_score, best_canonical = score, canonical

    return best_canonical if best_score >= score_cutoff else None


def extract_entities(text: str) -> dict:
    fabric = _match_canonical(text, _FABRIC_MATCHER) or _fuzzy_fallback(text, FABRIC_TERMS)
    process = _match_canonical(text, _PROCESS_MATCHER) or _fuzzy_fallback(text, PROCESS_TERMS)
    problem = _match_canonical(text, _PROBLEM_MATCHER) or _fuzzy_fallback(text, PROBLEM_TERMS)

    return {
        "fabric": fabric,
        "process": process,
        "problem": problem,
        "cleaned_text": preprocess(text),
    }


if __name__ == "__main__":
    samples = [
        "Cotton knit fabric is showing dull navy shades after reactive dyeing.",
        "We have crease marks on the rayon after wet processing.",
        "The poly blend feels stiff, customer wants better softness.",
        "Mismach in shade betwen batches of cotton fabric",
    ]
    for s in samples:
        print(s, "->", extract_entities(s))

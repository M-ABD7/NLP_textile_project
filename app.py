
import streamlit as st
import nlp_engine
import kb_engine
import classifier

st.set_page_config(page_title="Textile Problem Diagnosis Assistant", layout="centered")

# Load knowledge base & train the optional classifier once 
@st.cache_resource
def load_resources():
    kb = kb_engine.load_kb()
    training_df = classifier.load_training_data()
    pipeline, acc, report = classifier.train_and_evaluate(training_df)
    return kb, pipeline, acc


kb, clf_pipeline, clf_accuracy = load_resources()

st.title("Textile Problem Diagnosis Assistant")
st.caption(
    "Describe a fabric processing problem in plain English. "
)

example = "Cotton knit fabric is showing dull navy shades after reactive dyeing."
user_text = st.text_area("Describe the issue:", value="", placeholder=example, height=100)

if st.button("Diagnose", type="primary") and user_text.strip():

    # rule-based extraction
    entities = nlp_engine.extract_entities(user_text)
    fabric, process, problem = entities["fabric"], entities["process"], entities["problem"]

    method_used = "Dictionary / fuzzy-match extractor"

    # condition when found no problem keyword,
    if not problem:
        predicted, confidence = classifier.predict_problem(clf_pipeline, user_text)
        if predicted:
            problem = predicted
            method_used = f"TF-IDF + Logistic Regression classifier (confidence={confidence:.2f})"

    with st.expander("🔍 How this answer was derived (extraction transparency)", expanded=True):
        st.write(f"**Cleaned/lemmatized text:** {entities['cleaned_text']}")
        st.write(f"**Fabric detected:** {fabric or '_not detected_'}")
        st.write(f"**Process detected:** {process or '_not detected_'}")
        st.write(f"**Problem detected:** {problem or '_not detected_'}")
        st.write(f"**Method used for Problem:** {method_used}")

    st.divider()

    if problem:
        matches = kb_engine.lookup(kb, fabric, process, problem)
    else:
        matches = kb.iloc[0:0]  # empty

    if matches.empty:
        # Last-resort: TF-IDF similarity search against the whole KB
        st.warning("No exact rule/category match -- showing closest matches by text similarity.")
        matches = kb_engine.tfidf_similarity_search(kb, user_text)

    if matches.empty:
        st.error("No relevant recommendation found in the knowledge base for this description.")
    else:
        st.subheader("📋 Recommendations")
        for _, row in matches.iterrows():
            st.markdown(f"### {row['Problem']}  —  {row['Fabric']} / {row['Process']}")
            st.write(f"**Possible causes:** {row['Causes']}")
            st.write(f"**Recommended action:** {row['Recommended_Action']}")
            st.write(f"**Chemical category:** {row['Chemical_Category']}")
            st.success(f"**Recommended product:** {row['Product']}")
            st.divider()

elif user_text.strip() == "":
    st.info("Enter a problem description above and click **Diagnose**.")

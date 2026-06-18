
import nltk

for package in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(package)

print("\nNLTK setup complete. You can now run: streamlit run app.py")

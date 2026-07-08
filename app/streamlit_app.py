import re
import pickle

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------------------------------------------------------------------
# One-time setup
# ---------------------------------------------------------------------------
nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

MAX_TEXT_CHARS = 1000
CONFIDENCE_THRESHOLD = 0.65  # below this, the app says "Uncertain" instead of guessing

port_stem = PorterStemmer()
english_stopwords = set(stopwords.words('english'))


@st.cache_resource
def load_model():
    """Load the pickled vectorizer + model once, cache across reruns."""
    with open('tfidf.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model


def stemming(content: str) -> str:
    """Same cleaning function used during training — must match exactly,
    or the model will see differently-shaped input than it was trained on."""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [
        port_stem.stem(word) for word in stemmed_content
        if word not in english_stopwords
    ]
    return ' '.join(stemmed_content)


def predict_with_confidence(vectorizer, model, title: str, text: str = '',
                             threshold: float = CONFIDENCE_THRESHOLD):
    raw_content = f'{title} {text[:MAX_TEXT_CHARS]}'
    cleaned = stemming(raw_content)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]  # [P(Real), P(Fake)]
    confidence = proba.max() * 100

    if proba.max() < threshold:
        return 'Uncertain', confidence, proba

    label = 'Fake' if proba[1] >= 0.5 else 'Real'
    return label, confidence, proba


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📰 Fake News Detector")
st.write(
    "Paste a news headline (and optionally the article body) below. "
    "The model estimates whether it resembles patterns of real or fake "
    "news based on what it learned from its training data."
)
st.markdown("**❗Works best on full news-style headlines/articles — short factual statements may not classify meaningfully.**")

with st.form("prediction_form"):
    title = st.text_input("Headline", placeholder="e.g. Local council approves new budget for 2027")
    text = st.text_area("Article text (optional, improves accuracy)", height=150,
                         placeholder="Paste the article body here if you have it...")
    submitted = st.form_submit_button("Check")

if submitted:
    if not title.strip():
        st.warning("Please enter a headline.")
    else:
        vectorizer, model = load_model()
        label, confidence, proba = predict_with_confidence(vectorizer, model, title, text)

        if label == 'Uncertain':
            st.info(f"🤔 **Uncertain** — the model isn't confident either way ({confidence:.1f}%).")
        elif label == 'Fake':
            st.error(f"🚫 **Likely Fake** — {confidence:.1f}% confident")
        else:
            st.success(f"✅ **Likely Real** — {confidence:.1f}% confident")

        col1, col2 = st.columns(2)
        col1.metric("P(Real)", f"{proba[0]*100:.1f}%")
        col2.metric("P(Fake)", f"{proba[1]*100:.1f}%")

st.divider()
with st.expander("⚠️ Known limitation — please read"):
    st.write(
        "This model was trained on news-style text (politics, general news) "
        "and uses word-frequency patterns (TF-IDF), not fact-checking or "
        "world knowledge. It can be **confidently wrong** on inputs unlike "
        "its training data — for example, obvious satire may be misclassified "
        "as real, because the model recognizes word patterns, not plausibility "
        "or truth. Confidence scores reflect *how much an input resembles "
        "training patterns*, not how true the claim actually is."
    )
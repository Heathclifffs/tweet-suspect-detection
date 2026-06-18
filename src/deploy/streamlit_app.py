import streamlit as st
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import clean_text

MODEL_DIR = "models"
MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}


@st.cache_resource
def load_models():
    vectorizer = joblib.load(f"{MODEL_DIR}/vectorizer.pkl")
    models = {}
    for name, filename in MODELS.items():
        models[name] = joblib.load(f"{MODEL_DIR}/{filename}")
    return vectorizer, models


st.set_page_config(
    page_title="Détection de Tweets Suspects",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Détection de Tweets Suspects")
st.markdown(
    "Saisissez un tweet ci-dessous pour savoir s'il est **suspect** ou **non suspect**."
)

vectorizer, models = load_models()

tweet = st.text_area(
    "✏️ Votre tweet",
    placeholder="Ex: I hate this product, it's terrible!",
    height=120,
)

if st.button("Analyser", type="primary", use_container_width=True):
    if not tweet.strip():
        st.warning("Veuillez saisir un tweet.")
    else:
        cleaned = clean_text(tweet)
        if not cleaned:
            st.warning("Le tweet est vide après nettoyage.")
        else:
            vec = vectorizer.transform([cleaned])
            for name, model in models.items():
                pred = model.predict(vec)[0]
                proba = model.predict_proba(vec)[0]
                label = "🛑 Suspect" if pred == 1 else "✅ Non suspect"
                confidence = proba[pred] * 100
                st.markdown(f"### {name}")
                col1, col2 = st.columns(2)
                col1.metric("Prédiction", label)
                col2.metric("Confiance", f"{confidence:.2f}%")
                st.progress(int(confidence))
                st.markdown("---")

else:
    st.info("👆 Entrez un tweet et cliquez sur Analyser.")

st.markdown("---")
st.markdown("### 📊 Exemples de tweets à tester")
examples = [
    "I love this! Great job everyone 👏",
    "This is the worst thing ever, I hate it!",
    "Check out this article: https://example.com",
    "You are all idiots and this is stupid",
]
for ex in examples:
    if st.button(f'📝 "{ex}"', key=ex):
        cleaned = clean_text(ex)
        if cleaned:
            vec = vectorizer.transform([cleaned])
            for name, model in models.items():
                pred = model.predict(vec)[0]
                proba = model.predict_proba(vec)[0]
                label = "🛑 Suspect" if pred == 1 else "✅ Non suspect"
                st.success(f"{name} → {label} (confiance: {proba[pred]*100:.1f}%)")

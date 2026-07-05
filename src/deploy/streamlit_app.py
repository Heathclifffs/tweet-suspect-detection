import streamlit as st
import joblib
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import clean_text

import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

MODEL_DIR = "models"
REPORT_DIR = "reports/figures"
METRICS_PATH = "models/metrics.json"
PREDICTIONS_PATH = "models/test_predictions.csv"

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


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_predictions():
    return pd.read_csv(PREDICTIONS_PATH)


def predict_tweet(tweet_text, vectorizer, models):
    cleaned = clean_text(tweet_text)
    if not cleaned:
        return None
    vec = vectorizer.transform([cleaned])
    results = []
    for name, model in models.items():
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        label = "SUSPECT" if pred == 0 else "NON SUSPECT"
        results.append({
            "modele": name,
            "prediction": label,
            "confiance": round(proba[pred] * 100, 1),
        })
    return cleaned, results


def plot_confusion_matrix(cm, model_name):
    fig = px.imshow(
        cm,
        x=["Non suspect", "Suspect"],
        y=["Non suspect", "Suspect"],
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Prediction", "y": "Reel"},
        title=f"Matrice de confusion - {model_name}",
    )
    fig.update_layout(
        width=400, height=400,
        xaxis={"side": "bottom"},
        coloraxis_showscale=False,
    )
    return fig


def plot_roc_curves(df):
    fig = go.Figure()
    colors = {"logistic_regression": "blue", "naive_bayes": "green", "random_forest": "red"}
    for model_name in df["model"].unique():
        subset = df[df["model"] == model_name]
        fpr, tpr, _ = roc_curve(subset["y_true"], subset["y_proba"])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{model_name.replace('_', ' ').title()} (AUC={roc_auc:.4f})",
            line=dict(color=colors.get(model_name, "gray"), width=2),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Aleatoire (AUC=0.5)",
        line=dict(color="black", dash="dash", width=1),
    ))
    fig.update_layout(
        title="Courbes ROC",
        xaxis_title="Taux de faux positifs",
        yaxis_title="Taux de vrais positifs",
        width=700, height=500,
        legend=dict(x=0.6, y=0.1),
    )
    return fig


def plot_metrics_comparison(metrics):
    df = pd.DataFrame([
        {"Modele": k.replace("_", " ").title(), "Metrique": m, "Valeur": v}
        for k, vals in metrics.items()
        for m, v in vals.items()
    ])
    fig = px.bar(
        df, x="Modele", y="Valeur", color="Metrique",
        barmode="group", title="Comparaison des performances",
        text_auto=".0%", range_y=[0.8, 1.0],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(width=700, height=400)
    return fig


def plot_feature_importance(vectorizer, model, model_name, top_n=20):
    if hasattr(model, "coef_"):
        importances = model.coef_.flatten()
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return None
    feature_names = vectorizer.get_feature_names_out()
    top_idx = np.argsort(np.abs(importances))[-top_n:]
    fig = px.bar(
        x=importances[top_idx],
        y=feature_names[top_idx],
        orientation="h",
        title=f"Top {top_n} features - {model_name}",
        labels={"x": "Importance", "y": ""},
        color=np.sign(importances[top_idx]),
        color_continuous_scale="RdBu",
    )
    fig.update_layout(width=600, height=500, yaxis={"categoryorder": "total ascending"})
    return fig


if "history" not in st.session_state:
    st.session_state.history = []


st.set_page_config(
    page_title="Detection de Tweets Suspects",
    layout="wide",
)

st.title("Detection de Tweets Suspects")

st.warning(
    "**Limite du modele** : le dataset d'entrainement ne contient pas de tweets de "
    "harcelement ou de discours haineux. Le modele detecte principalement la negativite "
    "generale (plaintes, souffrance personnelle). Il peut ne pas identifier correctement "
    "le racisme, les insultes ciblees ou les menaces."
)

vectorizer, models = load_models()

tab1, tab2, tab3 = st.tabs(["Prediction", "Tableau de bord", "Historique"])

with tab1:
    st.markdown(
        "Saisissez un tweet ci-dessous pour savoir s'il est **suspect** ou **non suspect**."
    )

    tweet = st.text_area(
        "Votre tweet",
        placeholder="Ex: I hate this product, it's terrible!",
        height=120,
    )

    if st.button("Analyser", type="primary", width="stretch"):
        if not tweet.strip():
            st.warning("Veuillez saisir un tweet.")
        else:
            result = predict_tweet(tweet, vectorizer, models)
            if result is None:
                st.warning("Le tweet est vide apres nettoyage.")
            else:
                cleaned, predictions = result
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "tweet": tweet,
                    "cleaned": cleaned,
                    "predictions": predictions,
                })
                for r in predictions:
                    st.markdown(f"### {r['modele']}")
                    col1, col2 = st.columns(2)
                    col1.metric("Prediction", r["prediction"])
                    col2.metric("Confiance", f"{r['confiance']}%")
                    st.progress(int(r["confiance"]))
                    st.markdown("---")

    else:
        st.info("Entrez un tweet et cliquez sur Analyser.")

    st.markdown("---")
    st.markdown("### Exemples de tweets a tester")
    examples = [
        "I love this! Great job everyone",
        "This is the worst thing ever, I hate it!",
        "Check out this article: https://example.com",
        "I feel so sick, my head hurts and I am dizzy",
    ]
    for ex in examples:
        if st.button(f'Test: "{ex}"', key=ex):
            result = predict_tweet(ex, vectorizer, models)
            if result:
                cleaned, predictions = result
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "tweet": ex,
                    "cleaned": cleaned,
                    "predictions": predictions,
                })
                for r in predictions:
                    st.success(f"{r['modele']} -> {r['prediction']} (confiance: {r['confiance']}%)")

with tab2:
    st.header("Tableau de bord dynamique")

    metrics = load_metrics()
    df_preds = load_predictions()
    model_list = list(metrics.keys())

    selected = st.selectbox(
        "Modele", model_list,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    col_metrics, _ = st.columns([1, 2])
    with col_metrics:
        m = metrics[selected]
        cols = st.columns(4)
        for i, (k, v) in enumerate(m.items()):
            cols[i].metric(k.capitalize(), f"{v:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        subset = df_preds[df_preds["model"] == selected]
        cm = confusion_matrix(subset["y_true"], subset["y_pred"])
        st.plotly_chart(plot_confusion_matrix(cm, selected.replace("_", " ").title()), width="stretch")
    with col2:
        st.plotly_chart(plot_roc_curves(df_preds), width="stretch")

    st.plotly_chart(plot_metrics_comparison(metrics), width="stretch")

    model_obj = models.get(selected.replace("_", " ").title())
    if model_obj and selected != "naive_bayes":
        fig_fi = plot_feature_importance(vectorizer, model_obj, selected.replace("_", " ").title())
        if fig_fi:
            st.plotly_chart(fig_fi, width="stretch")

    learning_curve_path = f"{REPORT_DIR}/learning_curve.png"
    if os.path.exists(learning_curve_path):
        st.subheader("Courbe d'apprentissage")
        st.image(learning_curve_path, width="stretch")

with tab3:
    st.header("Historique des analyses")

    if not st.session_state.history:
        st.info("Aucune analyse pour le moment.")
    else:
        if st.button("Vider l'historique", type="secondary"):
            st.session_state.history = []
            st.rerun()

        for i, entry in enumerate(reversed(st.session_state.history)):
            with st.expander(f"#{len(st.session_state.history) - i} - {entry['timestamp']} - \"{entry['tweet'][:60]}{'...' if len(entry['tweet']) > 60 else ''}\""):
                st.markdown(f"**Tweet original :** {entry['tweet']}")
                st.markdown(f"**Tweet nettoye :** {entry['cleaned']}")
                st.markdown("**Resultats :**")
                for r in entry["predictions"]:
                    st.markdown(f"- {r['modele']} : **{r['prediction']}** (confiance: {r['confiance']}%)")

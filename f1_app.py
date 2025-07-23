import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

model = pickle.load(open("f1_model.pkl", "rb"))

st.set_page_config(page_title="F1 Predictor 🏁")
st.title("🏎️ Formula 1 Win Probability Predictor")

grid = st.sidebar.number_input("Starting Grid Position", 1, 30, 5)
points = st.sidebar.slider("Driver Points", 0, 500, 100)

input_df = pd.DataFrame({'grid': [grid], 'points': [points]})

if st.button("Predict"):
    pred = model.predict_proba(input_df)[0][1]
    st.subheader(f"🏁 Win Probability: {round(pred * 100, 2)}%")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    st.write("### 🔍 Feature Impact (SHAP)")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches="tight")

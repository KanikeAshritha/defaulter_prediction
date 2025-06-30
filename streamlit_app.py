import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")

def preprocess(df):
    df = df.copy()
    df.drop(columns=['ID', 'ZIP Code'], inplace=True, errors='ignore')
    df = df[df['Experience'] >= 0]

    for col in ['Income', 'CCAvg', 'Mortgage']:
        df[col] = df[col].apply(lambda x: max(x, 0))
        df[col] = np.log1p(df[col])

    df['HasMortgage'] = (df['Mortgage'] > 0).astype(int)
    df.drop(columns=['Experience', 'Mortgage'], inplace=True, errors='ignore')
    return df

st.set_page_config(page_title="Loan Risk Predictor", layout="wide")
st.title(" Bank Loan Default Predictor")
st.write("Upload a CSV file to get predictions")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df_input = preprocess(df)
        X = df_input[model.feature_names_in_]

        df["Prediction"] = model.predict(X)
        df["Probability"] = model.predict_proba(X)[:, 1]

        st.success(" Prediction complete!")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="loan_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a CSV file to begin.")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

model = joblib.load("model.pkl")
required_features = model.feature_names_in_

st.set_page_config(page_title="Bank Loan Batch Predictor", layout="wide")
st.title(" Bank Loan Prediction on Uploaded CSV")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

def preprocess(df):
    df = df.copy()
    df.drop(columns=['ID', 'ZIP Code'], inplace=True, errors='ignore')
    df = df[df['Experience'] >= 0]

    for col in ['Income', 'CCAvg', 'Mortgage']:
        df[col] = df[col].apply(lambda x: max(x, 0))  
        df[col] = np.log1p(df[col])

    df['HasMortgage'] = df['Mortgage'].apply(lambda x: 1 if x > 0 else 0)
    df.drop(columns=['Experience', 'Mortgage'], inplace=True, errors='ignore')

    return df

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        df_processed = preprocess(df)

        X = df_processed[required_features]

        predictions = model.predict(X)
        df_processed.insert(0, "Prediction", predictions)

        st.success("Prediction Completed!")
        st.subheader(" Prediction Results")
        st.dataframe(df_processed)

        csv = df_processed.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv,
            file_name='loan_predictions.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("👆 Please upload a CSV file to begin.")

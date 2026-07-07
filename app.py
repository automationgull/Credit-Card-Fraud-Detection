import streamlit as st
import pandas as pd
import joblib

st.title("Credit Card Fraud Detection App")

model = joblib.load(r"D:\Pycharm\Fraude-Detector.Project\models\random_forest_model.pkl")
scaler = joblib.load(r"D:\Pycharm\Fraude-Detector.Project\models\scaler.pkl")

uploaded_file = st.file_uploader("Upload transaction CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    X = df.copy()

    if "Class" in X.columns:
        X = X.drop("Class", axis=1)

    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])

    predictions = model.predict(X)

    df["Prediction"] = predictions
    df["Prediction_Label"] = df["Prediction"].map({
        0: "Genuine",
        1: "Fraud"
    })

    st.subheader("Prediction Results")
    st.dataframe(df.head(20))

    st.success(f"Genuine Transactions: {(predictions == 0).sum()}")
    st.error(f"Fraud Transactions: {(predictions == 1).sum()}")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Results",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )
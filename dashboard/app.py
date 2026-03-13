import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("AI Legal Risk Intelligence Dashboard")

df = pd.read_csv("data/processed/legal_cases_enriched.csv")

# -------- Summary Metrics --------

col1, col2, col3 = st.columns(3)

col1.metric("Total Cases", len(df))
col2.metric("High Risk Cases", len(df[df["risk_level"] == "high_risk"]))
col3.metric("Unique Organizations", df["organization"].nunique())

# -------- Risk Distribution --------

st.subheader("Risk Distribution")

risk_counts = df["risk_level"].value_counts()

st.bar_chart(risk_counts)


# -------- Top Organizations --------

st.subheader("Organizations Appearing Most in Legal Cases")

org_counts = df["organization"].value_counts().head(10)

st.bar_chart(org_counts)

# -------- High Risk Cases --------

st.subheader("Sample High Risk Cases")

high_risk = df[df["risk_level"] == "high_risk"]

st.dataframe(high_risk[["case_name","organization","court","year"]].head(10))










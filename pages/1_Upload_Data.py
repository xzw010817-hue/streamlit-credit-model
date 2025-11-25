import streamlit as st
import pandas as pd

st.title("📁 데이터 업로드 (Upload Data)")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, skiprows=1, low_memory=False, engine='python')

    # strip spaces
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()

    st.session_state["raw_data"] = df

    st.success("데이터 업로드 성공!")
    st.dataframe(df.head())
else:
    st.info("CSV 파일을 업로드하세요.")

import streamlit as st
import pandas as pd

st.title("🎯 Feature Selection")

if "clean_data" not in st.session_state:
    st.error("먼저 전처리를 완료하세요.")
    st.stop()

df = st.session_state["clean_data"]

df = df.copy()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].astype('category')

all_features = [col for col in df.columns if col != "target"]

choice = st.multiselect("사용할 특성을 선택하세요", all_features)

if st.button("선택 완료"):
    st.session_state["selected_features"] = choice
    st.success("특성 선택 완료!")

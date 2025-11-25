import streamlit as st
import pandas as pd

st.set_page_config(page_title="Preprocessing", layout="wide")

st.title("데이터 전처리 (Preprocessing)")

# 업로드 데이터 확인
if "raw_data" not in st.session_state or st.session_state["raw_data"] is None:
    st.error("업로드된 데이터가 없습니다. 먼저 'Upload Data' 페이지에서 CSV 파일을 업로드하세요.")
    st.stop()

df = st.session_state["raw_data"].copy()

st.subheader("전처리 이전 데이터 미리보기")
st.dataframe(df.head())

# clean_lendingclub_final.csv 용 처리
if "target" not in df.columns:
    st.error("'target' 변수가 존재하지 않습니다. 원본 Lending Club 데이터를 업로드한 것이 맞는지 확인하세요.")
    st.stop()

# 전처리 완료(추가 작업 없음)
st.session_state["clean_data"] = df

st.success("전처리가 완료되었습니다.")
st.subheader("전처리 후 데이터 미리보기")
st.dataframe(df.head())

st.write(f"총 행 수: {df.shape[0]}, 총 열 수: {df.shape[1]}")

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Preprocessing", layout="wide")

st.title("데이터 전처리 (Preprocessing)")

# 세션 상태에 raw_data가 존재하는지 확인
if "raw_data" not in st.session_state or st.session_state["raw_data"] is None:
    st.error("업로드된 데이터가 없습니다. 먼저 'Upload Data' 페이지에서 CSV 파일을 업로드하세요.")
    st.stop()

df = st.session_state["raw_data"].copy()

st.subheader("전처리 이전 데이터 미리보기")
st.dataframe(df.head())


# 1. 불필요한 변수 제거
drop_cols = ["id", "member_id", "url", "desc", "zip_code"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])


# 2. loan_status → target 변환
if "loan_status" not in df.columns:
    st.error("loan_status 변수가 존재하지 않습니다. 올바른 Lending Club 데이터인지 확인하세요.")
    st.stop()

bad_status = [
    "Charged Off", "Default",
    "Late (31-120 days)", "Late (16-30 days)",
    "In Grace Period"
]
good_status = ["Fully Paid", "Current"]

mask = df["loan_status"].isin(bad_status + good_status)
df = df[mask].copy()

df["target"] = df["loan_status"].apply(lambda x: 1 if x in bad_status else 0)

# loan_status 삭제
df = df.drop(columns=["loan_status"])


# 3. 결측치 처리
for col in df.columns:
    if df[col].dtype == "float64" or df[col].dtype == "int64":
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna("Unknown")


# 4. 전처리 결과 저장
st.session_state["clean_data"] = df

st.success("전처리가 완료되었습니다.")

st.subheader("전처리 후 데이터 미리보기")
st.dataframe(df.head())

st.write(f"총 행 수: {df.shape[0]}, 총 열 수: {df.shape[1]}")

import streamlit as st
import pandas as pd

st.title("전처리 (Preprocessing)")

# CSV 체크
if "raw_data" not in st.session_state:
    st.error("먼저 원본 데이터를 업로드하세요.")
    st.stop()

df = st.session_state["raw_data"].copy()

st.subheader("1. 원본 데이터 미리보기")
st.dataframe(df.head())

# ---------------------------------------------------------
# term 컬럼 정리 (예: '60 months' → 60)
# ---------------------------------------------------------
if "term" in df.columns:
    df["term"] = (
        df["term"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )

# ---------------------------------------------------------
# int_rate 컬럼 정리 (예: '13.56%' → 13.56)
# ---------------------------------------------------------
if "int_rate" in df.columns:
    df["int_rate"] = (
        df["int_rate"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )

# ---------------------------------------------------------
# emp_length 컬럼 정리
# '10+ years' → 10
# '3 years' → 3
# '< 1 year' → 0
# 'n/a', NaN, Unknown → 0
# ---------------------------------------------------------

def clean_emp(x):
    x = str(x).lower()

    if "10" in x:
        return 10
    if "<" in x:
        return 0
    if "year" in x:
        try:
            return int(x.split()[0])
        except:
            return 0
    return 0

if "emp_length" in df.columns:
    df["emp_length"] = df["emp_length"].apply(clean_emp)

# ---------------------------------------------------------
# annual_inc 숫자 변환
# ---------------------------------------------------------
if "annual_inc" in df.columns:
    df["annual_inc"] = pd.to_numeric(df["annual_inc"], errors="coerce")

# ---------------------------------------------------------
# 사용할 주요 변수만 선택
# target 반드시 포함
# ---------------------------------------------------------
important_cols = [
    "loan_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "purpose",
    "addr_state", "dti", "fico_range_low", "fico_range_high"
]

# target 존재 여부 확인
if "target" not in df.columns:
    st.error("target 변수가 존재하지 않습니다. 전처리가 불가능합니다.")
    st.stop()

# 리스트에 실제 존재하는 변수만 필터링
final_cols = [c for c in important_cols if c in df.columns] + ["target"]

df = df[final_cols]

st.subheader("2. 전처리 후 데이터 미리보기")
st.dataframe(df.head())

# 최종 저장
st.session_state["clean_data"] = df
st.success("전처리가 완료되었습니다.")

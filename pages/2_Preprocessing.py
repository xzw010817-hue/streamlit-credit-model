import streamlit as st
import pandas as pd

st.title("🧹 전처리 (Preprocessing)")

if "raw_data" not in st.session_state:
    st.error("먼저 데이터를 업로드하세요!")
    st.stop()

df = st.session_state["raw_data"].copy()

st.subheader("📌 1. 원본 데이터 미리보기")
st.dataframe(df.head())

# ---------------------------
# Clean term
# ---------------------------
if "term" in df.columns:
    df["term"] = df["term"].str.extract("(\d+)").astype(int)

# ---------------------------
# Clean int_rate
# ---------------------------
if "int_rate" in df.columns:
    df["int_rate"] = df["int_rate"].str.replace("%","").astype(float)

# ---------------------------
# Clean emp_length
# ---------------------------
def clean_emp(x):
    x = str(x)
    if "10+" in x: return 10
    if "year" in x: return int(x.split()[0])
    return 0

if "emp_length" in df.columns:
    df["emp_length"] = df["emp_length"].apply(clean_emp)

# ---------------------------
# Convert income
# ---------------------------
if "annual_inc" in df.columns:
    df["annual_inc"] = pd.to_numeric(df["annual_inc"], errors='coerce')

# ---------------------------
#  Keep important features
# ---------------------------
important_cols = [
    'loan_amnt','term','int_rate','installment',
    'grade','sub_grade','emp_length','home_ownership',
    'annual_inc','verification_status','purpose',
    'addr_state','dti','fico_range_low','fico_range_high'
]

df = df[[c for c in important_cols if c in df.columns] + ['target']]

st.subheader("📌 2. 전처리 후 데이터")
st.dataframe(df.head())

st.session_state["clean_data"] = df
st.success("전처리가 완료되었습니다!")

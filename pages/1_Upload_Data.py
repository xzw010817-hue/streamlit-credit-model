import streamlit as st
import pandas as pd

st.set_page_config(page_title="Upload Data", layout="wide")

st.title("데이터 업로드 (Upload Data)")

st.markdown(
    """
    Lending Club 원본 CSV 데이터를 업로드하는 페이지입니다.

    업로드된 데이터는 **세션 상태(Session State)** 에 저장되며  
    이후 페이지(전처리, 특징 선택, 모델 학습 등)에서 자동으로 사용됩니다.
    """
)

# 파일 업로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

# 세션 상태 초기화
if "raw_data" not in st.session_state:
    st.session_state["raw_data"] = None

# 파일 처리
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["raw_data"] = df

    st.success("데이터가 성공적으로 업로드되었습니다.")

    st.subheader("데이터 미리보기")
    st.dataframe(df.head())

    st.write(f"총 행 수: {df.shape[0]},  총 열 수: {df.shape[1]}")

else:
    st.info("CSV 파일을 업로드해 주세요.")

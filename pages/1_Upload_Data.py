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

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if "raw_data" not in st.session_state:
    st.session_state["raw_data"] = None

if uploaded_file is not None:
    try:
        # Lending Club CSV의 첫 행은 설명문이므로 skiprows=1 적용
        df = pd.read_csv(uploaded_file, low_memory=False, skiprows=1)
    except:
        df = pd.read_csv(uploaded_file, low_memory=False)  # fallback

    st.session_state["raw_data"] = df

    st.success("데이터가 성공적으로 업로드되었습니다.")
    st.dataframe(df.head())
    st.write(f"총 행 수: {df.shape[0]}, 총 열 수: {df.shape[1]}")

else:
    st.info("CSV 파일을 업로드해 주세요.")
if st.button("🔄 세션 초기화 (Reset Session)"):
    for key in st.session_state.keys():
        st.session_state[key] = None
    st.success("세션이 초기화되었습니다. 다시 데이터를 업로드하세요.")
    st.stop()

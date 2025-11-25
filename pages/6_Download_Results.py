import streamlit as st

st.title("⬇️ Download Results")

if "result_df" not in st.session_state:
    st.error("먼저 모델을 학습하세요.")
    st.stop()

df = st.session_state["result_df"]

st.download_button(
    label=" 모델 예측 결과 다운로드 (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="credit_prediction_result.csv"
)

st.success("다운로드 준비 완료!")

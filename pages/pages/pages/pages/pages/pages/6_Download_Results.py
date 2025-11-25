import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Download Results", layout="wide")

st.title("결과 다운로드 (Download Results)")

# 데이터 존재 여부 확인
if "clean_data" not in st.session_state or st.session_state["clean_data"] is None:
    st.error("전처리된 데이터가 없습니다. 먼저 'Preprocessing' 페이지를 실행하세요.")
    st.stop()

if "trained_model" not in st.session_state or st.session_state["trained_model"] is None:
    st.error("학습된 모델이 없습니다. 'Model Training' 페이지에서 모델을 학습하세요.")
    st.stop()

clean_data = st.session_state["clean_data"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]
y_pred = st.session_state["y_pred"]
y_prob = st.session_state["y_prob"]
model = st.session_state["trained_model"]


st.subheader("전처리된 데이터 다운로드")
clean_csv = clean_data.to_csv(index=False).encode("utf-8")
st.download_button(
    label="전처리된 데이터 다운로드 (clean_data.csv)",
    data=clean_csv,
    file_name="clean_data.csv",
    mime="text/csv"
)


st.subheader("X_test 다운로드")
xtest_csv = X_test.to_csv(index=False).encode("utf-8")
st.download_button(
    label="X_test 다운로드 (X_test.csv)",
    data=xtest_csv,
    file_name="X_test.csv",
    mime="text/csv"
)


st.subheader("y_test 다운로드")
ytest_csv = pd.DataFrame({"y_test": y_test}).to_csv(index=False).encode("utf-8")
st.download_button(
    label="y_test 다운로드 (y_test.csv)",
    data=ytest_csv,
    file_name="y_test.csv",
    mime="text/csv"
)


st.subheader("예측값 다운로드")
pred_csv = pd.DataFrame({
    "y_test": y_test,
    "y_pred": y_pred,
    "y_prob": y_prob
}).to_csv(index=False).encode("utf-8")
st.download_button(
    label="예측 결과 다운로드 (predictions.csv)",
    data=pred_csv,
    file_name="predictions.csv",
    mime="text/csv"
)


st.subheader("모델 파일 다운로드 (pickle)")
model_pickle = pickle.dumps(model)
st.download_button(
    label="모델 다운로드 (model.pkl)",
    data=model_pickle,
    file_name="model.pkl",
    mime="application/octet-stream"
)


st.info("모든 결과 파일은 CSV 또는 PKL 형식으로 다운로드 가능합니다.")

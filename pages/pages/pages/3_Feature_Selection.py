import streamlit as st
import pandas as pd

st.set_page_config(page_title="Feature Selection", layout="wide")

st.title("특징 선택 (Feature Selection)")

# 전처리된 데이터 확인
if "clean_data" not in st.session_state or st.session_state["clean_data"] is None:
    st.error("전처리된 데이터가 없습니다. 먼저 'Preprocessing' 페이지에서 전처리를 진행하세요.")
    st.stop()

df = st.session_state["clean_data"].copy()

st.subheader("사용 가능한 전체 변수 목록")
st.write(list(df.columns))

# target 제거 후 Feature 후보만 선택 가능하게 함
candidate_features = [c for c in df.columns if c != "target"]

st.subheader("모델에 사용할 특징(Feature)을 선택하세요.")
selected_features = st.multiselect(
    "특징 선택",
    options=candidate_features,
)

if "selected_features" not in st.session_state:
    st.session_state["selected_features"] = None

if st.button("특징 선택 확인"):
    if len(selected_features) == 0:
        st.error("하나 이상의 특징을 선택해야 합니다.")
    else:
        st.session_state["selected_features"] = selected_features
        st.success("특징이 성공적으로 저장되었습니다.")
        st.write("선택된 특징:", selected_features)

st.info("선택된 특징은 이후 모델 학습 단계에서 자동으로 사용됩니다.")

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Credit Scoring System",
    layout="wide",
    page_icon="📊"
)

# --- CSS Styling ---
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
h1 {
    color: #1F4E79;
    text-align: center;
    font-weight: 800 !important;
    font-size: 2.5rem !important;
}
.subtitle {
    color: #4A4A4A;
    font-size: 1.15rem;
    text-align: center;
    margin-bottom: 2rem;
}
.box {
    background-color: #EAF2F8;
    padding: 1.2rem;
    border-radius: 10px;
    border-left: 5px solid #1F4E79;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# Page Title
st.markdown("<h1>📊 Intelligent Credit Scoring System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>LendingClub 데이터를 기반으로 한 지능형 신용평가 모델링 시스템</p>", unsafe_allow_html=True)

# Intro Boxes
st.markdown("""
<div class='box'>
<b>📁 1. 데이터 업로드 → </b> LendingClub 원본 CSV 업로드  
</div>

<div class='box'>
<b>🧹 2. 데이터 전처리 → </b> 숫자·문자열·범주형 변수를 학술 기준에 맞추어 정제  
</div>

<div class='box'>
<b>🎯 3. Feature Selection → </b> 모델 성능에 직접 영향을 미치는 핵심 변수 선택  
</div>

<div class='box'>
<b>🤖 4. Model Training → </b> Logistic / RandomForest / XGBoost 모델 학습  
</div>

<div class='box'>
<b>📈 5. Evaluation → </b> ROC Curve, Confusion Matrix로 모델 성능 검증  
</div>

<div class='box'>
<b>⬇️ 6. Download → </b> 등급(1~5등급) 점수 포함한 최종 예측 결과 다운로드  
</div>
""", unsafe_allow_html=True)

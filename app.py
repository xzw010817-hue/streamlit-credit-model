import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="신용평가 대시보드 (Credit Risk Dashboard)",
    page_icon="📊",
    layout="wide"
)

# 메인 페이지 제목
st.title("📊 신용평가 모델 대시보드")
st.markdown(
    """
    ### 환영합니다!  
    이 대시보드는 **Lending Club 데이터**를 기반으로  
    - 데이터 업로드  
    - 전처리  
    - 특징 선택  
    - 머신러닝 모델 학습(Logistic / XGBoost / MLP)  
    - 하이브리드 스태킹 모델  
    - 성능평가 (AUC, 혼동행렬 등)  
    - 결과 다운로드  

    을 진행할 수 있도록 설계된 **다중 페이지 Streamlit 시스템**입니다.
    
    왼쪽 사이드바에서 원하는 메뉴를 선택해 주세요.
    """
)

st.info("왼쪽 사이드바의 페이지 목록에서 단계별로 진행하세요."）

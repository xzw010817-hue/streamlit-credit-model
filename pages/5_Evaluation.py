import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="Model Evaluation", layout="wide")

st.title("모델 성능 평가 (Model Evaluation)")

# 1. 학습된 모델 확인
if ("trained_model" not in st.session_state or 
    st.session_state["trained_model"] is None):
    st.error("학습된 모델이 없습니다. 'Model Training' 페이지에서 먼저 모델을 학습하세요.")
    st.stop()

model = st.session_state["trained_model"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]
y_pred = st.session_state["y_pred"]
y_prob = st.session_state["y_prob"]

# 2. 성능 지표 계산
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

st.subheader("평가 지표")
st.write(f"정확도 (Accuracy): {accuracy:.4f}")
st.write(f"정밀도 (Precision): {precision:.4f}")
st.write(f"재현율 (Recall): {recall:.4f}")
st.write(f"F1-score: {f1:.4f}")
st.write(f"AUC (ROC Curve): {auc:.4f}")


# 3. ROC Curve
st.subheader("ROC Curve")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend()

st.pyplot(fig_roc)


# 4. Confusion Matrix
st.subheader("혼동 행렬 (Confusion Matrix)")
cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("예측값 (Predicted)")
ax_cm.set_ylabel("실제값 (Actual)")

st.pyplot(fig_cm)


# 5. Classification Report
st.subheader("분류 보고서 (Classification Report)")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)


st.info("모든 평가 지표는 학습된 모델의 테스트 데이터 기반으로 계산되었습니다.")

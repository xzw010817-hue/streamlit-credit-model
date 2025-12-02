import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

st.title("모델 평가 (Evaluation)")

# ---------------------------------------------
# 필수 상태 확인
# ---------------------------------------------
required_keys = [
    "trained_models", "X_test", "y_test"
]

for k in required_keys:
    if k not in st.session_state:
        st.error("Model Training을 먼저 진행하십시오.")
        st.stop()

models = st.session_state["trained_models"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]

# ---------------------------------------------
# 모델 성능 비교
# ---------------------------------------------
st.subheader("모델 성능 비교")

results = []

pred_probs = {}

for name, clf in models.items():
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    results.append([name, acc, auc])
    pred_probs[name] = prob

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "AUC"])
st.dataframe(results_df)

# ---------------------------------------------
# Accuracy 비교
# ---------------------------------------------
st.subheader("Accuracy 비교")
fig_acc, ax_acc = plt.subplots()
sns.barplot(data=results_df, x="Model", y="Accuracy", ax=ax_acc)
ax_acc.set_ylim(0, 1)
st.pyplot(fig_acc)

# ---------------------------------------------
# AUC 비교
# ---------------------------------------------
st.subheader("ROC-AUC 비교")
fig_auc, ax_auc = plt.subplots()
sns.barplot(data=results_df, x="Model", y="AUC", ax=ax_auc)
ax_auc.set_ylim(0, 1)
st.pyplot(fig_auc)

# ---------------------------------------------
# ROC Curve
# ---------------------------------------------
st.subheader("ROC Curve")
fig_roc, ax_roc = plt.subplots()

for name, prob in pred_probs.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax_roc.plot(fpr, tpr, label=name)

ax_roc.set_xlabel("FPR")
ax_roc.set_ylabel("TPR")
ax_roc.legend()

st.pyplot(fig_roc)

# ---------------------------------------------
# Confusion Matrix 3개 모델
# ---------------------------------------------
st.subheader("Confusion Matrix (모델별)")

for name, clf in models.items():
    pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    st.write(f"모델: {name}")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("예측값")
    ax_cm.set_ylabel("실제값")
    st.pyplot(fig_cm)

st.success("모델 평가가 완료되었습니다.")

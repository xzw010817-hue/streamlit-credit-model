import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.title(" 모델 평가 (Evaluation)")

if "result_df" not in st.session_state:
    st.error("모델 학습부터 진행하세요.")
    st.stop()

df = st.session_state["result_df"]

st.subheader("등급 분포")
st.bar_chart(df["grade"].value_counts())

st.subheader("Confusion Matrix")
y_true = df["y_test"]
y_pred = (df["probability"]>=0.5).astype(int)

cm = confusion_matrix(y_true,y_pred)

fig,ax=plt.subplots()
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
st.pyplot(fig)

st.subheader("ROC Curve")
fpr,tpr,_ = roc_curve(y_true,df["probability"])
fig2,ax2 = plt.subplots()
ax2.plot(fpr,tpr,label=f"AUC={auc(fpr,tpr):.3f}")
ax2.set_title("ROC Curve")
ax2.set_xlabel("FPR")
ax2.set_ylabel("TPR")
st.pyplot(fig2)

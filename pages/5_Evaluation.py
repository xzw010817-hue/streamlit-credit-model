import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, roc_auc_score

st.title("모델 평가 (Evaluation)")

# ------------------------------------------------
# 1. 训练是否完成？
# ------------------------------------------------
required_keys = [
    "trained_models", "trained_model_single",
    "result_df", "X_train", "X_test", "y_train", "y_test"
]

for key in required_keys:
    if key not in st.session_state:
        st.error("Model Training을 먼저 진행하십시오.")
        st.stop()

models = st.session_state["trained_models"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]

# ------------------------------------------------
# 2. 多模型性能比较
# ------------------------------------------------
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

# ------------------------------
# Accuracy Bar Chart
# ------------------------------
st.subheader("Accuracy 비교")
fig_acc, ax_acc = plt.subplots()
sns.barplot(data=results_df, x="Model", y="Accuracy", ax=ax_acc)
plt.ylim(0, 1)
st.pyplot(fig_acc)

# ------------------------------
# AUC Bar Chart
# ------------------------------
st.subheader("AUC 비교")
fig_auc, ax_auc = plt.subplots()
sns.barplot(data=results_df, x="Model", y="AUC", ax=ax_auc)
plt.ylim(0, 1)
st.pyplot(fig_auc)

# ------------------------------
# ROC Curve
# ------------------------------
st.subheader("ROC Curve")
fig_roc, ax_roc = plt.subplots()
for name, prob in pred_probs.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax_roc.plot(fpr, tpr, label=name)

ax_roc.set_xlabel("FPR")
ax_roc.set_ylabel("TPR")
ax_roc.legend()
st.pyplot(fig_roc)

# ------------------------------
# Confusion Matrix
# ------------------------------
st.subheader("Confusion Matrix")
for name, clf in models.items():
    pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    st.write(f"모델: {name}")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)

# ------------------------------
# Feature Importance (RF)
# ------------------------------
st.subheader("Feature Importance (RandomForest)")

rf_model = models["RandomForest"]

# 预处理器
pre = rf_model.named_steps["preprocess"]
rf = rf_model.named_steps["model"]

# 生成特征名称
ohe = pre.named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(pre.transformers_[1][2]))
num_feature_names = pre.transformers_[0][2]
feature_names = num_feature_names + cat_feature_names

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

fig_imp, ax_imp = plt.subplots()
sns.barplot(data=importance_df.head(15),
            x="importance", y="feature", ax=ax_imp)
st.pyplot(fig_imp)

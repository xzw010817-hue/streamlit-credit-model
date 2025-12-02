import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

# ------------------------------------------------------------
# Page Title
# ------------------------------------------------------------
st.title("Model Evaluation")

# ------------------------------------------------------------
# Check if training has been completed
# ------------------------------------------------------------
required_keys = [
    "trained_models",
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "preprocessor"
]

for key in required_keys:
    if key not in st.session_state:
        st.error("Please run Model Training first.")
        st.stop()

models = st.session_state["trained_models"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]
preprocessor = st.session_state["preprocessor"]

# ------------------------------------------------------------
# Evaluate models
# ------------------------------------------------------------
st.subheader("Model Performance Comparison")

results = []
pred_probs = {}

for name, clf in models.items():
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    results.append([name, acc, auc])
    pred_probs[name] = {"pred": pred, "prob": prob}

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "AUC"])
st.dataframe(results_df, use_container_width=True)

# ------------------------------------------------------------
# Accuracy Bar Chart
# ------------------------------------------------------------
st.subheader("Accuracy Comparison")

fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
sns.barplot(data=results_df, x="Model", y="Accuracy", ax=ax_acc)
ax_acc.set_ylim(0, 1)
plt.xticks(rotation=30)
st.pyplot(fig_acc)

# ------------------------------------------------------------
# AUC Bar Chart
# ------------------------------------------------------------
st.subheader("AUC Comparison")

fig_auc, ax_auc = plt.subplots(figsize=(6, 4))
sns.barplot(data=results_df, x="Model", y="AUC", ax=ax_auc)
ax_auc.set_ylim(0, 1)
plt.xticks(rotation=30)
st.pyplot(fig_auc)

# ------------------------------------------------------------
# ROC Curve
# ------------------------------------------------------------
st.subheader("ROC Curve Comparison")

fig_roc, ax_roc = plt.subplots(figsize=(7, 5))

for name, data in pred_probs.items():
    fpr, tpr, _ = roc_curve(y_test, data["prob"])
    ax_roc.plot(fpr, tpr, label=name)

ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()
st.pyplot(fig_roc)

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------
st.subheader("Confusion Matrix (All Models)")

for name, data in pred_probs.items():
    st.write(f"### Model: {name}")

    cm = confusion_matrix(y_test, data["pred"])

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)

    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")

    st.pyplot(fig_cm)

# ------------------------------------------------------------
# Feature Importance (RandomForest)
# ------------------------------------------------------------
st.subheader("Feature Importance (RandomForest)")

if "RandomForest" in models:
    rf_model = models["RandomForest"]

    fitted_rf = rf_model.named_steps["model"]
    encoder = rf_model.named_steps["preprocess"].named_transformers_["cat"]

    X_train = st.session_state["X_train"]
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "category"]
    num_cols = [c for c in X_train.columns if X_train[c].dtype != "category"]

    if len(cat_cols) > 0:
        cat_feature_names = list(encoder.get_feature_names_out(cat_cols))
    else:
        cat_feature_names = []

    feature_names = num_cols + cat_feature_names

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": fitted_rf.feature_importances_
    }).sort_values("importance", ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(6, 6))
    sns.barplot(
        data=importance_df.head(15),
        x="importance",
        y="feature",
        ax=ax_imp
    )

    ax_imp.set_xlabel("Importance Score")
    ax_imp.set_ylabel("Feature")
    st.pyplot(fig_imp)
else:
    st.info("RandomForest model is not available.")

st.success("Evaluation completed successfully!")

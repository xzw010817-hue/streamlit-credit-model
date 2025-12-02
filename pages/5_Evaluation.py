import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, roc_curve
)

st.title("모델 평가 (Evaluation)")

if "clean_data" not in st.session_state or "selected_features" not in st.session_state:
    st.error("전처리 및 Feature 선택을 먼저 진행하십시오.")
    st.stop()

# Import models re-train
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier

df = st.session_state["clean_data"]
features = st.session_state["selected_features"]
X = df[features]
y = df["target"]

cat_cols = [c for c in X.columns if X[c].dtype == "category"]
num_cols = [c for c in X.columns if X[c].dtype != "category"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=42
)

# Define models
model_lr = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=500, class_weight="balanced"))
])

model_rf = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced"))
])

stacking_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
            ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced"))
        ],
        final_estimator=LogisticRegression(max_iter=500),
        stack_method="predict_proba"
    ))
])

models = {
    "Logistic Regression": model_lr,
    "RandomForest": model_rf,
    "Hybrid Stacking": stacking_model
}

results = []

# Train all models
for name, clf in models.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    results.append([name, acc, auc])

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "AUC"])

st.subheader("모델 성능 비교 표")
st.dataframe(results_df)

# Accuracy Bar Chart
st.subheader("Accuracy 비교")
fig1, ax1 = plt.subplots()
sns.barplot(data=results_df, x="Model", y="Accuracy", ax=ax1)
plt.ylim(0, 1)
st.pyplot(fig1)

# AUC Bar Chart
st.subheader("ROC-AU

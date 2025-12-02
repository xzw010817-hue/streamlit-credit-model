import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

st.title("모델 평가 (Evaluation)")

# ---------------------------------------------
# 필수 데이터 확인
# ---------------------------------------------
if "clean_data" not in st.session_state or "selected_features" not in st.session_state:
    st.error("전처리와 Feature 선택을 먼저 진행하십시오.")
    st.stop()

df = st.session_state["clean_data"]
features = st.session_state["selected_features"]

X = df[features]
y = df["target"]

# ---------------------------------------------
# 컬럼 분리
# ---------------------------------------------
cat_cols = [col for col in X.columns if X[col].dtype == "category"]
num_cols = [col for col in X.columns if X[col].dtype != "category"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# ---------------------------------------------
# Train Test Split
# ---------------------------------------------
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=42
)

# ---------------------------------------------
# 세 가지 모델 정의
# ---------------------------------------------
model_lr = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=500, class_weight="balanced"))
])

model_rf = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced"))
])

model_stacking = Pipeline([
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
    "Hybrid Stacking": model_stacking
}

# ---------------------------------------------
# 모델 학습 및 성능 저장
# ---------------------------------------------
results = []
pred_probs = {}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    results.append([name, acc, auc])
    pred_probs[name] = prob

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "AUC"])

# ---------------------------------------------
# 성능 비교 표
# ---------------------------------------------
st.subheader("모델 성능 비교")
st.dataframe(results_df)

# ---------------------------------------------
# Accuracy Chart
# ---------------------------------------------
st.subheader("Accuracy 비교")
fig_acc, ax_acc = plt.subplots()
sns.barplot(data=results_df, x="Model", y="Accuracy", ax=ax_acc)
ax_acc.set_ylim(0, 1)
st.pyplot(fig_acc)

# ---------------------------------------------
# AUC Chart
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
# Confusion Matrix
# ---------------------------------------------
st.subheader("Confusion Matrix")

for name, clf in models.items():
    pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    st.write(f"모델: {name}")

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("예측값")
    ax_cm.set_ylabel("실제값")
    st.pyplot(fig_cm)

# ---------------------------------------------
# Feature Importance (RandomForest)
# ---------------------------------------------
st.subheader("Feature Importance (RandomForest)")

# ① 전처리와 모델을 분리하여 다시 학습 (오류 방지)
ohe_encoder = OneHotEncoder(handle_unknown="ignore")

preprocessor_fit = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", ohe_encoder, cat_cols)
    ]
)

# ② fit_transform 수행 → 모든 feature 생성 완료
X_train_trans = preprocessor_fit.fit_transform(X_train)

# ③ RandomForest만 다시 학습 (preprocess 포함하지 않음)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced")
rf.fit(X_train_trans, y_train)

# ④ OneHotEncoder에서 실제 컬럼명 가져오기
cat_feature_names = list(ohe_encoder.get_feature_names_out(cat_cols))
feature_names = num_cols + cat_feature_names

# ⑤ 중요도 테이블 생성
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

# ⑥ 중요도 그래프 출력
fig_imp, ax_imp = plt.subplots()
sns.barplot(data=importance_df.head(15), x="importance", y="feature", ax=ax_imp)
st.pyplot(fig_imp)

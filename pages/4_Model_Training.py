import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

st.title("모델 학습 (Model Training)")

if "clean_data" not in st.session_state or "selected_features" not in st.session_state:
    st.error("전처리 및 Feature 선택을 먼저 진행하십시오.")
    st.stop()

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

model_option = st.selectbox("모델 선택", ["Logistic Regression", "RandomForest", "Hybrid Stacking"])

test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=42
)

# Logistic Regression
model_lr = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=500, class_weight="balanced"))
])

# RandomForest
model_rf = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced"))
])

# XGBoost
model_xgb = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        eval_metric="auc"
    ))
])

# Hybrid Stacking
stacking_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
            ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced")),
        ],
        final_estimator=LogisticRegression(max_iter=500),
        stack_method="predict_proba"
    ))
])

if model_option == "Logistic Regression":
    clf = model_lr
elif model_option == "RandomForest":
    clf = model_rf
else:
    clf = stacking_model

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)[:, 1]

# 등급 변환 함수
def credit_grade(p):
    if p >= 0.8: return "1등급"
    if p >= 0.6: return "2등급"
    if p >= 0.4: return "3등급"
    if p >= 0.2: return "4등급"
    return "5등급"

grades = [credit_grade(p) for p in prob]

st.subheader("예측 결과")
result_df = pd.DataFrame({
    "y_test": y_test,
    "probability": prob,
    "grade": grades
})
st.dataframe(result_df.head())

# 성능 출력
acc = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, prob)

st.write("Accuracy:", acc)
st.write("ROC-AUC:", auc)

# Classification Report
st.subheader("Classification Report")
st.text(classification_report(y_test, pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
st.pyplot(fig)

# ROC Curve
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, prob)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr)
ax2.set_xlabel("FPR")
ax2.set_ylabel("TPR")
st.pyplot(fig2)

st.session_state["trained_model"] = clf
st.session_state["result_df"] = result_df

st.success("모델 학습 완료")

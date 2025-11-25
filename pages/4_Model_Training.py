import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

st.title("모델 학습 (Model Training)")

# 세션 상태 확인
if "clean_data" not in st.session_state or "selected_features" not in st.session_state:
    st.error("전처리 및 Feature 선택을 먼저 진행하십시오.")
    st.stop()

df = st.session_state["clean_data"]
features = st.session_state["selected_features"]

X = df[features]
y = df["target"]

# 수치형 / 범주형 자동 분리
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if X[c].dtype != "object"]

# ColumnTransformer 정의
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# 모델 선택
model_option = st.selectbox("모델 선택", ["Logistic Regression", "RandomForest", "XGBoost"])

if model_option == "Logistic Regression":
    clf = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=700, class_weight="balanced"))
    ])
elif model_option == "RandomForest":
    clf = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        ))
    ])
else:
    clf = Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        ))
    ])

# 테스트 데이터 비율 선택
test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.3)

# 데이터 분할 (층화추출)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    stratify=y,
    random_state=42
)

# 모델 학습
clf.fit(X_train, y_train)

# 예측
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)[:, 1]

# 등급 함수
def credit_grade(p):
    if p >= 0.8: return "1등급"
    if p >= 0.6: return "2등급"
    if p >= 0.4: return "3등급"
    if p >= 0.2: return "4등급"
    return "5등급"

grades = [credit_grade(p) for p in prob]

# 결과 테이블
st.subheader("예측 결과")
result_df = pd.DataFrame({
    "y_test": y_test.values,
    "probability": prob,
    "grade": grades
})

st.dataframe(result_df.head())

# 평가 지표
st.write("Accuracy:", accuracy_score(y_test, pred))
st.write("ROC-AUC:", roc_auc_score(y_test, prob))
# -----------------------
# 이탈율 계산
# -----------------------

# 실제 이탈율 (실제 y_test 기준)
true_churn_rate = y_test.mean()

# 모델 예측 이탈율 (예측 확률 기준)
pred_churn_rate = (prob >= 0.5).mean()

# 등급별 이탈율 계산
grade_churn_rate = (
    pd.DataFrame({"grade": grades, "y_test": y_test})
    .groupby("grade")["y_test"]
    .mean()
)

st.subheader("이탈율(Default / Churn Rate)")
st.write(f"전체 실제 이탈율: {true_churn_rate:.4f}")
st.write(f"모델 예측 이탈율(0.5 기준): {pred_churn_rate:.4f}")

st.write("등급별 이탈율:")
st.dataframe(grade_churn_rate)

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

st.title("🤖 모델 학습 (Model Training)")

if "clean_data" not in st.session_state or "selected_features" not in st.session_state:
    st.error("전처리 및 Feature 선택을 먼저 진행하세요.")
    st.stop()

df = st.session_state["clean_data"]
features = st.session_state["selected_features"]

X = df[features]
y = df["target"]

cat_cols = [c for c in X.columns if X[c].dtype.name == 'category']
num_cols = [c for c in X.columns if X[c].dtype.name != 'category']

preprocessor = ColumnTransformer(
    transformers=[
        ("num","passthrough",num_cols),
        ("cat",OneHotEncoder(handle_unknown='ignore'),cat_cols)
    ]
)

model_option = st.selectbox("모델 선택", ["Logistic Regression","RandomForest","XGBoost"])

if model_option == "Logistic Regression":
    clf = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=500,class_weight="balanced"))
    ])
elif model_option == "RandomForest":
    clf = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300,class_weight="balanced"))
    ])
else:
    clf = Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,learning_rate=0.05,max_depth=6,
            eval_metric='auc'
        ))
    ])

test_size = st.slider("테스트 데이터 비율",0.1,0.5,0.3)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=test_size,stratify=y,random_state=42
)

clf.fit(X_train,y_train)
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)[:,1]

# 등급화 함수
def credit_grade(p):
    if p>=0.8: return "1등급"
    if p>=0.6: return "2등급"
    if p>=0.4: return "3등급"
    if p>=0.2: return "4등급"
    return "5등급"

grades = [credit_grade(p) for p in prob]

st.subheader("📌 예측 결과")
result_df = pd.DataFrame({
    "y_test":y_test,
    "probability":prob,
    "grade":grades
})

st.dataframe(result_df.head())

st.write("Accuracy:", accuracy_score(y_test,pred))
st.write("ROC-AUC:", roc_auc_score(y_test,prob))

st.subheader("Classification Report")
st.text(classification_report(y_test,pred))

st.session_state["trained_model"] = clf
st.session_state["result_df"] = result_df

st.success("모델 학습 완료!")

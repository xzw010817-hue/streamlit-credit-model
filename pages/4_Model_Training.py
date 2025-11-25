import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

st.set_page_config(page_title="Model Training", layout="wide")
st.title("모델 학습 (Model Training)")

# ------------------------
# 1. Check preprocessed data
# ------------------------
if "clean_data" not in st.session_state:
    st.error("전처리 단계의 데이터가 없습니다. Preprocessing 페이지에서 먼저 진행하세요.")
    st.stop()

df = st.session_state["clean_data"].copy()

if "selected_features" not in st.session_state:
    st.error("선택된 Feature가 없습니다. Feature Selection 페이지에서 먼저 선택하세요.")
    st.stop()

features = st.session_state["selected_features"]
st.write("### 사용된 특성")
st.json(features)

# ------------------------
# 2. Train-test split
# ------------------------
X = df[features]
y = df["target"]

test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.3)

# detect categorical columns
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

# preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

# ------------------------
# 3. Model selection
# ------------------------
st.write("### 학습할 모델 선택")

model_choice = st.selectbox(
    "모델을 선택하세요.",
    ["Logistic Regression", "RandomForest", "XGBoost", "MLP Neural Network"]
)

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "RandomForest":
    model = RandomForestClassifier(n_estimators=300)
elif model_choice == "XGBoost":
    model = xgb.XGBClassifier(eval_metric="logloss")
elif model_choice == "MLP Neural Network":
    model = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300)

# Full pipeline
clf = Pipeline(steps=[("preprocess", preprocessor),
                     ("model", model)])

# ------------------------
# 4. Train model
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)[:,1]

# ------------------------
# 5. Show evaluation
# ------------------------
st.write("### 예측 결과 미리보기")
st.dataframe(pd.DataFrame({"y_test": y_test.values, "y_pred": pred}).head())

st.write("### Accuracy:", accuracy_score(y_test, pred))
st.write("### ROC-AUC:", roc_auc_score(y_test, prob))

st.write("### Classification Report")
st.text(classification_report(y_test, pred))

# Save model to session_state
st.session_state["trained_model"] = clf

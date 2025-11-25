import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="Model Training", layout="wide")

st.title("모델 학습 (Model Training)")

# 1. 전처리 데이터 확인
if "clean_data" not in st.session_state or st.session_state["clean_data"] is None:
    st.error("전처리된 데이터가 없습니다. 'Preprocessing' 페이지에서 먼저 실행하세요.")
    st.stop()

df = st.session_state["clean_data"].copy()

# 2. 선택된 특징 확인
if "selected_features" not in st.session_state or st.session_state["selected_features"] is None:
    st.error("선택된 특징이 없습니다. 'Feature Selection' 페이지에서 먼저 특징을 선택하세요.")
    st.stop()

features = st.session_state["selected_features"]
X = df[features]
y = df["target"]

st.subheader("사용된 특징")
st.write(features)


# 3. Train/Test Split
test_size = st.slider("테스트 데이터 비율", 0.1, 0.4, 0.3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=42
)

# 4. 스케일링 (MLP, Logistic용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


st.subheader("학습할 모델 선택")

model_option = st.selectbox(
    "모델을 선택하세요.",
    ("Logistic Regression", "RandomForest", "XGBoost", "MLP Neural Network", "Hybrid Stacking")
)

train_button = st.button("모델 학습 시작")


# 학습 결과 저장 공간
if train_button:

    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    elif model_option == "RandomForest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    elif model_option == "XGBoost":
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    elif model_option == "MLP Neural Network":
        model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    elif model_option == "Hybrid Stacking":
        estimators = [
            ("logit", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("xgb", XGBClassifier()),
        ]

        model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            passthrough=False,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # 학습 결과 저장
    st.session_state["trained_model"] = model
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["y_pred"] = y_pred
    st.session_state["y_prob"] = y_prob

    st.success("모델 학습이 완료되었습니다.")
    st.write("선택된 모델:", model_option)

    st.subheader("예측 결과 미리보기")
    preview_df = pd.DataFrame({"y_test": y_test.values, "y_pred": y_pred})
    st.dataframe(preview_df.head())

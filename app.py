import streamlit as st
import pandas as pd
import pickle

# モデルの読み込み
@st.cache_resource
def load_model(path="models/best_model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("回帰予測アプリ - FIMアウト予測")

# 入力方法の選択
input_type = st.radio("入力方法を選択してください", ("手入力", "CSVアップロード"))

# 入力データの準備
if input_type == "手入力":
    fim_in = st.number_input("FIM入院時点（fim_in）", min_value=0, max_value=126, value=60)
    age = st.number_input("年齢（age）", min_value=0, max_value=120, value=80)
    mmse = st.number_input("MMSE", min_value=0, max_value=30, value=20)
    paralysis = st.selectbox("麻痺の種類（paralysis）", options=[0, 1, 2, 3, 4], format_func=lambda x: f"{x}（例）")
    
    input_df = pd.DataFrame([{
        "fim_in": fim_in,
        "age": age,
        "mmse": mmse,
        "paralysis": paralysis
    }])
else:
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        input_df = None

# 予測ボタン
if input_df is not None and st.button("予測実行"):
    expected_features = ["fim_in", "age", "mmse", "paralysis"]  # 必要なら正確な順番に変更
    input_df = input_df[expected_features]

    preds = model.predict(input_df)
    input_df["予測FIMアウト"] = preds
    st.subheader("予測結果")
    st.dataframe(input_df)

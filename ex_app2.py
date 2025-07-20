import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import lightgbm as lgb # LightGBMをインポート
from sklearn.metrics import mean_squared_error # 学習の評価用にインポート

# モデルの読み込み
@st.cache_resource
def load_model(path="models/best_model.pkl"):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning("既存のモデルファイルが見つかりませんでした。新しいモデルを学習させてください。")
        return None # モデルが見つからない場合はNoneを返す

model = load_model()

st.title("回帰予測アプリ - FIMアウト予測")

# --- 学習用データ読み込みとモデル学習/保存機能 ---
st.header("モデル学習・保存機能")

# 学習用CSVファイルのアップロード
uploaded_train_file = st.file_uploader("学習用CSVファイルをアップロードしてください (オプション)", type=["csv"], key="train_uploader")

if uploaded_train_file is not None:
    train_df = pd.read_csv(uploaded_train_file)
    st.subheader("アップロードされた学習用データ（一部）")
    st.dataframe(train_df.head())

    # ターゲット変数と特徴量の選択
    st.markdown("---")
    st.subheader("学習設定")
    target_column = st.selectbox("ターゲット変数を選択してください", train_df.columns)
    feature_columns = st.multiselect("特徴量を選択してください", [col for col in train_df.columns if col != target_column])

    if st.button("モデルを学習して保存"):
        if target_column and feature_columns:
            X = train_df[feature_columns]
            y = train_df[target_column]

            # 訓練データとテストデータに分割 (例: 80%訓練、20%テスト)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # ここをLightGBMモデルに変更
            new_model = lgb.LGBMRegressor(random_state=42) # ハイパーパラメータは適宜調整してください
            new_model.fit(X_train, y_train)

            # モデルの評価 (オプション)
            y_pred = new_model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False) # RMSEを計算
            st.write(f"モデル学習が完了しました。テストセットでのRMSE: {rmse:.2f}")

            # 学習済みモデルの保存
            model_save_path = "models/new_trained_model.pkl" # 保存パスを適宜変更してください
            with open(model_save_path, "wb") as f:
                pickle.dump(new_model, f)
            st.success(f"新しいモデルが **{model_save_path}** に保存されました。")

            # 新しいモデルを現在のセッションで使用するように更新
            model = new_model
            st.info("予測機能で新しいモデルが使用されます。")
        else:
            st.warning("ターゲット変数と特徴量をすべて選択してください。")
else:
    st.info("学習用CSVファイルをアップロードすると、新しいモデルを学習・保存できます。")

st.markdown("---")

# --- 既存の予測機能 ---
st.header("回帰予測")

# 入力方法の選択
input_type = st.radio("入力方法を選択してください", ("手入力", "CSVアップロード"), key="input_type_prediction")

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
    uploaded_file = st.file_uploader("予測用CSVファイルをアップロードしてください", type=["csv"], key="prediction_uploader")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        input_df = None

# 予測ボタン
if input_df is not None and st.button("予測実行"):
    if model is not None:
        expected_features = ["fim_in", "age", "mmse", "paralysis"]  # 必要なら正確な順番に変更
        
        # アップロードされたCSVの場合、特徴量が揃っているか確認
        if not all(feature in input_df.columns for feature in expected_features):
            st.error(f"アップロードされたCSVには、必要な特徴量 ({', '.join(expected_features)}) がすべて含まれていません。")
        else:
            input_df = input_df[expected_features]

            preds = model.predict(input_df)
            input_df["予測FIMアウト"] = preds
            st.subheader("予測結果")
            st.dataframe(input_df)
    else:
        st.warning("モデルがロードされていないか、まだ学習されていません。モデルを学習させるか、既存のモデルをロードしてください。")
import streamlit as st
import pandas as pd
import lightgbm as lgb
import pickle
import os

# --- データの読み込み、モデル学習、モデル保存 ---
def train_and_save_model():
    st.header("モデルの学習と保存")
    st.write("ここではサンプルデータを使ってLightGBMモデルを学習し、保存します。")

    # サンプルデータの作成（実際にはCSVなどから読み込む）
    data = pd.DataFrame({
        'feature1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'target': [15, 25, 35, 45, 55, 65, 75, 85, 95, 105]
    })

    X = data[['feature1', 'feature2']]
    y = data['target']

    # LightGBMモデルの学習
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X, y)

    # モデル保存ディレクトリの確認
    if not os.path.exists('models'):
        os.makedirs('models')

    # モデルの保存
    model_path = 'models/lightgbm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    st.success(f"モデルが '{model_path}' に保存されました！")
    return model

# --- 予測の実行 ---
def make_prediction(model):
    st.header("予測の実行")
    st.write("保存されたモデルを使って予測を行います。")

    st.subheader("予測用データ入力")
    input_feature1 = st.number_input("Feature 1を入力してください", value=55.0)
    input_feature2 = st.number_input("Feature 2を入力してください", value=5.5)

    if st.button("予測する"):
        if model is None:
            st.error("モデルがロードされていません。先にモデルを学習・保存してください。")
            return

        input_df = pd.DataFrame([[input_feature1, input_feature2]], columns=['feature1', 'feature2'])
        prediction = model.predict(input_df)[0]
        st.success(f"予測結果: {prediction:.2f}")

# --- メイン処理 ---
def main():
    st.title("LightGBMモデル予測アプリ with Streamlit")

    # サイドバーメニュー
    st.sidebar.title("メニュー")
    option = st.sidebar.radio(
        "選択してください",
        ("モデル学習・保存", "予測アプリ")
    )

    trained_model = None
    model_path = 'models/lightgbm_model.pkl'

    # 既存モデルのロードを試みる
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            trained_model = pickle.load(f)
        st.sidebar.success("学習済みモデルをロードしました。")

    if option == "モデル学習・保存":
        trained_model = train_and_save_model()
    elif option == "予測アプリ":
        make_prediction(trained_model)

if __name__ == '__main__':
    main()
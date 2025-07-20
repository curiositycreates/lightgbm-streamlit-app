# 1. ライブラリ読み込み
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from lightgbm import early_stopping
from lightgbm import early_stopping, log_evaluation

import os
os.getcwd()
os.chdir("C:\\Users\\haiir\\Documents\\python_DIY\\Python_Learning\\lightgbm-app")


# 2. データ読み込み
df = pd.read_csv("data/sample_data.csv")  # パスは適宜調整
df.head()

# 3. 特徴量と目的変数の分割
X = df.drop(columns=["fim_out"])
y = df["fim_out"]

# 4. 訓練・検証データに分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Optuna 最適化関数の定義
def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)



    gbm = lgb.train(
    params,
    dtrain,
    valid_sets=[dvalid],
    num_boost_round=1000,
    callbacks=[
        early_stopping(50),
        log_evaluation(0)  # ← 必要に応じて 100 などに変更してもOK
    ]
)

    preds = gbm.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    return rmse

# 6. Optuna スタディ実行（軽めに10試行）
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# 7. 結果の確認
print("Best RMSE:", study.best_value)
print("Best params:", study.best_params)

# 8. ベストパラメータでモデル再学習
best_params = study.best_params
best_params["objective"] = "regression"
best_params["metric"] = "rmse"

final_model = lgb.train(
    best_params,
    lgb.Dataset(X, label=y),
    num_boost_round=study.best_trial.number
)

# 9. モデルを保存
import pickle
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

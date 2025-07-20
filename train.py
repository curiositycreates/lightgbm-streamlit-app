import argparse
import os
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import early_stopping, log_evaluation
import pickle

def load_data(data_path):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.replace(r"[^\w]", "_", regex=True)
    X = df[["fim_in", "age", "mmse", "paralysis"]]
    y = df["fim_out"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial, X_train, y_train, X_valid, y_valid):
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
        callbacks=[early_stopping(50), log_evaluation(0)]
    )

    preds = gbm.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    return rmse

def main(args):
    print(f"📂 Loading data from {args.data} ...")
    X_train, X_valid, y_train, y_valid = load_data(args.data)

    print(f"🔍 Running Optuna with {args.trials} trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_valid, y_valid), n_trials=args.trials)

    print("✅ Best RMSE:", study.best_value)
    print("🏆 Best Parameters:", study.best_params)

    print("🚀 Training final model on all data...")
    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip().str.replace(r"[^\w]", "_", regex=True)
    X = df[["fim_in", "age", "mmse", "paralysis"]]
    y = df["fim_out"]

    best_params = study.best_params
    best_params["objective"] = "regression"
    best_params["metric"] = "rmse"

    final_model = lgb.train(
        best_params,
        lgb.Dataset(X, label=y),
        num_boost_round=study.best_trial.number or 100
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(final_model, f)

    print(f"💾 Model saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM + Optuna Training Script")
    parser.add_argument("--data", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--output", type=str, required=True, help="Path to save the model file")
    args = parser.parse_args()
    main(args)

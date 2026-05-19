"""XGBoost 학습 + Optuna 하이퍼파라미터 튜닝"""
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import log_loss

from src.common.logger import get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 20,
) -> xgb.XGBClassifier:

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective":        "binary:logistic",
            "eval_metric":      "logloss",
            "random_state":     42,
            "verbosity":        0,
        }
        model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        return log_loss(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("XGBoost best log_loss=%.4f", study.best_value)

    best = study.best_params
    best.update({"objective": "binary:logistic", "verbosity": 0, "random_state": 42})
    model = xgb.XGBClassifier(**best)
    model.fit(X_train, y_train)
    return model

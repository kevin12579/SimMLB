"""
MLB 학습 데이터 빌드 + 모델 학습 통합 스크립트

피처:
  - 팀 롤링 승률 (최근 20경기, 룩어헤드 없음)
  - 팀 투구 품질 (BBref 전/당년 ERA/FIP/WHIP/K9)
  - 팀 타격 품질 (BBref 전/당년 OPS/OBP/SLG/BA/HR_PA)
  - 파크팩터 (run_factor, hr_factor, is_dome)
"""
import sys
import io
import json
import joblib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import text

import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression

from src.db.session import get_session
from src.common.logger import get_logger

logger = get_logger("build_and_train")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_RAW = Path("data/raw")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
Path("data/training_sets").mkdir(parents=True, exist_ok=True)

# BBref 팀 약자 → MLB StatsAPI 약자
BREF_TO_MLB: dict[str, str] = {
    "CHW": "CWS", "KCR": "KC", "SDP": "SD",
    "SFG": "SF", "TBR": "TB", "WSN": "WSH",
}

FEATURE_COLS = [
    "home_roll_win", "away_roll_win", "roll_win_diff",
    "home_era", "away_era", "era_diff",
    "home_fip", "away_fip", "fip_diff",
    "home_whip", "away_whip",
    "home_k9", "away_k9",
    "home_bb9", "away_bb9",
    "home_ops", "away_ops", "ops_diff",
    "home_obp", "away_obp",
    "home_slg", "away_slg",
    "home_ba", "away_ba",
    "home_hr_pa", "away_hr_pa",
    "park_run_factor", "park_hr_factor", "is_dome",
]


def _norm(abbr: str) -> str:
    return BREF_TO_MLB.get(str(abbr).strip(), str(abbr).strip())


# ──────────────────────────────────────────
# 데이터 로더
# ──────────────────────────────────────────

def load_bref_pitching() -> dict[tuple[str, int], dict]:
    """BBref 투구 CSV → {(mlb_abbr, season): {era, fip, whip, k9, bb9}}"""
    result: dict[tuple[str, int], dict] = {}
    for season in [2023, 2024]:
        path = DATA_RAW / f"bref_pitching_{season}.csv"
        if not path.exists():
            logger.warning("Missing: %s", path)
            continue
        df = pd.read_csv(path)
        # 유효 행: Rk notna + 멀티팀 집계 행(TOT/2TM/3TM) 제외
        _excl = {"TOT", "2TM", "3TM"}
        df = df[pd.notna(df["Rk"]) & (~df["Team"].astype(str).str.strip().isin(_excl))].copy()
        for col in ["IP", "ER", "H", "BB", "SO", "FIP"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Team"] = df["Team"].astype(str).str.strip()

        grp = df.groupby("Team").agg(
            total_ip=("IP", "sum"),
            total_er=("ER", "sum"),
            total_h=("H", "sum"),
            total_bb=("BB", "sum"),
            total_so=("SO", "sum"),
            mean_fip=("FIP", "mean"),
        ).reset_index()

        for _, row in grp.iterrows():
            abbr = _norm(row["Team"])
            ip = row["total_ip"]
            if ip < 10:
                continue
            result[(abbr, season)] = {
                "era":  row["total_er"] * 9.0 / ip,
                "fip":  float(row["mean_fip"]) if not np.isnan(row["mean_fip"]) else 4.20,
                "whip": (row["total_h"] + row["total_bb"]) / ip,
                "k9":   row["total_so"] * 9.0 / ip,
                "bb9":  row["total_bb"] * 9.0 / ip,
            }
    logger.info("Pitching stats loaded: %d (team, season) pairs", len(result))
    return result


def load_bref_batting() -> dict[tuple[str, int], dict]:
    """BBref 타격 CSV → {(mlb_abbr, season): {ops, obp, slg, ba, hr_pa}}"""
    result: dict[tuple[str, int], dict] = {}
    for season in [2023, 2024]:
        path = DATA_RAW / f"bref_batting_{season}.csv"
        if not path.exists():
            logger.warning("Missing: %s", path)
            continue
        df = pd.read_csv(path)
        df = df[pd.notna(df["Rk"]) & (df["Team"].astype(str).str.strip() != "TOT")].copy()
        for col in ["PA", "AB", "H", "2B", "3B", "HR", "BB", "HBP", "SF", "TB"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["Team"] = df["Team"].astype(str).str.strip()

        grp = df.groupby("Team").agg(
            total_pa=("PA", "sum"),
            total_ab=("AB", "sum"),
            total_h=("H", "sum"),
            total_hr=("HR", "sum"),
            total_bb=("BB", "sum"),
            total_hbp=("HBP", "sum"),
            total_sf=("SF", "sum"),
            total_tb=("TB", "sum"),
        ).reset_index()

        for _, row in grp.iterrows():
            abbr = _norm(row["Team"])
            ab = row["total_ab"]
            obp_den = ab + row["total_bb"] + row["total_hbp"] + row["total_sf"]
            if ab < 10 or obp_den < 1:
                continue
            obp = (row["total_h"] + row["total_bb"] + row["total_hbp"]) / obp_den
            slg = row["total_tb"] / ab
            result[(abbr, season)] = {
                "ops":   obp + slg,
                "obp":   obp,
                "slg":   slg,
                "ba":    row["total_h"] / ab,
                "hr_pa": row["total_hr"] / row["total_pa"] if row["total_pa"] > 0 else 0.030,
            }
    logger.info("Batting stats loaded: %d (team, season) pairs", len(result))
    return result


def load_park_factors() -> dict[str, dict]:
    """park_factors.json → {team_abbr: {run_factor, hr_factor, is_dome}}"""
    pj = DATA_RAW / "park_factors.json"
    if pj.exists():
        with open(pj) as f:
            return json.load(f)
    # CSV 폴백
    pc = DATA_RAW / "park_factors.csv"
    df = pd.read_csv(pc)
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        out[str(row["team"])] = {
            "run_factor": float(row["run_factor"]),
            "hr_factor":  float(row["hr_factor"]),
            "is_dome":    bool(str(row.get("is_dome", "False")).lower() == "true"),
        }
    return out


# ──────────────────────────────────────────
# 데이터셋 빌드
# ──────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    pitch_stats  = load_bref_pitching()
    bat_stats    = load_bref_batting()
    park_factors = load_park_factors()

    # 리그 평균 폴백
    LEAGUE_PITCH = {"era": 4.33, "fip": 4.20, "whip": 1.27, "k9": 8.7, "bb9": 3.0}
    LEAGUE_BAT   = {"ops": 0.726, "obp": 0.320, "slg": 0.410, "ba": 0.248, "hr_pa": 0.031}
    LEAGUE_PARK  = {"run_factor": 1.0, "hr_factor": 1.0, "is_dome": False}

    with get_session() as session:
        teams = session.execute(text("SELECT mlbam_team_id, abbreviation FROM teams")).fetchall()
        id_to_abbr = {row[0]: row[1] for row in teams}

        games = session.execute(text("""
            SELECT game_pk, game_date, home_team_id, away_team_id,
                   home_score, away_score, season
            FROM games
            WHERE status = 'Final'
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
            ORDER BY game_date
        """)).fetchall()
    logger.info("Total completed games: %d", len(games))

    # 팀별 (날짜, 승리 여부) 사전 구성 — 롤링 승률 계산용
    team_results: dict[int, list[tuple]] = {}
    for g in games:
        home_won = g.home_score > g.away_score
        for tid, won in [(g.home_team_id, home_won), (g.away_team_id, not home_won)]:
            team_results.setdefault(tid, []).append((g.game_date, won))

    def rolling_win(tid: int, before_date) -> float:
        past = [(d, w) for d, w in team_results.get(tid, []) if d < before_date]
        if not past:
            return 0.500
        recent = past[-20:]
        return sum(w for _, w in recent) / len(recent)

    rows = []
    for i, g in enumerate(games):
        season = g.season
        h_abbr = id_to_abbr.get(g.home_team_id, "")
        a_abbr = id_to_abbr.get(g.away_team_id, "")

        # 2024 게임은 2023 BBref(전년도), 2023 게임은 2023 BBref(당해)
        prev = season - 1 if season >= 2024 else season

        hp  = pitch_stats.get((h_abbr, prev), LEAGUE_PITCH)
        ap  = pitch_stats.get((a_abbr, prev), LEAGUE_PITCH)
        hb  = bat_stats.get((h_abbr, prev),   LEAGUE_BAT)
        ab_ = bat_stats.get((a_abbr, prev),   LEAGUE_BAT)
        pk  = park_factors.get(h_abbr,         LEAGUE_PARK)

        h_roll = rolling_win(g.home_team_id, g.game_date)
        a_roll = rolling_win(g.away_team_id, g.game_date)

        rows.append({
            "game_pk":   g.game_pk,
            "game_date": g.game_date,
            "season":    season,
            # 롤링 승률
            "home_roll_win":  h_roll,
            "away_roll_win":  a_roll,
            "roll_win_diff":  h_roll - a_roll,
            # 투구
            "home_era":  hp["era"],  "away_era":  ap["era"],  "era_diff":  ap["era"]  - hp["era"],
            "home_fip":  hp["fip"],  "away_fip":  ap["fip"],  "fip_diff":  ap["fip"]  - hp["fip"],
            "home_whip": hp["whip"], "away_whip": ap["whip"],
            "home_k9":   hp["k9"],   "away_k9":   ap["k9"],
            "home_bb9":  hp["bb9"],  "away_bb9":  ap["bb9"],
            # 타격
            "home_ops":   hb["ops"],   "away_ops":   ab_["ops"],  "ops_diff": hb["ops"] - ab_["ops"],
            "home_obp":   hb["obp"],   "away_obp":   ab_["obp"],
            "home_slg":   hb["slg"],   "away_slg":   ab_["slg"],
            "home_ba":    hb["ba"],    "away_ba":    ab_["ba"],
            "home_hr_pa": hb["hr_pa"], "away_hr_pa": ab_["hr_pa"],
            # 파크
            "park_run_factor": float(pk.get("run_factor", 1.0)),
            "park_hr_factor":  float(pk.get("hr_factor",  1.0)),
            "is_dome":         int(bool(pk.get("is_dome", False))),
            # 타깃
            "target": int(g.home_score > g.away_score),
        })

        if i % 1000 == 0:
            logger.info("Feature build: %d / %d", i, len(games))

    df = pd.DataFrame(rows)
    logger.info("Dataset shape: %s  home_win_rate=%.3f", df.shape, df["target"].mean())
    return df


# ──────────────────────────────────────────
# 모델 학습
# ──────────────────────────────────────────

def train_lgbm(X_tr, y_tr, X_val, y_val, n_trials: int = 20) -> lgb.LGBMClassifier:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 100),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "binary", "metric": "binary_logloss", "verbose": -1, "random_state": 42,
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        return log_loss(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("LGBM best logloss=%.4f params=%s", study.best_value, study.best_params)

    best = {**study.best_params, "objective": "binary", "verbose": -1, "random_state": 42}
    model = lgb.LGBMClassifier(**best)
    model.fit(X_tr, y_tr)
    return model


def train_xgb(X_tr, y_tr, X_val, y_val, n_trials: int = 20) -> xgb.XGBClassifier:
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
            "objective": "binary:logistic", "eval_metric": "logloss",
            "verbosity": 0, "random_state": 42,
        }
        m = xgb.XGBClassifier(**params, early_stopping_rounds=50)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return log_loss(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("XGB best logloss=%.4f params=%s", study.best_value, study.best_params)

    best = {**study.best_params, "objective": "binary:logistic", "verbosity": 0, "random_state": 42}
    model = xgb.XGBClassifier(**best)
    model.fit(X_tr, y_tr)
    return model


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

if __name__ == "__main__":
    # 1. 데이터셋 빌드
    logger.info("=== 학습 데이터 빌드 시작 ===")
    df = build_dataset()
    df.to_parquet("data/training_sets/training_set.parquet", index=False)
    logger.info("Saved: data/training_sets/training_set.parquet")

    # 2. 시간순 70/15/15 분할
    df = df.sort_values("game_date").reset_index(drop=True)
    n = len(df)
    val_start  = int(n * 0.70)
    test_start = int(n * 0.85)
    train_df = df.iloc[:val_start]
    val_df   = df.iloc[val_start:test_start]
    test_df  = df.iloc[test_start:]

    X_tr, y_tr   = train_df[FEATURE_COLS].values, train_df["target"].values
    X_val, y_val = val_df[FEATURE_COLS].values,   val_df["target"].values
    X_te, y_te   = test_df[FEATURE_COLS].values,  test_df["target"].values
    logger.info("Split — train:%d  val:%d  test:%d", len(train_df), len(val_df), len(test_df))

    # 3. LightGBM
    logger.info("=== LightGBM 학습 (Optuna 20 trials) ===")
    lgbm_model = train_lgbm(X_tr, y_tr, X_val, y_val, n_trials=20)
    joblib.dump(lgbm_model, MODEL_DIR / "lgbm_v1.pkl")
    lgbm_prob_val = lgbm_model.predict_proba(X_val)[:, 1]
    lgbm_prob_te  = lgbm_model.predict_proba(X_te)[:, 1]
    lgbm_ll  = log_loss(y_te, lgbm_prob_te)
    lgbm_auc = roc_auc_score(y_te, lgbm_prob_te)
    logger.info("LightGBM test — logloss=%.4f  AUC=%.4f", lgbm_ll, lgbm_auc)

    # 4. XGBoost
    logger.info("=== XGBoost 학습 (Optuna 20 trials) ===")
    xgb_model = train_xgb(X_tr, y_tr, X_val, y_val, n_trials=20)
    joblib.dump(xgb_model, MODEL_DIR / "xgb_v1.pkl")
    xgb_prob_val = xgb_model.predict_proba(X_val)[:, 1]
    xgb_prob_te  = xgb_model.predict_proba(X_te)[:, 1]
    xgb_ll  = log_loss(y_te, xgb_prob_te)
    xgb_auc = roc_auc_score(y_te, xgb_prob_te)
    logger.info("XGBoost test — logloss=%.4f  AUC=%.4f", xgb_ll, xgb_auc)

    # 5. 앙상블 가중치 (역 logloss)
    inv_l = 1.0 / lgbm_ll
    inv_x = 1.0 / xgb_ll
    w_l   = inv_l / (inv_l + inv_x)
    w_x   = inv_x / (inv_l + inv_x)
    logger.info("Ensemble weights — LGBM:%.3f  XGB:%.3f", w_l, w_x)

    # 6. Isotonic Calibration (val set 기준)
    ensemble_val = w_l * lgbm_prob_val + w_x * xgb_prob_val
    calibrator   = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(ensemble_val, y_val)

    ensemble_te = w_l * lgbm_prob_te + w_x * xgb_prob_te
    cal_prob_te = calibrator.predict(ensemble_te)
    cal_ll  = log_loss(y_te, cal_prob_te)
    cal_auc = roc_auc_score(y_te, cal_prob_te)
    logger.info("Calibrated ensemble — logloss=%.4f  AUC=%.4f", cal_ll, cal_auc)

    # 저장
    joblib.dump(
        {"weights": {"lgbm": w_l, "xgb": w_x}, "calibrator": calibrator, "feature_cols": FEATURE_COLS},
        MODEL_DIR / "calibrator_v1.pkl",
    )

    print("\n========== 학습 완료 ==========")
    print(f"  LightGBM    — logloss={lgbm_ll:.4f}  AUC={lgbm_auc:.4f}")
    print(f"  XGBoost     — logloss={xgb_ll:.4f}  AUC={xgb_auc:.4f}")
    print(f"  앙상블(캘)  — logloss={cal_ll:.4f}  AUC={cal_auc:.4f}")
    print(f"  저장 위치: {MODEL_DIR}/")
    print(f"    lgbm_v1.pkl / xgb_v1.pkl / calibrator_v1.pkl")

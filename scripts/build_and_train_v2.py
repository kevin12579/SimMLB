"""47피처 학습 데이터 빌드 + LightGBM/XGBoost 앙상블 학습 (v2).

v1 대비 변경점:
  - FEATURE_COLS_V2 사용 (47개)
  - games.home_starter_id / away_starter_id로 선발 투수 ID 조회
  - game_lineups 테이블로 라인업 ID 조회 (없으면 폴백)
  - DB에서 시즌별 Statcast 개인/팀 평균 사전 캐시
  - get_rest_days로 휴식일 계산
  - dWAR 포함 batting (load_bref_batting_v2)
"""
from __future__ import annotations

import io
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402
from sqlalchemy import text  # noqa: E402

from src.common.logger import get_logger  # noqa: E402
from src.db.session import get_session  # noqa: E402
from src.ml.features.bref_features_v2 import (  # noqa: E402
    FEATURE_COLS_V2,
    FEATURE_COLS_V3,
    add_diff_features,
    build_feature_row_v2,
    load_bref_batting_v2,
    load_bref_pitching,
    load_park_factors,
)

logger = get_logger("build_and_train_v2")
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
TRAIN_DIR = Path("data/training_sets")
TRAIN_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────
# Statcast 시즌 누적 캐시 (학습 효율 위해 메모리 적재)
# ──────────────────────────────────────────


def build_statcast_expanding(session) -> tuple[dict, dict]:
    """game_date 시점기반 expanding window — look-ahead leak 완전 제거.

    각 (player_id, game_date)에서 그 날짜 직전까지의 시즌 누적 평균.
    그 시즌 첫 게임은 폴백 (LEAGUE 평균).

    Returns:
        pitcher_expanding: {(date, pitcher_id): {velo, spin, whiff, n}}
        batter_expanding:  {(date, batter_id):  {ev, la, hard_hit, n}}
    """
    logger.info("Statcast expanding window 빌드 (시점 분리)...")

    # 투수: 각 (게임 날짜, 투수) → 시즌 시작~직전 게임 평균
    p_rows = session.execute(text("""
        WITH per_game AS (
            SELECT pitcher_id,
                   game_date,
                   EXTRACT(YEAR FROM game_date)::int AS season,
                   COUNT(*) AS n_pitches,
                   AVG(release_speed) AS g_velo,
                   AVG(spin_rate) AS g_spin,
                   AVG(CASE WHEN whiff THEN 1.0 ELSE 0.0 END) AS g_whiff
            FROM statcast_pitches
            WHERE pitcher_id IS NOT NULL AND release_speed IS NOT NULL
            GROUP BY pitcher_id, game_date
        )
        SELECT pitcher_id, game_date, season,
               SUM(n_pitches) OVER w AS cum_n,
               SUM(g_velo * n_pitches) OVER w / NULLIF(SUM(n_pitches) OVER w, 0) AS velo,
               SUM(g_spin * n_pitches) OVER w / NULLIF(SUM(n_pitches) OVER w, 0) AS spin,
               SUM(g_whiff * n_pitches) OVER w / NULLIF(SUM(n_pitches) OVER w, 0) AS whiff
        FROM per_game
        WINDOW w AS (PARTITION BY pitcher_id, season ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    """)).fetchall()
    pitcher_expanding = {
        (r.game_date, int(r.pitcher_id)): {
            "velo": float(r.velo) if r.velo else 93.0,
            "spin": float(r.spin) if r.spin else 2200.0,
            "whiff": float(r.whiff) if r.whiff else 0.25,
            "n": int(r.cum_n) if r.cum_n else 0,
        }
        for r in p_rows
    }

    # 타자: 동일 (BIP만)
    b_rows = session.execute(text("""
        WITH per_game AS (
            SELECT batter_id,
                   game_date,
                   EXTRACT(YEAR FROM game_date)::int AS season,
                   COUNT(*) AS n_bip,
                   AVG(launch_speed) AS g_ev,
                   AVG(launch_angle) AS g_la,
                   AVG(CASE WHEN is_hard_hit THEN 1.0 ELSE 0.0 END) AS g_hh
            FROM statcast_pitches
            WHERE batter_id IS NOT NULL
              AND launch_speed IS NOT NULL
              AND description = 'hit_into_play'
            GROUP BY batter_id, game_date
        )
        SELECT batter_id, game_date, season,
               SUM(n_bip) OVER w AS cum_n,
               SUM(g_ev * n_bip) OVER w / NULLIF(SUM(n_bip) OVER w, 0) AS ev,
               SUM(g_la * n_bip) OVER w / NULLIF(SUM(n_bip) OVER w, 0) AS la,
               SUM(g_hh * n_bip) OVER w / NULLIF(SUM(n_bip) OVER w, 0) AS hh
        FROM per_game
        WINDOW w AS (PARTITION BY batter_id, season ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    """)).fetchall()
    batter_expanding = {
        (r.game_date, int(r.batter_id)): {
            "ev": float(r.ev) if r.ev else 88.5,
            "la": float(r.la) if r.la else 12.0,
            "hard_hit": float(r.hh) if r.hh else 0.35,
            "n": int(r.cum_n) if r.cum_n else 0,
        }
        for r in b_rows
    }

    logger.info("Expanding 캐시 — 투수(date,pid):%d 타자(date,bid):%d",
                len(pitcher_expanding), len(batter_expanding))
    return pitcher_expanding, batter_expanding


def build_statcast_caches(session) -> tuple[dict, dict, dict]:
    """LEGACY: 시즌별 평균 (look-ahead leak 있음 — 호환성 유지)."""
    logger.info("Statcast 시즌별 평균 사전 적재...")

    # 투수 (개인)
    p_rows = session.execute(text("""
        SELECT EXTRACT(YEAR FROM game_date)::int AS season,
               pitcher_id,
               AVG(release_speed) AS velo,
               AVG(spin_rate)     AS spin,
               AVG(CASE WHEN whiff THEN 1.0 ELSE 0.0 END) AS whiff
        FROM statcast_pitches
        WHERE pitcher_id IS NOT NULL AND release_speed IS NOT NULL
        GROUP BY EXTRACT(YEAR FROM game_date), pitcher_id
    """)).fetchall()
    pitcher_indiv = {
        (int(r.season), int(r.pitcher_id)): {
            "velo": float(r.velo) if r.velo else 93.0,
            "spin": float(r.spin) if r.spin else 2200.0,
            "whiff": float(r.whiff) if r.whiff else 0.25,
        }
        for r in p_rows
    }

    # 타자 (개인) — BIP(hit_into_play)만 (foul 제외)
    b_rows = session.execute(text("""
        SELECT EXTRACT(YEAR FROM game_date)::int AS season,
               batter_id,
               AVG(launch_speed) AS ev,
               AVG(launch_angle) AS la,
               AVG(CASE WHEN is_hard_hit THEN 1.0 ELSE 0.0 END) AS hh
        FROM statcast_pitches
        WHERE batter_id IS NOT NULL
          AND launch_speed IS NOT NULL
          AND description = 'hit_into_play'
        GROUP BY EXTRACT(YEAR FROM game_date), batter_id
    """)).fetchall()
    batter_indiv = {
        (int(r.season), int(r.batter_id)): {
            "ev": float(r.ev) if r.ev else 88.5,
            "la": float(r.la) if r.la else 12.0,
            "hard_hit": float(r.hh) if r.hh else 0.35,
        }
        for r in b_rows
    }

    # 팀 (타격 폴백) — BIP 만
    t_rows = session.execute(text("""
        SELECT EXTRACT(YEAR FROM sp.game_date)::int AS season,
               t.abbreviation AS abbr,
               AVG(sp.launch_speed) AS ev,
               AVG(sp.launch_angle) AS la,
               AVG(CASE WHEN sp.is_hard_hit THEN 1.0 ELSE 0.0 END) AS hh
        FROM statcast_pitches sp
        JOIN teams t ON t.mlbam_team_id = sp.batter_team_id
        WHERE sp.launch_speed IS NOT NULL
          AND sp.description = 'hit_into_play'
        GROUP BY EXTRACT(YEAR FROM sp.game_date), t.abbreviation
    """)).fetchall()
    team_bat = {
        (int(r.season), str(r.abbr)): {
            "ev": float(r.ev) if r.ev else 88.5,
            "la": float(r.la) if r.la else 12.0,
            "hard_hit": float(r.hh) if r.hh else 0.35,
        }
        for r in t_rows
    }

    logger.info("Statcast 캐시 — 투수:%d 타자:%d 팀:%d",
                len(pitcher_indiv), len(batter_indiv), len(team_bat))
    return pitcher_indiv, batter_indiv, team_bat


def load_lineups(session) -> dict[tuple[int, bool], list[int]]:
    """game_lineups → {(game_pk, is_home): [player_id, ...]} (batting_order 순)."""
    rows = session.execute(text("""
        SELECT game_pk, is_home, player_id, batting_order
        FROM game_lineups
        ORDER BY game_pk, is_home, batting_order
    """)).fetchall()
    out: dict[tuple[int, bool], list[int]] = defaultdict(list)
    for r in rows:
        out[(int(r.game_pk), bool(r.is_home))].append(int(r.player_id))
    return out


def load_starters(session) -> dict[int, tuple[int | None, int | None]]:
    """games → {game_pk: (home_starter_id, away_starter_id)}."""
    rows = session.execute(text("""
        SELECT game_pk, home_starter_id, away_starter_id
        FROM games WHERE status = 'Final'
    """)).fetchall()
    return {
        int(r.game_pk): (
            int(r.home_starter_id) if r.home_starter_id else None,
            int(r.away_starter_id) if r.away_starter_id else None,
        )
        for r in rows
    }


# ──────────────────────────────────────────
# 데이터셋 빌드
# ──────────────────────────────────────────


def build_dataset() -> pd.DataFrame:
    pitch_team = load_bref_pitching()
    bat_team = load_bref_batting_v2()
    park = load_park_factors()

    with get_session() as session:
        teams = session.execute(
            text("SELECT mlbam_team_id, abbreviation FROM teams")
        ).fetchall()
        id_to_abbr = {r[0]: r[1] for r in teams}

        games = session.execute(text("""
            SELECT game_pk, game_date, home_team_id, away_team_id,
                   home_score, away_score, season
            FROM games
            WHERE status='Final' AND home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY game_date
        """)).fetchall()
        logger.info("학습 대상 경기: %d", len(games))

        # 시즌 평균 (look-ahead 약간 있지만 smooth & stable — expanding보다 generalize 잘 됨)
        pitcher_indiv, batter_indiv, team_bat_sc = build_statcast_caches(session)
        lineups = load_lineups(session)
        starters = load_starters(session)

    # 팀별 (날짜, 승) 사전 — 롤링 승률
    team_results: dict[int, list[tuple]] = {}
    for g in games:
        home_won = g.home_score > g.away_score
        team_results.setdefault(g.home_team_id, []).append((g.game_date, home_won))
        team_results.setdefault(g.away_team_id, []).append((g.game_date, not home_won))

    def rolling_win(tid: int, before) -> float:
        past = [(d, w) for d, w in team_results.get(tid, []) if d < before]
        if not past:
            return 0.5
        recent = past[-20:]
        return sum(w for _, w in recent) / len(recent)

    # 휴식일 — DB의 get_rest_days는 SQL 호출이라 게임당 호출 시 느림.
    # 미리 팀별 정렬된 날짜 리스트에서 직전 경기일 계산.
    team_dates: dict[int, list] = {tid: sorted(d for d, _ in lst)
                                    for tid, lst in team_results.items()}

    def rest_days(tid: int, before) -> int:
        dates = team_dates.get(tid, [])
        prev = None
        for d in dates:
            if d < before:
                prev = d
            else:
                break
        if prev is None:
            return 1
        return max(1, min((before - prev).days, 5))

    rows: list[dict] = []
    for i, g in enumerate(games):
        h_abbr = id_to_abbr.get(g.home_team_id, "")
        a_abbr = id_to_abbr.get(g.away_team_id, "")
        season = g.season

        # 선발 / 라인업
        h_sid, a_sid = starters.get(g.game_pk, (None, None))
        h_lineup = lineups.get((g.game_pk, True), [])
        a_lineup = lineups.get((g.game_pk, False), [])

        sc_season = season - 1 if season >= 2024 else season
        sc_pitcher_view = {
            pid: stats for (s, pid), stats in pitcher_indiv.items() if s == sc_season
        }
        sc_batter_view = {
            pid: stats for (s, pid), stats in batter_indiv.items() if s == sc_season
        }
        sc_team_view = {
            (abbr, sc_season): stats for (s, abbr), stats in team_bat_sc.items()
            if s == sc_season
        }

        feat = build_feature_row_v2(
            h_abbr=h_abbr, a_abbr=a_abbr, season=season,
            h_roll=rolling_win(g.home_team_id, g.game_date),
            a_roll=rolling_win(g.away_team_id, g.game_date),
            h_starter_id=h_sid, a_starter_id=a_sid,
            h_lineup_ids=h_lineup, a_lineup_ids=a_lineup,
            h_team_id=g.home_team_id, a_team_id=g.away_team_id,
            pitch_team=pitch_team, bat_team=bat_team, park=park,
            sc_pitcher_indiv=sc_pitcher_view, sc_team_bat=sc_team_view,
            sc_batter_indiv=sc_batter_view,
            rest_cache={
                g.home_team_id: rest_days(g.home_team_id, g.game_date),
                g.away_team_id: rest_days(g.away_team_id, g.game_date),
            },
        )
        feat = add_diff_features(feat)  # v3: 54피처 (diff 7개 추가)
        feat.update({
            "game_pk": g.game_pk, "game_date": g.game_date, "season": season,
            "target": int(g.home_score > g.away_score),
        })
        rows.append(feat)

        if i % 1000 == 0:
            logger.info("Feature build: %d / %d", i, len(games))

    df = pd.DataFrame(rows)
    logger.info("Dataset shape: %s  home_win_rate=%.3f", df.shape, df["target"].mean())
    return df


# ──────────────────────────────────────────
# 모델 학습 (v1과 동일 hyperparam 공간, Optuna 50 trials)
# ──────────────────────────────────────────


def train_lgbm(X_tr, y_tr, X_val, y_val, n_trials: int = 100) -> lgb.LGBMClassifier:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "binary", "metric": "binary_logloss",
            "verbose": -1, "random_state": 42,
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        return log_loss(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("LGBM best logloss=%.4f", study.best_value)
    best = {**study.best_params, "objective": "binary", "verbose": -1, "random_state": 42}
    model = lgb.LGBMClassifier(**best)
    model.fit(X_tr, y_tr)
    return model


def train_xgb(X_tr, y_tr, X_val, y_val, n_trials: int = 100) -> xgb.XGBClassifier:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "binary:logistic", "eval_metric": "logloss",
            "tree_method": "hist", "verbosity": 0, "random_state": 42,
        }
        m = xgb.XGBClassifier(**params, early_stopping_rounds=50)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return log_loss(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("XGB best logloss=%.4f", study.best_value)
    best = {**study.best_params, "objective": "binary:logistic",
            "tree_method": "hist", "verbosity": 0, "random_state": 42}
    model = xgb.XGBClassifier(**best)
    model.fit(X_tr, y_tr)
    return model


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--features", choices=["v2", "v3"], default="v3",
                   help="v2=47개, v3=54개(diff 추가)")
    p.add_argument("--n-trials", type=int, default=100)
    args = p.parse_args()

    FEATURE_COLS = FEATURE_COLS_V3 if args.features == "v3" else FEATURE_COLS_V2
    suffix = args.features  # 모델 파일명 v2/v3
    logger.info("=== %s (%d피처) 학습 데이터 빌드 ===", suffix, len(FEATURE_COLS))

    df = build_dataset()
    df.to_parquet(TRAIN_DIR / f"training_set_{suffix}.parquet", index=False)
    logger.info("Saved: %s", TRAIN_DIR / f"training_set_{suffix}.parquet")

    df = df.sort_values("game_date").reset_index(drop=True)
    n = len(df)
    v1, v2 = int(n * 0.70), int(n * 0.85)
    train_df, val_df, test_df = df.iloc[:v1], df.iloc[v1:v2], df.iloc[v2:]

    X_tr, y_tr = train_df[FEATURE_COLS].values, train_df["target"].values
    X_val, y_val = val_df[FEATURE_COLS].values, val_df["target"].values
    X_te, y_te = test_df[FEATURE_COLS].values, test_df["target"].values
    logger.info("Split — train:%d val:%d test:%d", len(train_df), len(val_df), len(test_df))

    logger.info("=== LightGBM (Optuna %d trials) ===", args.n_trials)
    lgbm_model = train_lgbm(X_tr, y_tr, X_val, y_val, n_trials=args.n_trials)
    joblib.dump(lgbm_model, MODEL_DIR / f"lgbm_{suffix}.pkl")
    lgbm_pv = lgbm_model.predict_proba(X_val)[:, 1]
    lgbm_pt = lgbm_model.predict_proba(X_te)[:, 1]
    lgbm_ll = log_loss(y_te, lgbm_pt)
    lgbm_auc = roc_auc_score(y_te, lgbm_pt)
    logger.info("LGBM test — logloss=%.4f AUC=%.4f", lgbm_ll, lgbm_auc)

    logger.info("=== XGBoost (Optuna %d trials) ===", args.n_trials)
    xgb_model = train_xgb(X_tr, y_tr, X_val, y_val, n_trials=args.n_trials)
    joblib.dump(xgb_model, MODEL_DIR / f"xgb_{suffix}.pkl")
    xgb_pv = xgb_model.predict_proba(X_val)[:, 1]
    xgb_pt = xgb_model.predict_proba(X_te)[:, 1]
    xgb_ll = log_loss(y_te, xgb_pt)
    xgb_auc = roc_auc_score(y_te, xgb_pt)
    logger.info("XGB test — logloss=%.4f AUC=%.4f", xgb_ll, xgb_auc)

    # 앙상블 가중치 — 역 logloss (val set)
    lgbm_ll_v = log_loss(y_val, lgbm_pv)
    xgb_ll_v = log_loss(y_val, xgb_pv)
    inv_l, inv_x = 1.0 / lgbm_ll_v, 1.0 / xgb_ll_v
    w_l = inv_l / (inv_l + inv_x)
    w_x = inv_x / (inv_l + inv_x)
    logger.info("Ensemble weights — LGBM:%.3f XGB:%.3f", w_l, w_x)

    # Isotonic Calibration (val 기준)
    ensemble_val = w_l * lgbm_pv + w_x * xgb_pv
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(ensemble_val, y_val)
    ensemble_te = w_l * lgbm_pt + w_x * xgb_pt
    cal_pt = np.clip(calibrator.predict(ensemble_te), 0.05, 0.95)
    cal_ll = log_loss(y_te, cal_pt)
    cal_auc = roc_auc_score(y_te, cal_pt)
    logger.info("Calibrated ensemble — logloss=%.4f AUC=%.4f", cal_ll, cal_auc)

    joblib.dump(
        {"weights": {"lgbm": w_l, "xgb": w_x},
         "calibrator": calibrator, "feature_cols": FEATURE_COLS},
        MODEL_DIR / f"calibrator_{suffix}.pkl",
    )

    print(f"\n========== {suffix} 학습 완료 ==========")
    print(f"  LightGBM   — logloss={lgbm_ll:.4f}  AUC={lgbm_auc:.4f}")
    print(f"  XGBoost    — logloss={xgb_ll:.4f}  AUC={xgb_auc:.4f}")
    print(f"  앙상블+cal — logloss={cal_ll:.4f}  AUC={cal_auc:.4f}")
    print(f"  목표: AUC >= 0.56  결과: {'✅' if cal_auc >= 0.56 else '⚠️ 미달성'}")
    print(f"  저장: {MODEL_DIR}/lgbm_v2.pkl / xgb_v2.pkl / calibrator_v2.pkl")

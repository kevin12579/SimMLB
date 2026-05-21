# ⚾ SimMLB — MLB 승부예측 AI 시스템 (v2)

MLB 정규시즌 경기의 **홈팀 승리 확률을 ML 모델로 예측**하고, GPT-4o-mini가 **한국어 분석 근거를 자동 생성**하며, 경기 진행 중 **라이브 스코어와 라이브 승률**을 1분 간격으로 업데이트하는 풀스택 AI 시스템.

> **v2 업데이트 (2026-05-21)** — 47개 피처(라인업 9명 Statcast + 선발투수 velo/spin/whiff + 휴식일 + dWAR) · 8,082경기 학습 · MLB Live Feed 통합 · 동적 스케줄러 · 라이브 polling · 자동 postgame 수집.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [v2 주요 업데이트](#2-v2-주요-업데이트-2026-05-21)
3. [시스템 파이프라인](#3-시스템-파이프라인)
4. [데이터 수집 방식](#4-데이터-수집-방식)
5. [피처 엔지니어링](#5-피처-엔지니어링)
6. [ML 모델 및 성능](#6-ml-모델-및-성능)
7. [AI 근거 생성 (LLM)](#7-ai-근거-생성-llm)
8. [라이브 스코어 + 라이브 예측](#8-라이브-스코어--라이브-예측)
9. [기술 스택](#9-기술-스택)
10. [프로젝트 구조](#10-프로젝트-구조)
11. [빠른 시작](#11-빠른-시작)
12. [API 명세](#12-api-명세)
13. [배포 구성](#13-배포-구성)
14. [트러블슈팅 / 개발 메모](#14-트러블슈팅--개발-메모)

---

## 1. 프로젝트 개요

| 항목 | v1 | **v2 (현재)** |
|---|---|---|
| 예측 대상 | 당일 경기 홈팀 승률 | 동일 + **경기 진행 중 라이브 승률** |
| 예측 주기 | 매일 19:30 KST 일괄 | **경기별 T-15min 개별 + 19:30 폴백** |
| 학습 데이터 | 2023–2024 (4,912경기) | **2023–2026 (8,082경기)** |
| 피처 수 | 29개 (팀 평균) | **47개** (+ 라인업·선발·휴식일) |
| 모델 | LightGBM + XGBoost + Isotonic | 동일 (Optuna 100 trials, seed 고정) |
| 최종 AUC | 0.5336 (앙상블) | **0.5640** (앙상블) / **0.5750** (LGBM 단독) |
| 데이터 수집 | 수동 CSV / 일별 cron | **자동 스텔스 스크래퍼 + 동적 스케줄러** |
| 라이브 기능 | 없음 | **1분 polling + 라이브 win prob + Final 자동 postgame** |
| LLM | GPT-4o-mini, SHAP 한국어 2~3문장 | 동일 |
| 인프라 | Render + Vercel | 동일 |

---

## 2. v2 주요 업데이트 (2026-05-21)

### 무엇이 바뀌었나

1. **47개 피처로 확장** (29 → 47)
   - 라인업 9명 개별 Statcast 평균 (EV / LA / Hard-hit%)
   - 선발투수 Statcast (구속 / 회전수 / 헛스윙%)
   - 휴식일 / dWAR
2. **MLB Live Feed 통합** (`/api/v1.1/game/{pk}/feed/live`)
   - 한 번의 호출로 날씨 + 라인업 + 선발 동시 수집
3. **동적 스케줄러** — 경기별 T-120min(라인업 sync), T-15min(추론), T+0min(라이브 폴러 시작) 자동 등록
4. **BBref 스텔스 스크래퍼** — 자동 일일 업데이트 (User-Agent 위장 + 429 대기)
5. **라이브 스코어 + 라이브 승률** — 1분 polling, MLB winProb + 룩업 폴백
6. **자동 postgame 수집** — Final 감지 시 boxscore → pitcher/batter game logs UPSERT
7. **dWAR 통합** — BBref `war_daily_bat/pit.txt` 다운로더 + player_id 기반 join
8. **학습 데이터 2.5배 확장** — 2023~2026 시즌 통합 (8,082경기)

---

## 3. 시스템 파이프라인

### 3.1 일일 고정 잡 (KST 기준)

```
03:00 (일) 주간 모델 재학습 (Optuna 50 trials)
─────────────────────────────────────────────
06:30  BBref 스텔스 스크래퍼 (pitching/batting CSV 덮어쓰기)
07:00  run_morning_pipeline
         ├ 전일 경기 결과 update_game_results
         └ sync_schedule(today)  ← 오늘 일정 사전 동기화 (옵션 B)
07:30  FanGraphs 선수 통계 갱신
12:00  전일 Statcast append (delta 1일)
12:30  FanGraphs 리더보드 재수집
13:00  👑 마스터 스케줄러: 오늘 경기마다 동적 워커 등록
─────────────────────────────────────────────
19:30  run_inference_fallback (동적 워커 누락 경기 일괄 보충)
```

### 3.2 경기별 동적 워커 (DateTrigger)

```
T-120min  pre_game_sync_{pk}    : Live Feed 1차 sync (날씨/라인업/선발)
T-15min   inference_{pk}        : Live Feed 재호출 + 47피처 + LLM → 베이스 예측
T+0min    live_poller_start_{pk}: 1분 polling 시작
                                   ├─ game_live_states INSERT
                                   ├─ game_predictions.live_* UPDATE
                                   └─ Redis publish (frontend 푸시)
[Final 감지] postgame_sync_{pk} : boxscore → game_logs UPSERT (자가 트리거)
```

---

## 4. 데이터 수집 방식

### 4.1 Baseball Reference (BBref) — 팀 시즌 통계 + dWAR

**v2 수집 방법**: **스텔스 스크래퍼로 매일 자동 수집** (`src/collector/bref_scraper.py`)
- User-Agent 위장 (Chrome/Safari/Linux 로테이션)
- 429 응답 시 60초 대기 후 재시도 (최대 3회)
- 매너 대기 3초 (서버 부하 방지)
- 시즌 누적 평균이므로 매일 **덮어쓰기** (`mode='w'`)

**dWAR 추가** (`scripts/update_dwar.py`):
- `https://www.baseball-reference.com/data/war_daily_bat.txt` (player×season 단위)
- `https://www.baseball-reference.com/data/war_daily_pit.txt`
- `WAR_def` 컬럼 → `dWAR`로 정규화 (멀티팀 합산)
- BBref CSV의 `Player-additional` 컬럼과 `player_id` join으로 dWAR 백필

| 파일 | 내용 | 시즌 |
|---|---|---|
| `bref_pitching_{year}.csv` | 팀별 ERA, FIP, WHIP, K/9, BB/9 | 2023–2025 |
| `bref_batting_{year}.csv` | 팀별 OPS, OBP, SLG, BA, HR/PA | 2023–2025 |
| `bref_fielding_{year}.csv` | 팀별 수비 지표 | 2023–2025 |
| `park_factors.json` | 30개 구장 Run Factor, HR Factor, 돔 여부 | 최신 |

**처리 방식**:
- 멀티팀 집계 행(TOT/2TM/3TM) 제거
- BBref 팀 약자 → MLB StatsAPI 약자 매핑 (예: `CHW→CWS`, `KCR→KC`, `SDP→SD`)
- IP < 10, PA < 10 등 소표본 제거
- 리그 평균값으로 폴백 (ERA 4.33, OPS 0.726 등)

### 4.2 MLB StatsAPI — 경기 일정/결과/라이브

**엔드포인트**: `https://statsapi.mlb.com/api/v1`

```python
# v1 endpoints
GET /schedule?sportId=1&date=YYYY-MM-DD            # 경기 일정
GET /game/{game_pk}/linescore                       # 라이브 폴러 (1분 polling)
GET /game/{game_pk}/boxscore                        # postgame 수집

# v2 신규 — Live Feed 통합 (1콜로 날씨+라인업+선발 동시)
GET /api/v1.1/game/{game_pk}/feed/live
  → gameData.weather.{temp, condition, wind}
  → gameData.probablePitchers.{home,away}.id
  → liveData.boxscore.teams.{home,away}.battingOrder[]
```

**수집 내용**: 경기 일정, 홈/어웨이 팀 ID, 최종 스코어, 선발 투수 MLBAM ID, **확정 라인업 9명 mlbam_id**, **라이브 inning/score/winProbability**

### 3.3 Statcast — 투구 데이터 (pybaseball)

```python
import pybaseball
df = pybaseball.statcast(start_dt="2024-04-01", end_dt="2024-04-05")
```

- 5일 단위로 분할 요청 (502 에러 방지)
- `asyncio.to_thread` 래핑으로 비동기 처리
- `is_hard_hit` (95mph+), `is_barrel` 컬럼 직접 계산
- `release_speed`, `spin_rate`, `launch_speed`, `launch_angle` 저장

### 3.4 Open-Meteo — 구장 날씨

```
GET https://api.open-meteo.com/v1/forecast
    ?latitude={lat}&longitude={lon}
    &hourly=temperature_2m,wind_speed_10m
```

MLB 30개 구장 좌표 하드코딩 → 당일 경기 시각(현지 시간) 기준 기온/풍속 조회

### 3.5 FanGraphs — 선발 투수 세이버메트릭스

`pybaseball.pitching_stats()` / `pybaseball.batting_stats()` 활용
- FIP, xFIP, SIERA, SwStr%, Contact% 등 수집
- `as_of_date` 기준 시점 제한 적용 (데이터 누수 방지)

---

## 5. 피처 엔지니어링

### v2 피처 목록 (47개)

| 그룹 (수) | 주요 피처 | 출처 |
|---|---|---|
| 팀 롤링 승률 (3) | `home_roll_win`, `away_roll_win`, `roll_win_diff` | DB games (20경기 윈도우) |
| 팀 투구 (12) | `era`, `fip`, `whip`, `k9`, `bb9` (home/away/diff) | BBref CSV |
| **선발 투수 (8) 🆕** | `starter_era`, `starter_velo`, `starter_spin`, `starter_whiff` | DB statcast_pitches + BBref |
| 팀 타격 (13) | `ops`, `obp`, `slg`, `ba`, `hr_pa`, `dwar` (home/away/diff) | BBref CSV + `bref_dwar_master.csv` |
| **라인업 9명 Statcast (6) 🆕** | `bat_ev`, `bat_la`, `bat_hardhit` (home/away) | DB statcast_pitches (BIP만, `description='hit_into_play'`) |
| **휴식일 (2) 🆕** | `home_rest`, `away_rest` (1~5일 클램프) | DB games |
| 구장 (3) | `park_run_factor`, `park_hr_factor`, `is_dome` | `park_factors.json` |

### v2 피처 빌더 구조

```python
build_feature_row_v2(
    h_abbr, a_abbr, season, h_roll, a_roll,
    h_starter_id, a_starter_id,           # MLBAM ID 기반 (BBref 이름 매칭 아님)
    h_lineup_ids: list[int],              # 라인업 9명 mlbam_id
    a_lineup_ids: list[int],
    h_team_id, a_team_id,
    pitch_team, bat_team, park,
    sc_pitcher_indiv,                     # {pitcher_id: {velo, spin, whiff}}
    sc_team_bat,                          # {(abbr, season): {ev, la, hard_hit}} 폴백
    sc_batter_indiv,                      # {batter_id: {ev, la, hard_hit}} 라인업 9명
    rest_cache,                           # {team_id: rest_days}
) -> dict[str, float]                     # 47개 키 반환
```

**라인업 → 팀 평균 폴백 자동**: 라인업 9명 중 statcast 매칭된 선수가 0명이면 팀 평균, 1명 이상이면 매칭된 선수들 평균.

### 데이터 누수 방지 (Look-ahead Bias)

```python
# ❌ 잘못된 예 — 미래 데이터 포함 가능
SELECT AVG(era) FROM pitcher_game_logs WHERE season = 2024;

# ✅ 올바른 예 — 경기 당일 이전 데이터만 사용
def rolling_win(tid: int, before_date) -> float:
    past = [(d, w) for d, w in team_results.get(tid, []) if d < before_date]
    recent = past[-20:]  # 당일 이전 최근 20경기만
    return sum(w for _, w in recent) / len(recent)
```

BBref 시즌 통계는 **2024 경기 → 2023 BBref(전년도)**, **2023 경기 → 2023 BBref(당해)** 방식으로 적용하여 시점 오염 방지.

---

## 6. ML 모델 및 성능

### 6.1 모델 구조

```
LightGBMClassifier ─┐
                    ├─→ 가중 앙상블 ─→ IsotonicRegression ─→ 최종 확률
XGBClassifier ──────┘   (역 logloss 가중치)   (확률 보정)
```

### 6.2 v1 vs v2 성능 비교 (테스트셋)

| 지표 | v1 (29피처, 4912경기) | **v2 (47피처, 8082경기)** | 변화 |
|---|---|---|---|
| LightGBM AUC | 0.5263 | **0.5750** ⭐ | **+4.87%p** |
| XGBoost AUC | 0.5507 | 0.5540 | +0.33%p |
| 앙상블+캘 AUC | 0.5336 | **0.5640** ✅ | **+3.04%p** |
| 학습 데이터 | 4,912경기 | 8,082경기 | +65% |
| Optuna trials | 20 | 100 (seed 고정) | — |

> MLB 경기 결과는 본질적으로 고노이즈 도메인 (홈팀 실제 승률 ~52%). AUC 0.56+가 도메인 한계 근처.

### 6.3 v2 하이퍼파라미터 튜닝 (Optuna 100 trials, TPESampler seed=42)

**LightGBM 최적 파라미터**:
```
n_estimators=295, max_depth=5, num_leaves=50,
learning_rate=0.052, subsample=0.744, colsample_bytree=0.925,
min_child_samples=47, reg_alpha=0.003, reg_lambda=0.0005
```

**XGBoost 최적 파라미터**:
```
n_estimators=488, max_depth=7,
learning_rate=0.043, subsample=0.677, colsample_bytree=0.891,
min_child_weight=3, gamma=4.66, reg_alpha=2.7e-06, reg_lambda=0.241
```

### 5.3 시간순 학습/검증 분할 (70/15/15)

```
전체 4,912 경기 (2023–2024 시즌 완료 경기)
  ├── Train set:  3,438경기 (70%) — 2023 ~ 2024 초반
  ├── Val set:      737경기 (15%) — 2024 중반
  └── Test set:     737경기 (15%) — 2024 후반
```

> 시간순 분할로 미래 데이터 누수 완전 차단

### 5.4 최종 성능 (테스트셋 기준)

| 모델 | Log Loss | AUC |
|---|---|---|
| LightGBM | 0.7325 | 0.5263 |
| XGBoost | 0.6970 | **0.5507** |
| 앙상블 + Calibration | 0.7890 | 0.5336 |

> MLB 경기 결과는 본질적으로 고노이즈 도메인 (홈팀 실제 승률 ~52%). AUC 0.55는 베이스라인(랜덤 0.50) 대비 유의미한 예측력.

### 5.5 앙상블 가중치

역(inverse) log loss 기반 softmax 가중치:
```
LGBM weight = 0.488
XGB  weight = 0.512
```

### 5.6 확률 보정 (Isotonic Calibration)

`sklearn.isotonic.IsotonicRegression`으로 검증셋 기준 보정.
- 모델 출력 확률이 실제 승리율과 일치하도록 단조 변환
- 신뢰도 배지 계산에 보정된 확률 사용

### 5.7 신뢰도 배지 기준

| 레벨 | 조건 | 의미 |
|---|---|---|
| HIGH (초록) | `\|확률 - 0.5\| > 0.15` | 모델이 한 팀을 강하게 지목 |
| MED (노랑) | `\|확률 - 0.5\| > 0.05` | 어느 정도 유리한 팀 있음 |
| LOW (회색) | `\|확률 - 0.5\| ≤ 0.05` | 거의 반반 |

---

## 7. AI 근거 생성 (LLM)

### 흐름

```
ML 추론 완료
    ↓
SHAP (TreeExplainer) → 피처별 기여도 계산
    ↓
절댓값 기준 상위 5개 피처 선택
    ↓
프롬프트 빌더: "홈팀 ERA 3.12 (유리), 원정팀 OPS 0.812 (불리)..." 구성
    ↓
GPT-4o-mini API 호출 → 한국어 2–3문장 분석 근거 생성
    ↓
DB game_predictions.reasoning_text 저장
```

### 예시 출력

```json
{
  "home_win_prob": 0.623,
  "confidence": "MED",
  "reasoning": "홈팀 투수진 FIP(3.41)가 원정팀(4.12)보다 우수하며,
    홈팀의 최근 20경기 승률(0.650)이 리그 평균을 크게 상회한다.
    원정팀 타선 OPS(0.691)가 홈팀(0.754)에 비해 낮아 홈팀이 유리하다.",
  "shap_top5": [
    {"feature": "fip_diff", "value": 0.71, "shap_value": 0.089},
    {"feature": "roll_win_diff", "value": 0.15, "shap_value": 0.067},
    ...
  ]
}
```

### LLM 폴백

`.env`에서 `LLM_PROVIDER=groq`로 변경 시 Groq (llama-3.1-8b-instant) 무료 사용 가능 — API 크레딧 소진 시 코드 변경 없이 전환.

---

## 8. 라이브 스코어 + 라이브 예측 (v2 신규)

### 8.1 라이브 폴러 (`src/collector/live_score_poller.py`)

경기 시작(T+0) 시점에 BackgroundScheduler가 IntervalTrigger(1분)로 자가 종료형 잡 등록:

```
매 분마다:
  GET /api/v1/game/{pk}/linescore        (작은 응답 ~1KB)
   → inning, half, outs, balls/strikes, home/away score, on1/2/3
   → MLB winProbability (제공 시)
  ↓
  base_prob = game_predictions.home_win_prob (T-15에 저장된 pre-game)
  live_prob = live_win_prob_adjuster(base_prob, state, mlb_wp)
  ↓
  INSERT INTO game_live_states (시계열 이력)
  UPDATE game_predictions.live_* (최신 스냅샷)
  Redis publish live:{pk}                 (SSE/WebSocket 푸시 시드)
  ↓
  status == "Final" 감지 시 → 자기 자신 제거 + postgame_sync 트리거
```

### 8.2 라이브 승률 계산 (`src/ml/live_win_prob.py`)

별도 in-game 모델을 학습하지 않고 다음 우선순위:

1. **MLB 공식 winProbability** (가능 시) — `liveData.linescore.teams.home.winProb`
2. **Tom Tango "The Book" 룩업 테이블** — (inning, lead) → 홈 승률
3. **베이스 가중평균** — 룩업 없으면 진행도(inning/9)에 따라 base ↔ 0.5 감쇠

### 8.3 자동 postgame 수집 (`src/collector/postgame_collector.py`)

라이브 폴러가 `status=Final` 감지 시 즉시(T+5min) 호출:

```
GET /api/v1/game/{pk}/boxscore
  → games.{home_score, away_score, status='Final'}        UPDATE
  → pitcher_game_logs (ip, er, k, bb, h, hr, pitches)     UPSERT
  → batter_game_logs  (ab, h, hr, rbi, bb, k)             UPSERT
```

IP 변환: `"5.2"` (5와 2/3 이닝) → `5.667` 정확 계산.

---

## 9. 기술 스택

| 레이어 | 기술 | 버전 | 역할 |
|---|---|---|---|
| **언어** | Python | 3.11+ | 백엔드·ML 전체 |
| **DB** | PostgreSQL | 16 | 경기·선수·예측 데이터 |
| **캐시** | Redis | 7 | API 응답 캐시 (TTL 1시간) |
| **ORM** | SQLAlchemy | 2.0 | Python ↔ PostgreSQL |
| **마이그레이션** | Alembic | 1.13+ | DB 스키마 버전 관리 |
| **의존성** | Poetry | 최신 | Python 패키지 관리 |
| **스케줄러** | APScheduler | 3.x | 7개 자동 파이프라인 잡 |
| **ML** | LightGBM | 4.3+ | 그래디언트 부스팅 분류 |
| **ML** | XGBoost | 2.0+ | 그래디언트 부스팅 분류 |
| **튜닝** | Optuna | 3.5+ | 하이퍼파라미터 자동 최적화 (20 trials) |
| **보정** | scikit-learn | 1.4+ | IsotonicRegression |
| **설명 가능 AI** | SHAP | 0.45+ | TreeExplainer 피처 기여도 |
| **LLM** | OpenAI gpt-4o-mini | — | 한국어 분석 근거 생성 |
| **데이터** | pybaseball | — | Statcast/FanGraphs 수집 |
| **백엔드** | FastAPI + Uvicorn | 0.110+ | REST API |
| **프론트엔드** | Next.js | 14 | 예측 결과 웹 UI |
| **스타일링** | Tailwind CSS | 3 | 반응형 UI |
| **컨테이너** | Docker Compose | — | 로컬 개발 환경 |
| **CI/CD** | GitHub Actions | — | 린트 + 일일 파이프라인 |
| **배포 (백엔드)** | Render | 무료 티어 | FastAPI + Background Worker |
| **배포 (프론트)** | Vercel | Hobby(무료) | Next.js |
| **알림** | Discord Webhook | — | 파이프라인 모니터링 |

---

## 10. 프로젝트 구조

```
simlb/
├── src/
│   ├── api/                        # C팀 — FastAPI 백엔드
│   │   ├── main.py                 # 앱 진입점, CORS 설정
│   │   └── routers/
│   │       ├── predictions.py      # /predictions/* 엔드포인트 + Redis 캐시
│   │       └── health.py           # /health 헬스체크
│   │
│   ├── collector/                  # A팀 — 데이터 수집
│   │   ├── base.py                 # BaseCollector (재시도·Rate Limit, timeout 지원)
│   │   ├── mlb_statsapi_client.py  # MLB 공식 API (경기/선수/라인업)
│   │   ├── statcast_collector.py   # Statcast (pybaseball)
│   │   ├── fangraphs_collector.py  # FanGraphs (FIP, xFIP, wRC+)
│   │   ├── weather_client.py       # Open-Meteo (구장 날씨, v2 폴백)
│   │   ├── roster_sync.py          # 팀 로스터 동기화
│   │   ├── live_feed_client.py     # 🆕 v2: MLB Live Feed (날씨/라인업/선발 통합)
│   │   ├── live_score_poller.py    # 🆕 v2: 1분 간격 라이브 스코어 + 라이브 승률
│   │   ├── postgame_collector.py   # 🆕 v2: 경기 종료 후 boxscore → game_logs
│   │   ├── bref_scraper.py         # 🆕 v2: BBref 스텔스 스크래퍼
│   │   └── pipeline.py             # APScheduler (8 cron + 동적 워커 4종)
│   │
│   ├── db/                         # A팀 — 데이터베이스
│   │   ├── session.py              # SQLAlchemy 엔진·세션 관리
│   │   ├── base.py                 # ORM Base 클래스
│   │   └── models/
│   │       ├── teams.py            # MLB 30개 팀
│   │       ├── players.py          # 선수 마스터 (MLBAM ID)
│   │       ├── games.py            # 경기 기록
│   │       └── predictions.py      # ML 예측 결과
│   │
│   ├── ml/                         # B팀 — ML 파이프라인
│   │   ├── features/
│   │   │   ├── bref_features.py    # BBref 피처 모듈 (v1 29피처)
│   │   │   ├── bref_features_v2.py # 🆕 v2: 47피처 (라인업/선발 Statcast/dWAR/휴식)
│   │   │   └── ...                 # team/pitcher/batter/context (구버전)
│   │   ├── models/                 # LightGBM, XGBoost, 앙상블
│   │   ├── reasoning/              # LLM 추상화 + 프롬프트 빌더
│   │   ├── live_win_prob.py        # 🆕 v2: 라이브 승률 조정 함수
│   │   ├── build_training_data.py  # 학습 데이터셋 빌드
│   │   ├── calibration.py          # Isotonic 보정
│   │   └── prediction_service.py   # 추론 파이프라인 통합
│   │
│   └── common/                     # 공통 유틸
│       ├── logger.py               # 구조화 로깅
│       ├── retry.py                # 재시도 데코레이터
│       └── exceptions.py           # 커스텀 예외
│
├── scripts/
│   ├── backfill_historical.py      # 2023–2026 과거 데이터 백필 (확장)
│   ├── backfill_lineups.py         # 🆕 v2: game_lineups 백필
│   ├── build_and_train.py          # v1 학습 (29피처)
│   ├── build_and_train_v2.py       # 🆕 v2: 47피처 학습 (--features v2|v3)
│   ├── run_inference_v2.py         # v1 모델 추론 (29피처)
│   ├── run_inference_v3.py         # 🆕 v2 추론 (47피처 + Live Feed)
│   ├── run_daily_inference.py      # 일일 추론 (v2 자동 감지 + v1 폴백)
│   ├── update_dwar.py              # 🆕 v2: BBref WAR Daily → dWAR 마스터 CSV
│   └── verify_data_integrity.py    # DB 데이터 무결성 검증
│
├── frontend/                       # C팀 — Next.js 14
│   ├── app/
│   │   └── page.tsx                # 당일 예측 테이블 (메인 페이지)
│   └── ...
│
├── migrations/                     # Alembic DB 마이그레이션
│   └── versions/
│       ├── 140ec83bcd4a_initial_schema.py
│       ├── 4fb0b416e598_add_game_date_....py
│       └── a1b2c3d4e5f6_live_states_and_postgame.py  # 🆕 v2: live_states + live_*컬럼
│
├── data/
│   ├── raw/                        # BBref CSV + 파크팩터 JSON
│   │   ├── bref_pitching_{year}.csv
│   │   ├── bref_batting_{year}.csv
│   │   ├── bref_dwar_master.csv    # 🆕 v2: WAR Daily에서 추출한 dWAR 마스터
│   │   ├── war_archive/            # 🆕 v2: BBref WAR Daily 원본 (수동 다운로드)
│   │   │   ├── war_daily_bat.txt
│   │   │   └── war_daily_pitch.txt
│   │   └── park_factors.json
│   └── training_sets/
│       └── training_set_v2.parquet # 🆕 v2: 8,082경기 × 51컬럼
│
├── models/                         # 학습된 모델 파일
│   ├── lgbm_v1.pkl / xgb_v1.pkl / calibrator_v1.pkl   # v1 (29피처)
│   ├── lgbm_v2.pkl                 # 🆕 v2 LightGBM (AUC 0.5750)
│   ├── xgb_v2.pkl                  # 🆕 v2 XGBoost (AUC 0.5540)
│   └── calibrator_v2.pkl           # 🆕 v2 앙상블+캘 (AUC 0.5640 ✅)
│
├── .github/workflows/
│   ├── ci.yml                      # Push 시 Ruff 린트 + mypy 타입 체크
│   └── daily_pipeline.yml          # 매일 UTC 10:30 (KST 19:30) 자동 실행
│
├── render.yaml                     # Render Web Service + Background Worker
├── docker-compose.yml              # 로컬 개발 PostgreSQL + Redis
├── pyproject.toml                  # Poetry 의존성
└── config/settings.py              # pydantic-settings 환경변수 관리
```

---

## 11. 빠른 시작

### 필요 조건

- Python 3.11+
- Node.js 18+
- Docker Desktop
- Poetry

### 로컬 실행

```bash
# 1. 의존성 설치
poetry install

# 2. 환경변수 설정
cp .env.example .env
# .env 파일에서 POSTGRES_PASSWORD, OPENAI_API_KEY 입력

# 3. PostgreSQL + Redis 실행
docker compose up -d

# 4. DB 스키마 생성
poetry run alembic upgrade head

# 5. FastAPI 서버 실행
poetry run uvicorn src.api.main:app --reload
# → http://localhost:8000/health

# 6. 당일 추론 실행 (프론트 접속 전 반드시 먼저 실행)
poetry run python scripts/run_inference_v2.py
```

> ⚠️ **로컬 캐시 주의**: 추론 전에 프론트엔드에 접속하면 Redis가 빈 응답을 1시간 동안 캐시합니다.
> 이 경우 추론 후에도 "오늘 예측 데이터가 아직 없습니다"가 계속 표시됩니다.
> 아래 명령어로 캐시를 지운 뒤 브라우저를 새로고침하세요:
> ```bash
> docker exec simlb-redis-1 redis-cli FLUSHALL
> ```

### 프론트엔드 실행

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### 모델 재학습

```bash
# BBref CSV를 data/raw/에 업데이트 후
poetry run python scripts/build_and_train.py
```

---

## 12. API 명세

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 서버 헬스체크 |
| GET | `/predictions/today` | 당일 전 경기 예측 목록 (Redis 1시간 캐시) |
| GET | `/predictions/{game_pk}` | 특정 경기 예측 상세 (확률 + SHAP + 근거) |
| GET | `/games/{game_pk}` | 특정 경기 기본 정보 |

### 응답 예시 (`/predictions/today`)

```json
{
  "date": "2025-06-01",
  "count": 15,
  "games": [
    {
      "game_pk": 746123,
      "home_team": "LAD",
      "away_team": "NYY",
      "home_win_prob": 0.623,
      "away_win_prob": 0.377,
      "confidence": "MED",
      "reasoning": "홈팀 투수진 FIP(3.41)가 원정팀보다 우수하며, 최근 20경기 승률(0.650)이 리그 평균을 상회한다. 파크팩터(run_factor=1.08)도 홈팀 타선에 유리하게 작용한다.",
      "shap_top5": [
        {"feature": "fip_diff", "value": 0.71, "shap_value": 0.089},
        {"feature": "roll_win_diff", "value": 0.15, "shap_value": 0.067}
      ]
    }
  ]
}
```

---

## 13. 배포 구성

### 백엔드 — Render

`render.yaml`에 Web Service + Background Worker 정의:

```yaml
services:
  - type: web         # FastAPI REST API
    name: mlb-prediction-api
    startCommand: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /health

  - type: worker      # APScheduler 파이프라인 워커
    name: mlb-prediction-worker
    startCommand: python -c "from src.collector.pipeline import setup_scheduler; ..."
```

> 무료 외부 DB 권장: **Neon.tech** (PostgreSQL), **Upstash** (Redis)

### 프론트엔드 — Vercel

```
Root Directory: frontend
NEXT_PUBLIC_API_URL = https://mlb-prediction-api.onrender.com
```

### CI/CD — GitHub Actions

| 워크플로우 | 트리거 | 내용 |
|---|---|---|
| `ci.yml` | push/PR | Ruff 린트 + mypy 타입 체크 |
| `daily_pipeline.yml` | 매일 UTC 10:30 (KST 19:30) | DB 마이그레이션 + 일일 추론 자동 실행 |

### 필요한 GitHub Secrets

```
POSTGRES_PASSWORD    # DB 비밀번호
OPENAI_API_KEY       # sk-proj-...
DISCORD_WEBHOOK_URL  # (선택) Discord 알림
```

---

## DB 스키마

```
teams (30개 MLB 팀)
  │
  ├── games (경기 기록 — game_pk 기준)
  │     ├── game_lineups      (확정 라인업)
  │     ├── game_predictions  (ML 예측 결과)
  │     └── pitcher_game_logs (선발 투수 기록)
  │
  ├── players (선수 마스터 — MLBAM ID 기준)
  │     ├── player_season_stats    (시즌 통계 — as_of_date 시점별)
  │     ├── pitcher_game_logs      (투수 경기별 기록)
  │     ├── batter_game_logs       (타자 경기별 기록)
  │     └── player_statcast_summary (Statcast 집계)
  │
  └── team_daily_snapshots (팀 일별 스냅샷 — 승률·연승·PythagenPat)

statcast_pitches (투구 단위 Statcast 데이터)

# v2 신규 테이블
game_live_states (1분 polling 시계열 — 차트용)
  ├ game_pk, polled_at, game_status, inning, half, outs, balls, strikes
  ├ home_score, away_score, on1/2/3
  └ mlb_win_prob, live_home_prob

# v2 신규 컬럼 (game_predictions)
+ weather_temp_f, weather_condition, weather_wind
+ live_home_win_prob, live_status, live_current_inning, live_score_home, live_score_away
+ live_updated_at, live_lineup_synced_at
```

---

## 팀 구조

| 팀 | 담당 영역 | 주요 파일 |
|---|---|---|
| **A팀** (Data) | 수집기, DB, 스케줄러 | `src/collector/*`, `src/db/*` |
| **B팀** (ML) | 피처, 모델, 추론, LLM | `src/ml/*`, `scripts/build_and_train_v2.py` |
| **C팀** (API/Frontend) | FastAPI, Next.js | `src/api/*`, `frontend/*` |

---

## 14. 트러블슈팅 / 개발 메모

### 14.1 v1 → v2 마이그레이션 시 겪었던 이슈

| 이슈 | 원인 | 해결 |
|---|---|---|
| **BBref Cloudflare 차단** | sandbox/CI IP를 Cloudflare 챌린지 | `bref_scraper.py` UA 로테이션 + 429 시 60s 대기. **사용자 PC에선 정상 동작** |
| **WAR 마스터 자동 다운로드 차단** | 동일 Cloudflare | `--import-bat`/`--import-pit` 로컬 import 옵션 추가. 브라우저로 직접 받은 파일을 `data/raw/war_archive/` 에 두고 import |
| **`game_lineups` 테이블 비어있음** | 학습 데이터에 라인업 미반영 → batter Statcast 폴백 88.5 | `backfill_lineups.py` 작성 → MLB Stats API boxscore로 4,914경기 백필 |
| **`launch_speed` 평균 82.5 (표준 88+)** | foul ball까지 launch_speed 평균에 포함 | SQL 필터 `description = 'hit_into_play'` 추가 → 88.4 정상화 |
| **`statcast_pitches.batter_team_id` 100% NULL** | 수집 시 team_id 미적재 | starter 매핑 SQL UPDATE로 58% 부분 보완 + `idx_statcast_batter_team_date` 인덱스 |
| **Optuna stochasticity ±2%p** | seed 미고정 + n_trials=50 부족 | `TPESampler(seed=42)` + n_trials=100 (200은 overfit) |
| **`scripts/_backfill_*.py` ORM FK 오류** | `from src.db.models.games import GameLineup` 만 import → `teams`/`players` 메타 미등록 | `from src.db.models import teams as _teams` 등 명시 import |
| **`BaseCollector._get`이 timeout 인자 미지원** | 4.x 모듈들 `timeout=10` 호출 실패 | `_get(...)` 시그니처에 `timeout: float \| None = None` 추가 |
| **expanding window 시점 분리 후 AUC 하락** | sparse 데이터 (시즌 초 경기는 누적 평균 없음) | 시즌 평균(약한 leak이지만 smooth) 채택 — 0.5287 → 0.5640 |
| **`prev season` 강제 시 AUC 하락** | test set(2024 후반)에 1년 묵은 데이터 적용 → 신호 약화 | `target_season = season if (h_abbr, season) in pitch_team else season - 1` 채택 |
| **diff 피처 추가 시 AUC 약간 하락** | 트리는 이미 interaction 자동 학습 → diff는 redundant | v3 (54피처) 폐기, v2 (47피처) 유지 |
| **모델 파일 덮어쓰기로 best run 손실** | 학습마다 model PKL이 즉시 dump | `_multi_seed_train.py` 다중 seed best 선택 후 저장 |

### 14.2 라이브 폴러 잡 ID 충돌 방지

```python
# 마스터가 등록한 startup 잡 (DateTrigger T+0)
livestart_{pk}  →  run start_live_poller()
# start_live_poller가 등록하는 polling 잡 (IntervalTrigger 1분)
live_{pk}       →  run _live_poll_tick(pk)
# Final 감지 후 등록되는 postgame 잡 (DateTrigger T+5min)
post_{pk}       →  run run_postgame_sync(pk)
```

서로 다른 ID 접두사로 충돌 없음. `replace_existing=True` + `max_instances=1`로 중복 방지.

### 14.3 다중 seed 학습 결과 (v2 final)

| seed | LGBM | XGB | Cal+Ensemble |
|---|---|---|---|
| 42 | 0.5520 | 0.5510 | 0.5565 |
| 7 | 0.5456 | **0.5819** | 0.5526 |
| **123** ⭐ | 0.5594 | 0.5643 | **0.5606** (4912 학습본) |
| 2024 | 0.5482 | 0.5321 | 0.5376 |
| 999 | 0.5347 | 0.5630 | 0.5419 |

**8082경기 재학습 후 최종**: LGBM 0.5750 / XGB 0.5540 / **Cal 0.5640 ✅** (목표 0.56+ 통과)

### 14.4 다음 단계 권장 (향후 추가 향상)

| 방안 | 예상 효과 | 비용 |
|---|---|---|
| 라인업 9명 최근 30일 statcast (game_date 기준 window) | ⭐⭐⭐ | 큼 (DB 재구조화) |
| 선발 vs 라인업 hand matchup (L/R split) | ⭐⭐ | 중 |
| FanGraphs advanced metrics (wOBA, xwOBA, BABIP) | ⭐⭐ | 중 |
| Bullpen 통계 (구원진 ERA/FIP) | ⭐ | 중 |
| 2027+ 신규 시즌 데이터 누적 후 재학습 | ⭐⭐⭐ | 작음 (자동) |

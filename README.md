# ⚾ SimMLB — MLB 승부예측 AI 시스템 (v2.1)

MLB 정규시즌 경기의 **홈팀 승리 확률을 ML 모델로 예측**하고, GPT-4o-mini가 **한국어 분석 근거를 자동 생성**하며, 경기 진행 중 **라이브 스코어와 라이브 승률**을 1분 간격으로 업데이트하는 풀스택 AI 시스템.

> **v2 업데이트 (2026-05-21)** — 47개 피처(라인업 9명 Statcast + 선발투수 velo/spin/whiff + 휴식일 + dWAR) · 8,082경기 학습 · MLB Live Feed 통합 · 동적 스케줄러 · 라이브 polling · 자동 postgame 수집.
>
> **v2.1 패치 (2026-05-27)** — KST/ET 날짜 버그 다수 수정 · `/predictions/history` 신규 · 경기일정 ✓/✗ 아이콘 · 선수명 TBD 일괄 수정 · archive on-the-fly 채점 · park_factor 수집기 추가.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [v2 주요 업데이트](#2-v2-주요-업데이트-2026-05-21)
3. [v2.1 패치노트](#3-v21-패치노트-2026-05-27)
4. [시스템 파이프라인](#4-시스템-파이프라인)
5. [데이터 수집 방식](#5-데이터-수집-방식)
6. [피처 엔지니어링](#6-피처-엔지니어링)
7. [ML 모델 및 성능](#7-ml-모델-및-성능)
8. [AI 근거 생성 (LLM)](#8-ai-근거-생성-llm)
9. [라이브 스코어 + 라이브 예측](#9-라이브-스코어--라이브-예측)
10. [기술 스택](#10-기술-스택)
11. [프로젝트 구조](#11-프로젝트-구조)
12. [빠른 시작](#12-빠른-시작)
13. [API 명세](#13-api-명세)
14. [배포 구성](#14-배포-구성)
15. [트러블슈팅 / 개발 메모](#15-트러블슈팅--개발-메모)

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

## 3. v2.1 패치노트 (2026-05-27)

### 버그 수정

| 분류 | 내용 |
|---|---|
| **KST/ET 날짜** | `predictions/today`: DB `game_date`(ET 기준)를 KST로 조회할 때 `us_date = kst_today - 1일` 적용. 종전에는 당일 ET 날짜를 직접 사용해 밤 경기가 다음날 KST에서 누락됨 |
| **KST/ET 날짜** | `run_inference_v3.py`: `sync_schedule(us_today)` · `Game.game_date == us_today` 적용. `date.today()` → KST 기준 변환 |
| **KST/ET 날짜** | `archive/summary`: `us_date = kst_date - 1일` 적용. `archive/calendar` SQL: `game_date + INTERVAL '1 day'`로 KST 그룹핑 |
| **Redis 캐시 키** | `predictions/today` 캐시 키를 날짜 접미사 포함 `predictions:today:{YYYY-MM-DD}` 형식으로 변경. 자정 교체 시 이전 날 캐시 무효화 |
| **game_datetime 누락** | `_build_game_payload()` 응답에 `game_datetime` 필드 추가. 경기일정 탭 시간 표시에 필요 |
| **예측 날짜 UPSERT** | `run_inference_v3.py`: `prediction_date=today` 누락으로 archive 조회 시 날짜 미매칭 발생 → UPSERT `set_` 딕셔너리에 추가 |
| **선발 투수명 DB** | 추론 후 `players` 테이블에 선발 투수명을 즉시 upsert하도록 수정 (종전: 이름 미기재) |
| **GameWeather 중복** | `game_pk` unique 제약 누락으로 UPSERT가 중복 행 생성 → `unique=True` 추가 |
| **5/26 혼합 표시** | 5/26 KST 경기에 5/25 + 5/26 US 날짜 경기가 섞여 표시되던 문제 수정 |

### 신규 기능

| 기능 | 설명 |
|---|---|
| **`/predictions/history`** | 특정 날짜의 예측 이력 조회 (`?date=YYYY-MM-DD`). 달력 과거기록 뷰에서 사용 |
| **archive on-the-fly 채점** | `is_correct` DB 필드가 null이어도 스코어가 있으면 archive API에서 즉석 계산 후 반환 |
| **경기일정 ✓/✗ 아이콘** | 경기일정 탭의 각 경기 우측에 MED/HIGH 배지 + ✓(맞음)/✗(틀림) 아이콘 표시 |
| **달력 과거기록 뷰** | "결과 LOG" 탭 제거 후 경기일정 달력이 과거기록 열람 통합 역할 담당 |
| **`fix_player_stubs.py`** | MLB Stats API 벌크 조회로 DB 내 `"Player #N"` / `"TBD"` 이름 61건 실명 일괄 교체 |
| **`park_factor.py`** | 구장 파크팩터 수집 전용 모듈 추가 (`src/collector/park_factor.py`) |
| **Noto Sans KR 폰트** | 프론트엔드 한글 가시성 개선 |

---

## 4. 시스템 파이프라인

### 4.1 일일 고정 잡 (KST 기준)

```
03:00 (일) 주간 모델 재학습 (Optuna 50 trials)
─────────────────────────────────────────────
06:30  BBref 스텔스 스크래퍼 (pitching/batting CSV 덮어쓰기)
07:00  run_morning_pipeline
         ├ 전일 경기 결과 update_game_results (is_correct 채점)
         └ sync_schedule(today)  ← 오늘 일정 사전 동기화
07:30  FanGraphs 선수 통계 갱신
12:00  전일 Statcast append (delta 1일)
12:30  FanGraphs 리더보드 재수집
13:00  👑 마스터 스케줄러: 오늘 경기마다 동적 워커 등록
─────────────────────────────────────────────
19:30  run_inference_fallback (동적 워커 누락 경기 일괄 보충)
```

### 4.2 경기별 동적 워커 (DateTrigger)

```
T-120min  pre_game_sync_{pk}    : Live Feed 1차 sync (날씨/라인업/선발)
T-15min   inference_{pk}        : Live Feed 재호출 + 47피처 + LLM → 베이스 예측
T+0min    live_poller_start_{pk}: 1분 polling 시작
                                   ├─ game_live_states INSERT
                                   ├─ game_predictions.live_* UPDATE
                                   └─ Redis publish (frontend 푸시)
[Final 감지] postgame_sync_{pk} : boxscore → game_logs UPSERT (자가 트리거)
```

### 4.3 스케줄러 수동 시작

스케줄러(`pipeline.py`)는 uvicorn API와 **별개 프로세스**로 실행해야 한다.

```bash
# Poetry 가상환경으로 실행 (필수)
poetry run python src/collector/pipeline.py

# 또는 가상환경 Python 직접 지정
C:\Users\<user>\AppData\Local\pypoetry\Cache\virtualenvs\mlb-prediction-<hash>-py3.12\Scripts\python.exe src/collector/pipeline.py
```

> **주의**: 시작 직후 `master_daily_scheduler()`가 즉시 호출되어 당일 경기 워커를 등록한다.
> uvicorn만 실행해서는 스케줄러가 동작하지 않는다.

---

## 5. 데이터 수집 방식

### 5.1 Baseball Reference (BBref) — 팀 시즌 통계 + dWAR

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

### 5.2 MLB StatsAPI — 경기 일정/결과/라이브

**엔드포인트**: `https://statsapi.mlb.com/api/v1`

```python
# v1 endpoints
GET /schedule?sportId=1&date=YYYY-MM-DD            # 경기 일정 (ET 날짜 기준)
GET /game/{game_pk}/linescore                       # 라이브 폴러 (1분 polling)
GET /game/{game_pk}/boxscore                        # postgame 수집

# v2 신규 — Live Feed 통합 (1콜로 날씨+라인업+선발 동시)
GET /api/v1.1/game/{game_pk}/feed/live
  → gameData.weather.{temp, condition, wind}
  → gameData.probablePitchers.{home,away}.id
  → liveData.boxscore.teams.{home,away}.battingOrder[]
```

> **날짜 기준**: MLB Stats API는 **ET(Eastern Time) 기준** 날짜를 사용한다.
> KST 날짜 D의 경기를 조회할 때는 `us_date = kst_date - 1일`을 사용해야 한다.

**수집 내용**: 경기 일정, 홈/어웨이 팀 ID, 최종 스코어, 선발 투수 MLBAM ID, **확정 라인업 9명 mlbam_id**, **라이브 inning/score/winProbability**

### 5.3 Statcast — 투구 데이터 (pybaseball)

```python
import pybaseball
df = pybaseball.statcast(start_dt="2024-04-01", end_dt="2024-04-05")
```

- 5일 단위로 분할 요청 (502 에러 방지)
- `asyncio.to_thread` 래핑으로 비동기 처리
- `is_hard_hit` (95mph+), `is_barrel` 컬럼 직접 계산
- `release_speed`, `spin_rate`, `launch_speed`, `launch_angle` 저장

### 5.4 Open-Meteo — 구장 날씨

```
GET https://api.open-meteo.com/v1/forecast
    ?latitude={lat}&longitude={lon}
    &hourly=temperature_2m,wind_speed_10m
```

MLB 30개 구장 좌표 하드코딩 → 당일 경기 시각(현지 시간) 기준 기온/풍속 조회

### 5.5 FanGraphs — 선발 투수 세이버메트릭스

`pybaseball.pitching_stats()` / `pybaseball.batting_stats()` 활용
- FIP, xFIP, SIERA, SwStr%, Contact% 등 수집
- `as_of_date` 기준 시점 제한 적용 (데이터 누수 방지)

### 5.6 Park Factor 수집 (`src/collector/park_factor.py`)

구장 파크팩터 전용 수집 모듈 (v2.1 신규). `park_factors.json`을 자동 갱신한다.

---

## 6. 피처 엔지니어링

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

## 7. ML 모델 및 성능

### 7.1 모델 구조

```
LightGBMClassifier ─┐
                    ├─→ 가중 앙상블 ─→ IsotonicRegression ─→ 최종 확률
XGBClassifier ──────┘   (역 logloss 가중치)   (확률 보정)
```

### 7.2 v1 vs v2 성능 비교 (테스트셋)

| 지표 | v1 (29피처, 4912경기) | **v2 (47피처, 8082경기)** | 변화 |
|---|---|---|---|
| LightGBM AUC | 0.5263 | **0.5750** ⭐ | **+4.87%p** |
| XGBoost AUC | 0.5507 | 0.5540 | +0.33%p |
| 앙상블+캘 AUC | 0.5336 | **0.5640** ✅ | **+3.04%p** |
| 학습 데이터 | 4,912경기 | 8,082경기 | +65% |
| Optuna trials | 20 | 100 (seed 고정) | — |

> MLB 경기 결과는 본질적으로 고노이즈 도메인 (홈팀 실제 승률 ~52%). AUC 0.56+가 도메인 한계 근처.

### 7.3 v2 하이퍼파라미터 튜닝 (Optuna 100 trials, TPESampler seed=42)

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

### 7.4 신뢰도 배지 기준

| 레벨 | 조건 | 의미 |
|---|---|---|
| HIGH (초록) | `\|확률 - 0.5\| > 0.15` | 모델이 한 팀을 강하게 지목 |
| MED (노랑) | `\|확률 - 0.5\| > 0.05` | 어느 정도 유리한 팀 있음 |
| LOW (회색) | `\|확률 - 0.5\| ≤ 0.05` | 거의 반반 |

---

## 8. AI 근거 생성 (LLM)

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
    {"feature": "roll_win_diff", "value": 0.15, "shap_value": 0.067}
  ]
}
```

### LLM 폴백

`.env`에서 `LLM_PROVIDER=groq`로 변경 시 Groq (llama-3.1-8b-instant) 무료 사용 가능 — API 크레딧 소진 시 코드 변경 없이 전환.

---

## 9. 라이브 스코어 + 라이브 예측 (v2 신규)

### 9.1 라이브 폴러 (`src/collector/live_score_poller.py`)

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

### 9.2 라이브 승률 계산 (`src/ml/live_win_prob.py`)

별도 in-game 모델을 학습하지 않고 다음 우선순위:

1. **MLB 공식 winProbability** (가능 시) — `liveData.linescore.teams.home.winProb`
2. **Tom Tango "The Book" 룩업 테이블** — (inning, lead) → 홈 승률
3. **베이스 가중평균** — 룩업 없으면 진행도(inning/9)에 따라 base ↔ 0.5 감쇠

### 9.3 자동 postgame 수집 (`src/collector/postgame_collector.py`)

라이브 폴러가 `status=Final` 감지 시 즉시(T+5min) 호출:

```
GET /api/v1/game/{pk}/boxscore
  → games.{home_score, away_score, status='Final'}        UPDATE
  → pitcher_game_logs (ip, er, k, bb, h, hr, pitches)     UPSERT
  → batter_game_logs  (ab, h, hr, rbi, bb, k)             UPSERT
```

IP 변환: `"5.2"` (5와 2/3 이닝) → `5.667` 정확 계산.

---

## 10. 기술 스택

| 레이어 | 기술 | 버전 | 역할 |
|---|---|---|---|
| **언어** | Python | 3.12 | 백엔드·ML 전체 |
| **DB** | PostgreSQL | 16 | 경기·선수·예측 데이터 |
| **캐시** | Redis | 7 | API 응답 캐시 (날짜별 TTL) |
| **ORM** | SQLAlchemy | 2.0 | Python ↔ PostgreSQL |
| **마이그레이션** | Alembic | 1.13+ | DB 스키마 버전 관리 |
| **의존성** | Poetry | 최신 | Python 패키지 관리 |
| **스케줄러** | APScheduler | 3.x | 7개 자동 파이프라인 잡 (별도 프로세스) |
| **ML** | LightGBM | 4.3+ | 그래디언트 부스팅 분류 |
| **ML** | XGBoost | 2.0+ | 그래디언트 부스팅 분류 |
| **튜닝** | Optuna | 3.5+ | 하이퍼파라미터 자동 최적화 (100 trials) |
| **보정** | scikit-learn | 1.4+ | IsotonicRegression |
| **설명 가능 AI** | SHAP | 0.45+ | TreeExplainer 피처 기여도 |
| **LLM** | OpenAI gpt-4o-mini | — | 한국어 분석 근거 생성 |
| **데이터** | pybaseball | — | Statcast/FanGraphs 수집 |
| **백엔드** | FastAPI + Uvicorn | 0.110+ | REST API |
| **프론트엔드** | Next.js | 14 | 예측 결과 웹 UI |
| **폰트** | Noto Sans KR | — | 한국어 가시성 개선 |
| **컨테이너** | Docker Compose | — | 로컬 개발 환경 |
| **CI/CD** | GitHub Actions | — | 린트 + 일일 파이프라인 |
| **배포 (백엔드)** | Render | 무료 티어 | FastAPI + Background Worker |
| **배포 (프론트)** | Vercel | Hobby(무료) | Next.js |
| **알림** | Discord Webhook | — | 파이프라인 모니터링 |

---

## 11. 프로젝트 구조

```
SimMLB/
├── src/
│   ├── api/                        # FastAPI 백엔드
│   │   ├── main.py                 # 앱 진입점, CORS 설정
│   │   └── routers/
│   │       ├── predictions.py      # /predictions/* 엔드포인트 + Redis 캐시
│   │       ├── archive.py          # 🆕 v2.1: /archive/* (날짜별 예측 이력)
│   │       └── health.py           # /health 헬스체크
│   │
│   ├── collector/                  # 데이터 수집
│   │   ├── base.py                 # BaseCollector (재시도·Rate Limit, timeout 지원)
│   │   ├── mlb_statsapi_client.py  # MLB 공식 API (경기/선수/라인업)
│   │   ├── statcast_collector.py   # Statcast (pybaseball)
│   │   ├── fangraphs_collector.py  # FanGraphs (FIP, xFIP, wRC+)
│   │   ├── weather_client.py       # Open-Meteo (구장 날씨, v2 폴백)
│   │   ├── roster_sync.py          # 팀 로스터 동기화
│   │   ├── live_feed_client.py     # v2: MLB Live Feed (날씨/라인업/선발 통합)
│   │   ├── live_score_poller.py    # v2: 1분 간격 라이브 스코어 + 라이브 승률
│   │   ├── postgame_collector.py   # v2: 경기 종료 후 boxscore → game_logs
│   │   ├── bref_scraper.py         # v2: BBref 스텔스 스크래퍼
│   │   ├── park_factor.py          # 🆕 v2.1: 구장 파크팩터 수집 모듈
│   │   └── pipeline.py             # APScheduler (고정 cron + 동적 워커 4종)
│   │
│   ├── db/                         # 데이터베이스
│   │   ├── session.py              # SQLAlchemy 엔진·세션 관리
│   │   ├── base.py                 # ORM Base 클래스
│   │   ├── player_utils.py         # 선수 stub 처리 유틸
│   │   └── models/
│   │       ├── teams.py            # MLB 30개 팀
│   │       ├── players.py          # 선수 마스터 (MLBAM ID)
│   │       ├── games.py            # 경기 기록 (GameWeather unique 제약 포함)
│   │       └── predictions.py      # ML 예측 결과
│   │
│   ├── ml/                         # ML 파이프라인
│   │   ├── features/
│   │   │   ├── bref_features.py    # BBref 피처 모듈 (v1 29피처)
│   │   │   └── bref_features_v2.py # v2: 47피처 (라인업/선발 Statcast/dWAR/휴식)
│   │   ├── models/                 # LightGBM, XGBoost, 앙상블
│   │   ├── reasoning/              # LLM 추상화 + 프롬프트 빌더
│   │   ├── live_win_prob.py        # v2: 라이브 승률 조정 함수
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
│   ├── backfill_historical.py      # 2023–2026 과거 데이터 백필
│   ├── backfill_lineups.py         # game_lineups 백필
│   ├── build_and_train.py          # v1 학습 (29피처)
│   ├── build_and_train_v2.py       # v2: 47피처 학습 (--features v2|v3)
│   ├── run_inference_v2.py         # v1 모델 추론 (29피처)
│   ├── run_inference_v3.py         # v2 추론 (47피처 + Live Feed, KST 날짜 수정)
│   ├── run_daily_inference.py      # 일일 추론 (v2 자동 감지 + v1 폴백)
│   ├── fix_player_stubs.py         # 🆕 v2.1: "Player #N"/TBD 이름 MLB API로 일괄 수정
│   ├── update_dwar.py              # BBref WAR Daily → dWAR 마스터 CSV
│   └── verify_data_integrity.py    # DB 데이터 무결성 검증
│
├── frontend/                       # Next.js 14
│   ├── app/
│   │   ├── page.tsx                # 메인 페이지 (예측·일정·순위 탭 통합)
│   │   └── globals.css             # 전역 스타일 (Noto Sans KR + RC chip)
│   └── ...
│
├── migrations/                     # Alembic DB 마이그레이션
│   └── versions/
│       ├── 140ec83bcd4a_initial_schema.py
│       ├── 4fb0b416e598_add_game_date_....py
│       └── a1b2c3d4e5f6_live_states_and_postgame.py
│
├── data/
│   ├── raw/
│   │   ├── bref_pitching_{year}.csv
│   │   ├── bref_batting_{year}.csv
│   │   ├── bref_dwar_master.csv
│   │   ├── war_archive/
│   │   │   ├── war_daily_bat.txt
│   │   │   └── war_daily_pitch.txt
│   │   └── park_factors.json
│   └── training_sets/
│       └── training_set_v2.parquet
│
├── models/
│   ├── lgbm_v1.pkl / xgb_v1.pkl / calibrator_v1.pkl
│   ├── lgbm_v2.pkl                 # v2 LightGBM (AUC 0.5750)
│   ├── xgb_v2.pkl                  # v2 XGBoost (AUC 0.5540)
│   └── calibrator_v2.pkl           # v2 앙상블+캘 (AUC 0.5640 ✅)
│
├── .github/workflows/
│   ├── ci.yml
│   └── daily_pipeline.yml
│
├── render.yaml
├── docker-compose.yml
├── pyproject.toml
└── config/settings.py
```

---

## 12. 빠른 시작

### 필요 조건

- Python 3.12+
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

# 6. 스케줄러 실행 (별도 터미널 — API와 독립 프로세스)
poetry run python src/collector/pipeline.py

# 7. 당일 추론 수동 실행 (스케줄러 T-15 워커 전 수동 테스트 시)
poetry run python scripts/run_inference_v3.py
```

> **캐시 주의**: 추론 전 프론트엔드 접속 시 Redis가 빈 응답을 캐시합니다.
> 아래 명령어로 캐시를 지운 뒤 새로고침하세요:
> ```bash
> docker exec simmlb-redis-1 redis-cli FLUSHALL
> # 또는 날짜별 키만 삭제
> docker exec simmlb-redis-1 redis-cli DEL "predictions:today:2026-05-27"
> ```

### 프론트엔드 실행

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### 선수명 스텁 수정

DB에 `"Player #N"` 또는 `"TBD"` 형태의 선수명이 있을 경우 MLB Stats API로 일괄 교체:

```bash
poetry run python scripts/fix_player_stubs.py
```

### 모델 재학습

```bash
# BBref CSV를 data/raw/에 업데이트 후
poetry run python scripts/build_and_train_v2.py --features v2
```

---

## 13. API 명세

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 서버 헬스체크 |
| GET | `/predictions/today` | 당일 전 경기 예측 목록 (KST 기준, Redis 캐시) |
| GET | `/predictions/{game_pk}` | 특정 경기 예측 상세 (확률 + SHAP + 근거) |
| GET | `/predictions/history` | 특정 날짜 예측 이력 (`?date=YYYY-MM-DD`) |
| GET | `/games/{game_pk}` | 특정 경기 기본 정보 |
| GET | `/archive/summary` | 날짜별 예측 결과 요약 (`?target_date=YYYY-MM-DD`) |
| GET | `/archive/calendar` | 월별 달력 데이터 (`?year=2026&month=5`) |

### 응답 예시 (`/predictions/today`)

```json
{
  "date": "2026-05-27",
  "count": 15,
  "games": [
    {
      "game_pk": 746123,
      "home_team": "LAD",
      "away_team": "NYY",
      "game_datetime": "2026-05-27T19:10:00",
      "home_win_prob": 0.623,
      "away_win_prob": 0.377,
      "confidence": "MED",
      "home_starter_name": "Walker Buehler",
      "away_starter_name": "Gerrit Cole",
      "reasoning": "홈팀 투수진 FIP(3.41)가 원정팀보다 우수하며, 최근 20경기 승률(0.650)이 리그 평균을 상회한다.",
      "shap_top5": [
        {"feature": "fip_diff", "value": 0.71, "shap_value": 0.089}
      ]
    }
  ]
}
```

### 응답 예시 (`/archive/summary?target_date=2026-05-26`)

```json
{
  "date": "2026-05-26",
  "total": 11,
  "graded": 11,
  "correct": 5,
  "accuracy": 45.5,
  "high_med_accuracy": 50.0,
  "games": [
    {
      "game_pk": 825004,
      "home_team": "NYM",
      "away_team": "ATL",
      "home_score": 4,
      "away_score": 2,
      "home_win_prob": 0.572,
      "confidence": "MED",
      "pick_team": "NYM",
      "actual_winner": "NYM",
      "is_correct": 1
    }
  ]
}
```

---

## 14. 배포 구성

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
    startCommand: python src/collector/pipeline.py
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
  │     ├── game_weather      (날씨 정보, game_pk unique)
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
+ prediction_date  ← v2.1: KST 기준 날짜 (archive 조회에 사용)
```

---

## 팀 구조

| 팀 | 담당 영역 | 주요 파일 |
|---|---|---|
| **A팀** (Data) | 수집기, DB, 스케줄러 | `src/collector/*`, `src/db/*` |
| **B팀** (ML) | 피처, 모델, 추론, LLM | `src/ml/*`, `scripts/build_and_train_v2.py` |
| **C팀** (API/Frontend) | FastAPI, Next.js | `src/api/*`, `frontend/*` |

---

## 15. 트러블슈팅 / 개발 메모

### 15.1 v1 → v2 마이그레이션 시 겪었던 이슈

| 이슈 | 원인 | 해결 |
|---|---|---|
| **BBref Cloudflare 차단** | sandbox/CI IP를 Cloudflare 챌린지 | `bref_scraper.py` UA 로테이션 + 429 시 60s 대기 |
| **WAR 마스터 자동 다운로드 차단** | 동일 Cloudflare | `--import-bat`/`--import-pit` 로컬 import 옵션 추가 |
| **`game_lineups` 테이블 비어있음** | 학습 데이터에 라인업 미반영 | `backfill_lineups.py`로 MLB Stats API boxscore 4,914경기 백필 |
| **`launch_speed` 평균 82.5 (표준 88+)** | foul ball 포함 | SQL 필터 `description = 'hit_into_play'` 추가 |
| **Optuna stochasticity ±2%p** | seed 미고정 + n_trials 부족 | `TPESampler(seed=42)` + n_trials=100 |

### 15.2 v2.1 패치 시 겪었던 이슈

| 이슈 | 원인 | 해결 |
|---|---|---|
| **예측이 매일 자정 사라짐** | Redis 캐시 키 미포함 날짜 — 자정에 새 요청이 이전 날짜 키를 덮어씀 | 캐시 키를 `predictions:today:{YYYY-MM-DD}` 형식으로 변경 |
| **5/26 경기가 5/27 KST 쿼리에서 빠짐** | `Game.game_date == kst_today`로 직접 비교 → ET 기준 DB와 불일치 | `us_date = kst_today - timedelta(days=1)` 적용 |
| **game_datetime 필드 누락** | `_build_game_payload()`에 미포함 | `"game_datetime": game.game_datetime.isoformat()` 추가 |
| **archive 적중률 항상 null** | `is_correct`가 아침 파이프라인 전까지 null | archive API에서 스코어 있으면 즉석 계산 |
| **prediction_date 누락으로 archive 조회 실패** | UPSERT `set_` 딕셔너리에 `prediction_date` 미포함 | `set_=dict(..., prediction_date=today)` 추가 |
| **GameWeather 중복 행** | `game_pk` unique 제약 없음 | `models/games.py`에 `unique=True` 추가 |
| **선수명이 "Player #N" 형태** | 로스터 sync 시 MLB API 호출 실패 시 stub 저장 | `fix_player_stubs.py`로 61건 일괄 수정 |
| **스케줄러 미기동** | uvicorn만 실행하면 APScheduler가 동작 안 함 | `pipeline.py`를 별도 프로세스로 실행 필요 |
| **Poetry venv 외 Python 사용** | `python` 명령이 시스템 Python(3.12 bare) 실행 | 항상 `poetry run python` 또는 venv 절대 경로 사용 |

### 15.3 라이브 폴러 잡 ID 충돌 방지

```python
# 마스터가 등록한 startup 잡 (DateTrigger T+0)
livestart_{pk}  →  run start_live_poller()
# start_live_poller가 등록하는 polling 잡 (IntervalTrigger 1분)
live_{pk}       →  run _live_poll_tick(pk)
# Final 감지 후 등록되는 postgame 잡 (DateTrigger T+5min)
post_{pk}       →  run run_postgame_sync(pk)
```

서로 다른 ID 접두사로 충돌 없음. `replace_existing=True` + `max_instances=1`로 중복 방지.

### 15.4 다중 seed 학습 결과 (v2 final)

| seed | LGBM | XGB | Cal+Ensemble |
|---|---|---|---|
| 42 | 0.5520 | 0.5510 | 0.5565 |
| 7 | 0.5456 | **0.5819** | 0.5526 |
| **123** ⭐ | 0.5594 | 0.5643 | **0.5606** (4912 학습본) |
| 2024 | 0.5482 | 0.5321 | 0.5376 |
| 999 | 0.5347 | 0.5630 | 0.5419 |

**8082경기 재학습 후 최종**: LGBM 0.5750 / XGB 0.5540 / **Cal 0.5640 ✅**

### 15.5 다음 단계 권장 (향후 추가 향상)

| 방안 | 예상 효과 | 비용 |
|---|---|---|
| 라인업 9명 최근 30일 statcast (game_date 기준 window) | ⭐⭐⭐ | 큼 (DB 재구조화) |
| 선발 vs 라인업 hand matchup (L/R split) | ⭐⭐ | 중 |
| FanGraphs advanced metrics (wOBA, xwOBA, BABIP) | ⭐⭐ | 중 |
| Bullpen 통계 (구원진 ERA/FIP) | ⭐ | 중 |
| 2027+ 신규 시즌 데이터 누적 후 재학습 | ⭐⭐⭐ | 작음 (자동) |

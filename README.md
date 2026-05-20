# ⚾ SimMLB — MLB 승부예측 AI 시스템

MLB 정규시즌 당일 경기의 **홈팀 승리 확률을 ML 모델로 예측**하고, GPT-4o-mini가 **한국어 분석 근거를 자동 생성**하는 풀스택 AI 시스템.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 파이프라인](#2-시스템-파이프라인)
3. [데이터 수집 방식](#3-데이터-수집-방식)
4. [피처 엔지니어링](#4-피처-엔지니어링)
5. [ML 모델 및 성능](#5-ml-모델-및-성능)
6. [AI 근거 생성 (LLM)](#6-ai-근거-생성-llm)
7. [기술 스택](#7-기술-스택)
8. [프로젝트 구조](#8-프로젝트-구조)
9. [빠른 시작](#9-빠른-시작)
10. [API 명세](#10-api-명세)
11. [배포 구성](#11-배포-구성)

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 예측 대상 | MLB 정규시즌 당일 경기 홈팀 승리 확률 |
| 예측 주기 | 매일 19:30 KST 자동 실행 |
| 학습 데이터 | 2023–2024 시즌 완료 경기 **4,912경기** |
| 피처 수 | **29개** (팀 롤링승률·투구지표·타격지표·구장팩터) |
| 모델 | LightGBM + XGBoost 앙상블 + Isotonic Calibration |
| 최종 AUC | **0.5507** (XGBoost), 앙상블 보정 후 0.5336 |
| LLM | GPT-4o-mini — SHAP 상위 5개 피처 기반 한국어 분석 2–3문장 |
| 인프라 | Render (백엔드 API + Worker) / Vercel (프론트엔드) |

---

## 2. 시스템 파이프라인

### 일일 자동 파이프라인 (KST 기준)

```
[데이터 수집 레이어]
  07:00  MLB StatsAPI → 전일 경기 결과 + 팀 스냅샷 업데이트
  07:30  FanGraphs    → 선발 투수 / 타자 시즌 통계 갱신
  12:00  Statcast     → 전일 투구 데이터 수집 (pybaseball)
  12:30  FanGraphs    → 리더보드 업데이트
  18:00  Open-Meteo   → 당일 구장 날씨 수집
  18:30  MLB StatsAPI → 확정 라인업 수집
         ↓
[피처 엔지니어링 레이어]
  BBref 팀 통계 + 롤링 승률 + 파크팩터 → 29개 피처 벡터 생성
         ↓
  19:30  [ML 추론 레이어]
  LightGBM (w=0.488) + XGBoost (w=0.512) → 앙상블 확률
  → Isotonic Calibration (확률 보정)
  → SHAP top-5 피처 추출
  → GPT-4o-mini → 한국어 분석 근거 2–3문장
  → PostgreSQL 저장 + Redis 1시간 캐시
         ↓
[서빙 레이어]
  FastAPI REST API ← Next.js 웹 UI
```

### 주간 재학습 (일요일 03:00 KST)

BBref 최신 시즌 데이터 반영 → Optuna 20 trials 재튜닝 → 모델 파일 교체

---

## 3. 데이터 수집 방식

### 3.1 Baseball Reference (BBref) — 팀 시즌 통계

**수집 방법**: BBref 웹사이트에서 수동 CSV 다운로드 후 `data/raw/` 저장

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

### 3.2 MLB StatsAPI — 경기 일정/결과/라인업

**엔드포인트**: `https://statsapi.mlb.com/api/v1`

```python
# 경기 일정
GET /schedule?sportId=1&date=YYYY-MM-DD

# 경기 결과 (라인스코어)
GET /game/{game_pk}/linescore

# 팀 로스터
GET /teams/{team_id}/roster?season=YYYY
```

**수집 내용**: 경기 일정, 홈/어웨이 팀 ID, 최종 스코어, 선발 투수 MLBAM ID, 확정 라인업

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

## 4. 피처 엔지니어링

### 피처 목록 (29개)

| 그룹 | 피처명 | 계산 방법 |
|---|---|---|
| **팀 롤링 승률** | `home_roll_win` | 경기 당일 이전 최근 20경기 승률 |
| | `away_roll_win` | 동일 |
| | `roll_win_diff` | home - away |
| **투구 지표** | `home_era` / `away_era` | BBref 시즌 ERA (전년도 또는 당해) |
| | `home_fip` / `away_fip` | FIP (피안타 독립 ERA) |
| | `era_diff` / `fip_diff` | away - home (양수=홈 유리) |
| | `home_whip` / `away_whip` | (피안타+볼넷)/이닝 |
| | `home_k9` / `away_k9` | 9이닝당 탈삼진 |
| | `home_bb9` / `away_bb9` | 9이닝당 볼넷 |
| **타격 지표** | `home_ops` / `away_ops` | OPS (출루율+장타율) |
| | `home_obp` / `away_obp` | 출루율 |
| | `home_slg` / `away_slg` | 장타율 |
| | `home_ba` / `away_ba` | 타율 |
| | `home_hr_pa` / `away_hr_pa` | 타석당 홈런율 |
| | `ops_diff` | home - away |
| **구장 팩터** | `park_run_factor` | 리그 평균 대비 득점 발생 비율 |
| | `park_hr_factor` | 리그 평균 대비 홈런 발생 비율 |
| | `is_dome` | 돔 구장 여부 (0/1) |

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

## 5. ML 모델 및 성능

### 5.1 모델 구조

```
LightGBMClassifier ─┐
                    ├─→ 가중 앙상블 ─→ IsotonicRegression ─→ 최종 확률
XGBClassifier ──────┘   (역 logloss 가중치)   (확률 보정)
```

### 5.2 하이퍼파라미터 튜닝 (Optuna 20 trials)

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

## 6. AI 근거 생성 (LLM)

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

## 7. 기술 스택

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

## 8. 프로젝트 구조

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
│   │   ├── base.py                 # BaseCollector (재시도·Rate Limit)
│   │   ├── mlb_statsapi_client.py  # MLB 공식 API (경기/선수/라인업)
│   │   ├── statcast_collector.py   # Statcast (pybaseball)
│   │   ├── fangraphs_collector.py  # FanGraphs (FIP, xFIP, wRC+)
│   │   ├── weather_client.py       # Open-Meteo (구장 날씨)
│   │   ├── roster_sync.py          # 팀 로스터 동기화
│   │   └── pipeline.py             # APScheduler 7개 잡 설정
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
│   │   │   ├── bref_features.py    # BBref 피처 모듈 (학습·추론 공통)
│   │   │   ├── team_features.py    # 팀 누적 지표
│   │   │   ├── pitcher_features.py # 선발 투수 지표
│   │   │   ├── batter_features.py  # 타자·Statcast 지표
│   │   │   └── context_features.py # 구장·날씨·라인업
│   │   ├── models/
│   │   │   ├── lgbm_model.py       # LightGBM + Optuna
│   │   │   ├── xgb_model.py        # XGBoost + Optuna
│   │   │   └── ensemble.py         # 가중 앙상블
│   │   ├── reasoning/
│   │   │   ├── llm_client.py       # OpenAI/Groq 추상화 레이어
│   │   │   └── prompt_builder.py   # SHAP→프롬프트 변환
│   │   ├── build_training_data.py  # 학습 데이터셋 빌드
│   │   ├── calibration.py          # Isotonic 보정
│   │   ├── feature_engineering.py  # 피처 통합
│   │   └── prediction_service.py   # 추론 파이프라인 통합
│   │
│   └── common/                     # 공통 유틸
│       ├── logger.py               # 구조화 로깅
│       ├── retry.py                # 재시도 데코레이터
│       └── exceptions.py           # 커스텀 예외
│
├── scripts/
│   ├── backfill_historical.py      # 2023–2024 과거 데이터 백필
│   ├── build_and_train.py          # 학습 데이터 빌드 + 모델 학습 통합
│   ├── run_daily_inference.py      # 일일 추론 실행 (GitHub Actions용)
│   ├── run_inference_v2.py         # BBref 기반 추론 (v2)
│   └── verify_data_integrity.py    # DB 데이터 무결성 검증
│
├── frontend/                       # C팀 — Next.js 14
│   ├── app/
│   │   └── page.tsx                # 당일 예측 테이블 (메인 페이지)
│   └── ...
│
├── migrations/                     # Alembic DB 마이그레이션
│   └── versions/
│       ├── 140ec83bcd4a_initial_schema.py   # 초기 스키마 (11개 테이블)
│       └── 4fb0b416e598_add_game_date_....py
│
├── data/
│   ├── raw/                        # BBref CSV + 파크팩터 JSON
│   │   ├── bref_pitching_{year}.csv
│   │   ├── bref_batting_{year}.csv
│   │   └── park_factors.json
│   └── training_sets/
│       └── training_set.parquet    # 4,912경기 × 33컬럼
│
├── models/                         # 학습된 모델 파일
│   ├── lgbm_v1.pkl                 # LightGBM (481KB)
│   ├── xgb_v1.pkl                  # XGBoost (396KB)
│   └── calibrator_v1.pkl           # 앙상블 가중치 + Isotonic Calibrator
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

## 9. 빠른 시작

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

## 10. API 명세

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

## 11. 배포 구성

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
```

---

## 팀 구조

| 팀 | 담당 영역 | 주요 파일 |
|---|---|---|
| **A팀** (Data) | 수집기, DB, 스케줄러 | `src/collector/*`, `src/db/*` |
| **B팀** (ML) | 피처, 모델, 추론, LLM | `src/ml/*`, `scripts/build_and_train.py` |
| **C팀** (API/Frontend) | FastAPI, Next.js | `src/api/*`, `frontend/*` |

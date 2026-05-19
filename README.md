# ⚾ MLB 승부예측 AI 시스템

MLB 정규시즌 당일 경기의 홈팀 승리 확률을 ML 모델로 예측하고, AI가 한국어 분석 근거를 자동 생성하는 시스템.

## 빠른 시작

```bash
# 1. 의존성 설치
poetry install

# 2. 환경변수 설정
cp .env.example .env
# .env 파일 편집 (비밀번호, API 키 입력)

# 3. DB + Redis 실행
docker compose up -d

# 4. DB 스키마 생성
poetry run alembic upgrade head

# 5. FastAPI 서버 실행
poetry run uvicorn src.api.main:app --reload
```

## 기술 스택

| 레이어 | 기술 |
|---|---|
| DB | PostgreSQL 16 |
| 캐시 | Redis 7 |
| ML | LightGBM + XGBoost + Isotonic Calibration |
| LLM | OpenAI gpt-4o-mini |
| 백엔드 | FastAPI |
| 프론트엔드 | Next.js 14 |

## 팀 구조

- **A팀**: 데이터 수집 & DB (`src/collector/*`, `src/db/*`)
- **B팀**: ML 모델 & 피처 엔지니어링 (`src/ml/*`)
- **C팀**: API & 프론트엔드 (`src/api/*`, `frontend/*`)

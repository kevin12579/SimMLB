from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import predictions, health

app = FastAPI(
    title="MLB Prediction API",
    version="1.0.0",
    description="MLB 경기 승부 예측 API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 배포 시 Vercel URL로 제한
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predictions.router, prefix="/predictions")

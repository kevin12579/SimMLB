from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import predictions, health
from src.api.routers import users, live, archive, standings

# DB 테이블 자동 생성
from src.db.base import Base
from src.db.session import engine
from src.db.models import games, predictions as pred_model, teams, players
from src.db.models.users import User, UserPick  # noqa

Base.metadata.create_all(bind=engine)

app = FastAPI(title="SIMMLB API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predictions.router, prefix="/predictions")
app.include_router(users.router,       prefix="/users")
app.include_router(live.router,        prefix="/live")
app.include_router(archive.router,     prefix="/archive")
app.include_router(standings.router,   prefix="/standings")
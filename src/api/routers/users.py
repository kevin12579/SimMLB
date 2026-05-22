from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from src.db.session import get_session
from src.db.models.users import User, UserPick
from src.db.models.games import Game
from src.db.models.predictions import GamePrediction
from src.db.models.teams import Team
from src.api.auth import hash_password, verify_password, create_token, decode_token

router = APIRouter()

class RegisterIn(BaseModel):
    username: str
    email: str
    password: str

class LoginIn(BaseModel):
    username: str
    password: str

class PickIn(BaseModel):
    game_pk: int
    pick_team: str

def _auth(authorization: str = None) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "인증이 필요합니다")
    payload = decode_token(authorization.split(" ", 1)[1])
    if not payload:
        raise HTTPException(401, "토큰이 유효하지 않습니다")
    with get_session() as session:
        user = session.query(User).filter(User.id == payload["sub"]).first()
        if not user: raise HTTPException(401, "사용자를 찾을 수 없습니다")
        session.expunge(user)
    return user

@router.post("/register")
def register(body: RegisterIn):
    if len(body.username) < 2: raise HTTPException(400, "닉네임은 2자 이상이어야 합니다")
    if len(body.password) < 6: raise HTTPException(400, "비밀번호는 6자 이상이어야 합니다")
    with get_session() as session:
        if session.query(User).filter(User.username == body.username).first():
            raise HTTPException(409, "이미 사용 중인 닉네임입니다")
        if session.query(User).filter(User.email == body.email).first():
            raise HTTPException(409, "이미 사용 중인 이메일입니다")
        user = User(username=body.username, email=body.email, password_hash=hash_password(body.password))
        session.add(user)
        session.flush()
        uid = user.id
    return {"token": create_token({"sub": uid, "username": body.username}), "username": body.username, "user_id": uid}

@router.post("/login")
def login(body: LoginIn):
    with get_session() as session:
        user = session.query(User).filter(User.username == body.username).first()
        if not user or not verify_password(body.password, user.password_hash):
            raise HTTPException(401, "닉네임 또는 비밀번호가 틀렸습니다")
        uid, uname = user.id, user.username
    return {"token": create_token({"sub": uid, "username": uname}), "username": uname, "user_id": uid}

@router.get("/me")
def me(authorization: str = Header(None)):
    user = _auth(authorization)
    with get_session() as session:
        picks = session.query(UserPick).filter(UserPick.user_id == user.id).all()
        graded  = [p for p in picks if p.is_correct is not None]
        correct = [p for p in graded if p.is_correct == 1]
        by_conf = {}
        for lv in ['HIGH','MED','LOW']:
            b = [p for p in graded if p.confidence == lv]
            h = [p for p in b if p.is_correct == 1]
            by_conf[lv] = {"n": len(b), "correct": len(h), "acc": round(len(h)/len(b)*100,1) if b else None}
        streak = 0
        for p in sorted(graded, key=lambda x: x.created_at or "", reverse=True):
            if p.is_correct == 1: streak += 1
            else: break
        return {
            "username": user.username, "user_id": user.id,
            "total": len(picks), "graded": len(graded), "correct": len(correct),
            "accuracy": round(len(correct)/len(graded)*100,1) if graded else None,
            "streak": streak, "by_conf": by_conf,
        }

@router.get("/me/picks")
def my_picks(authorization: str = Header(None)):
    user = _auth(authorization)
    with get_session() as session:
        picks = session.query(UserPick).filter(UserPick.user_id == user.id).order_by(UserPick.created_at.desc()).limit(200).all()
        return {"picks": [{"id":p.id,"game_pk":p.game_pk,"game_date":p.game_date,"home_team":p.home_team,"away_team":p.away_team,"pick_team":p.pick_team,"pick_prob":p.pick_prob,"confidence":p.confidence,"is_correct":p.is_correct} for p in picks]}

@router.post("/me/picks")
def add_pick(body: PickIn, authorization: str = Header(None)):
    user = _auth(authorization)
    with get_session() as session:
        if session.query(UserPick).filter(UserPick.user_id==user.id, UserPick.game_pk==body.game_pk).first():
            raise HTTPException(409, "이미 픽한 경기입니다")
        pred = session.query(GamePrediction).filter(GamePrediction.game_pk==body.game_pk).first()
        game = session.query(Game).filter(Game.game_pk==body.game_pk).first()
        if not pred or not game: raise HTTPException(404, "경기 예측을 찾을 수 없습니다")
        team_map = {t.mlbam_team_id: t.abbreviation for t in session.query(Team).all()}
        home, away = team_map.get(game.home_team_id,""), team_map.get(game.away_team_id,"")
        if body.pick_team not in [home, away]: raise HTTPException(400, "유효하지 않은 팀입니다")
        pick_prob = pred.home_win_prob if body.pick_team == home else pred.away_win_prob
        session.add(UserPick(
            user_id=user.id, game_pk=body.game_pk, pick_team=body.pick_team,
            pick_prob=round(pick_prob,3), confidence=pred.confidence_level,
            is_correct=pred.is_correct, game_date=str(game.game_date),
            home_team=home, away_team=away,
        ))
    return {"ok": True, "pick_team": body.pick_team, "pick_prob": round(pick_prob*100,1)}

@router.delete("/me/picks/{game_pk}")
def delete_pick(game_pk: int, authorization: str = Header(None)):
    user = _auth(authorization)
    with get_session() as session:
        pick = session.query(UserPick).filter(UserPick.user_id==user.id, UserPick.game_pk==game_pk).first()
        if not pick: raise HTTPException(404, "픽을 찾을 수 없습니다")
        session.delete(pick)
    return {"ok": True}
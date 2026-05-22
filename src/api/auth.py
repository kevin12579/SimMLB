import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY = os.getenv("JWT_SECRET", "simmlb-secret-2026-change-in-prod")
ALGORITHM  = "HS256"
EXPIRE_MIN = 60 * 24 * 7  # 7일

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(p: str) -> str: return pwd_ctx.hash(p)
def verify_password(plain: str, hashed: str) -> bool: return pwd_ctx.verify(plain, hashed)
def create_token(data: dict) -> str:
    payload = {**data, "exp": datetime.now(timezone.utc) + timedelta(minutes=EXPIRE_MIN)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
def decode_token(token: str) -> Optional[dict]:
    try: return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError: return None
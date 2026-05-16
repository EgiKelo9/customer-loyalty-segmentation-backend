import jwt
from app.core.config import get_settings
from datetime import datetime, timedelta, timezone

class Token:
    def __init__(self):
        self.ALGORITHM = get_settings().ALGORITHM
        self.SECRET_KEY = get_settings().SECRET_KEY
        self.EXPIRATION_SECONDS = get_settings().ACCESS_TOKEN_EXPIRE_MINUTES * 60

    def generate_and_sign(self, user_id: str) -> str:
        expire = datetime.now(timezone.utc) + timedelta(seconds=self.EXPIRATION_SECONDS)
        payload = {"user_id": user_id, "exp": expire}
        return jwt.encode(payload, self.SECRET_KEY, algorithm=self.ALGORITHM)

    def verify_token(self, token: str):
        try:
            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
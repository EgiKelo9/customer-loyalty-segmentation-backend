import bcrypt
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime
from app.database.main import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True, index=True)
    password = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(
        timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(
        timezone.utc), nullable=False)
    deleted_at = Column(DateTime, nullable=True)

    def __init__(self, name: String, email: String, password: String):
        self.name = name
        self.email = email.strip().lower()
        self.password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def check_password(self, password: str) -> bool:
        return bcrypt.checkpw(password.encode(), self.password.encode())

    def with_email(self, email):
        self.email = email.strip().lower()
        return self

    def delete(self):
        self.deleted_at = datetime.now(timezone.utc)
        return self
import asyncio
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.exc import IntegrityError

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.controller import auth as auth_controller
from app.schemas.auth import UserLoginRequest, UserRegisterRequest
from app.shared import auth as shared_auth


class FakeUser:
    def __init__(self, name, email, password):
        self.id = 1
        self.name = name
        self.email = email.strip().lower()
        self.password = password


class FakeSession:
    def __init__(self, flush_error=None):
        self.flush_error = flush_error
        self.added = None
        self.flushed = False

    def add(self, obj):
        self.added = obj

    def flush(self):
        self.flushed = True
        if self.flush_error is not None:
            raise self.flush_error


class FakeTransactionManager:
    def __init__(self, session):
        self.session = session

    def transaction(self):
        return self

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeToken:
    def __init__(self, payload=None):
        self.payload = payload or {"user_id": 99}

    def generate_and_sign(self, user_id):
        return f"token-for-{user_id}"

    def verify_token(self, token):
        return self.payload


class FakeInvalidToken:
    def verify_token(self, token):
        return None


class FakeQuery:
    def __init__(self, result):
        self.result = result

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.result


class FakeDB:
    def __init__(self, result):
        self.result = result

    def query(self, model):
        return FakeQuery(self.result)


def test_register_user_success(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(auth_controller, "TransactionManager", lambda db: FakeTransactionManager(session))
    monkeypatch.setattr(auth_controller, "User", FakeUser)

    user = UserRegisterRequest(name="Alice", email="Alice@Example.com", password="secret")
    response = asyncio.run(auth_controller.register(user, db=object()))

    assert response.code == 200
    assert response.error is False
    assert response.message == "User registered successfully"
    assert response.data.name == "Alice"
    assert response.data.email == "alice@example.com"
    assert session.flushed is True
    assert session.added is response.data


def test_register_user_duplicate_email(monkeypatch):
    duplicate_error = IntegrityError("statement", "params", Exception("duplicate"))
    session = FakeSession(flush_error=duplicate_error)
    monkeypatch.setattr(auth_controller, "TransactionManager", lambda db: FakeTransactionManager(session))
    monkeypatch.setattr(auth_controller, "User", FakeUser)

    user = UserRegisterRequest(name="Alice", email="Alice@Example.com", password="secret")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth_controller.register(user, db=object()))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Email already registered"


def test_login_user_success(monkeypatch):
    existing_user = SimpleNamespace(
        id=7,
        name="Alice",
        email="alice@example.com",
        check_password=lambda password: True,
    )
    monkeypatch.setattr(auth_controller, "Token", lambda: FakeToken())

    user = UserLoginRequest(email="Alice@Example.com", password="secret")
    response = asyncio.run(auth_controller.login(user, db=FakeDB(existing_user)))

    assert response.code == 200
    assert response.error is False
    assert response.message == "Login successful"
    assert response.data.access_token == "token-for-7"
    assert response.data.token_type == "bearer"


def test_login_user_not_found(monkeypatch):
    user = UserLoginRequest(email="missing@example.com", password="secret")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth_controller.login(user, db=FakeDB(None)))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "User not found"


def test_login_user_incorrect_password(monkeypatch):
    existing_user = SimpleNamespace(id=7, check_password=lambda password: False)

    user = UserLoginRequest(email="Alice@Example.com", password="wrong-password")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth_controller.login(user, db=FakeDB(existing_user)))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Incorrect password"


def test_logout_user_success():
    response = asyncio.run(auth_controller.logout({"user_id": 7}))

    assert response.code == 200
    assert response.error is False
    assert response.message == "Logout successful"
    assert response.data == {"user_id": 7}


def test_get_current_user_success(monkeypatch):
    monkeypatch.setattr(shared_auth, "Token", lambda: FakeToken({"user_id": 42}))

    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="signed-token")
    payload = shared_auth.get_current_user(credentials)

    assert payload == {"user_id": 42}


def test_get_current_user_invalid_token(monkeypatch):
    monkeypatch.setattr(shared_auth, "Token", lambda: FakeInvalidToken())

    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid-token")

    with pytest.raises(HTTPException) as exc_info:
        shared_auth.get_current_user(credentials)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid or expired token"
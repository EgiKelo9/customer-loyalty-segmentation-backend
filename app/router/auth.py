from typing import Any, Dict
from sqlalchemy.orm import Session
from fastapi import Depends, APIRouter
from app.database.main import get_db
from app.schemas.base import StandardResponse
from app.schemas.auth import UserRegisterRequest, UserRegisterResponse, UserLoginRequest, UserLoginResponse
from app.controller.auth import register, login

router = APIRouter(prefix="/auth")


@router.post(
    "/register",
    response_model=StandardResponse[UserRegisterResponse],
    responses={
        422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
        401: {"model": StandardResponse[dict], "description": "Unauthorized"}
    }
)
async def register_user(user: UserRegisterRequest, db: Session = Depends(get_db)):
    return await register(user, db)

@router.post(
    "/login",
    response_model=StandardResponse[UserLoginResponse],
    responses={
        422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
        401: {"model": StandardResponse[dict], "description": "Unauthorized"}
    }
)
async def login_user(user: UserLoginRequest, db: Session = Depends(get_db)):
    return await login(user, db)
    
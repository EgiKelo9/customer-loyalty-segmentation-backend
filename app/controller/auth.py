from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.schemas.auth import UserRegisterRequest, UserLoginRequest, UserLoginResponse
from app.shared.transaction_manager import TransactionManager
from app.shared.token import Token
from app.models.user import User
from app.schemas.base import StandardResponse


async def register(user: UserRegisterRequest, db: Session):
    transaction_manager = TransactionManager(db)

    try:
        with transaction_manager.transaction() as session:
            new_user = User(name=user.name, email=user.email, password=user.password)
            session.add(new_user)
            session.flush()

    except IntegrityError as e:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    return StandardResponse(
        code=200,
        error=False,
        message="User registered successfully",
        data=new_user
    )
    
async def login(user: UserLoginRequest, db: Session):
    email_search = user.email.strip().lower()
    existing_user = db.query(User).filter(User.email == email_search).first()
    
    if not existing_user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    
    if not existing_user.check_password(user.password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect password"
        )
    
    token_manager = Token()
    token = token_manager.generate_and_sign(existing_user.id)
    return StandardResponse(
        code=200,
        error=False,
        message="Login successful",
        data=UserLoginResponse(access_token=token, token_type="bearer")
    )


async def logout(current_user: dict):
    return StandardResponse(
        code=200,
        error=False,
        message="Logout successful",
        data={"user_id": current_user.get("user_id")}
    )
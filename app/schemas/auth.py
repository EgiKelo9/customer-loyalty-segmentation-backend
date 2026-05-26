from pydantic import BaseModel, ConfigDict, EmailStr

class UserRegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserRegisterResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    model_config = ConfigDict(from_attributes=True)
        
class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserLoginResponse(BaseModel):
    access_token: str
    token_type: str
from typing import Generic, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar("T")

class StandardResponse(BaseModel, Generic[T]):
    code: int
    error: bool
    message: str
    data: Optional[T] = None

from pydantic import BaseModel
from typing import Any, Dict, Optional, Union

class BasicHealthResponse(BaseModel):
    status: str
    service: str
    version: str

class DependencyStatus(BaseModel):
    status: str
    url: Optional[str] = None
    model: Optional[str] = None
    detail: Union[str, Dict[str, Any], None] = None

class FullDependenciesStatus(BaseModel):
    ml_service: DependencyStatus
    gemini: DependencyStatus

class FullHealthResponse(BaseModel):
    status: str
    service: str
    dependencies: FullDependenciesStatus

class HealthMLServiceResponse(BaseModel):
    status: str
    url: str
    detail: Union[str, Dict[str, Any], None] = None
    
class HealthGeminiResponse(BaseModel):
    status: str
    model: str
    detail: Union[str, Dict[str, Any], None] = None
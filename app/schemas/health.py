from pydantic import BaseModel

class BasicHealthResponse(BaseModel):
    status: str
    service: str
    version: str
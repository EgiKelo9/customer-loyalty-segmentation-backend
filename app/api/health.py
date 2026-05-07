import os
import httpx
from typing import Any, Dict
from fastapi import APIRouter
from app.schemas.health import BasicHealthResponse

router = APIRouter()

@router.get("/", response_model=BasicHealthResponse)
def health_check():
    """
    Health check sederhana untuk backend itu sendiri.
    Tidak memanggil service lain — cocok untuk load balancer / tunnel probe.
    """
    return BasicHealthResponse(
        status="healthy",
        service="backend-fastapi",
        version="1.0.0"
    )

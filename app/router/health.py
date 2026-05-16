from typing import Any, Dict
from fastapi import APIRouter
from app.controller.health import check_health
from app.schemas.base import StandardResponse

router = APIRouter(prefix="/health")


@router.get(
    "/", 
    response_model=StandardResponse[dict],
    responses={
        422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
        401: {"model": StandardResponse[dict], "description": "Unauthorized"}
    }
)
def health_check():
    return check_health()

from app.schemas.base import StandardResponse


def check_health() -> StandardResponse[dict]:
    return StandardResponse(
        code=200,
        error=False,
        message="System is healthy",
        data={"status": "ok"}
    )

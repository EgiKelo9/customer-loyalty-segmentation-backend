import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.schemas.base import StandardResponse
from app.database.main import Base, engine, create_db
from app.middleware import cors, static
from app.router import auth, health, segmentation, analytics

create_db()
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Customer Loyalty Segmentation API",
    description="API untuk mengelola segmentasi loyalitas pelanggan menggunakan model machine learning.",
    version="1.0.0"
)

if not os.getenv("ENV"):
    os.environ["ENV"] = "dev"

cors.add(app)
static.add(app)

app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])
app.include_router(analytics.router, prefix="/api/v1", tags=["Analytics"])
app.include_router(segmentation.router, prefix="/api/v1", tags=["Segmentation"])
app.include_router(health.router, prefix="/api/v1", tags=["System Check"])

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=StandardResponse(
            code=exc.status_code,
            error=True,
            message=exc.detail,
            data=None
        ).model_dump()
    )

@app.get("/")
def root():
    return {"message": "Welcome to Customer Loyalty Segmentation API. Access /docs for API documentation."}

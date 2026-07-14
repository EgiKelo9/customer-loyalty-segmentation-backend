import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.schemas.base import StandardResponse
from app.database.main import Base, engine, create_db
from app.middleware import cors, static
from sqlalchemy import text

from app.router import auth, health, segmentation, analytics, promo

if not os.getenv("ENV"):
    os.environ["ENV"] = "dev"

app = FastAPI(
    title="Customer Loyalty Segmentation API",
    description="API untuk mengelola segmentasi loyalitas pelanggan menggunakan model machine learning.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """
    Ensures critical database indexes exist and are valid.
    Drops any invalid composite index before recreating it.
    Updates table statistics so the query planner uses indexes immediately.
    """
    try:
        with engine.connect() as conn:
            # 1. Drop the composite index if it exists AND is invalid.
            #    A valid index will not be touched.
            conn.execute(text("""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM pg_index
                        WHERE indexrelid = 'idx_seg_results_user_batch_created'::regclass
                        AND NOT indisvalid
                    ) THEN
                        DROP INDEX idx_seg_results_user_batch_created;
                    END IF;
                END $$;
            """))

            # 2. Create the composite index that perfectly matches the batch history query.
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_seg_results_user_batch_created "
                "ON segmentation_results (user_id, batch_id, created_at)"
            ))

            # 3. Create single‑column indexes (useful for other queries).
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_seg_results_user_id ON segmentation_results (user_id)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_seg_results_batch_id ON segmentation_results (batch_id)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_seg_results_created_at ON segmentation_results (created_at)"
            ))

            # 4. Indeks untuk query "latest per customer" (sering dipakai di KPI & customer list)
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_seg_results_latest_per_cust "
                "ON segmentation_results (user_id, customer_id, created_at DESC)"
            ))

            # 5. Indeks untuk filter berdasarkan transaction_date (segment trends)
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_seg_results_tx_date "
                "ON segmentation_results (user_id, transaction_date)"
            ))

            # 6. Update table statistics so the planner uses the new indexes immediately.
            conn.execute(text("ANALYZE segmentation_results"))
            conn.commit()

    except Exception as e:
        # Log the warning but don’t crash the startup.
        print(f"WARNING: Could not create indexes: {e}")

    # Check that the static dataset exists (warning only).
    path = "static/dataset/segmented_data.csv"
    if not os.path.exists(path):
        print(f"WARNING: Static dataset not found at {path}")

if not os.getenv("ENV"):
    os.environ["ENV"] = "dev"

cors.add(app)
static.add(app)

app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])
app.include_router(analytics.router, prefix="/api/v1", tags=["Analytics"])
app.include_router(segmentation.router, prefix="/api/v1", tags=["Segmentation"])
app.include_router(health.router, prefix="/api/v1", tags=["System Check"])
app.include_router(promo.router, prefix="/api/v1", tags=["Promo"])

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

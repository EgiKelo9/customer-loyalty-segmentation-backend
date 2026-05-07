from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import health

# Inisiasi FastAPI app
app = FastAPI(
    title="HomeIQ API",
    description="HomeIQ untuk AI House Price Prediction.",
    version="1.0.0",
)

# Konfigurasi CORS untuk mengizinkan frontend React (port 3000) dan domain produksi
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://0.0.0.0:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(health.router, prefix="/api/health", tags=["System Check"])

@app.get("/")
def root():
    return {"message": "Welcome to HomeIQ API. Akses /docs untuk melihat dokumentasi interaktif."}

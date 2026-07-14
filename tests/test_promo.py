import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database.main import Base, get_db
from app.shared.auth import get_current_user
from app.models.promo_config import PromoConfig

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB

@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(element, compiler, **kw):
    return "JSON"

from sqlalchemy.pool import StaticPool

# In-memory SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

def override_get_current_user():
    return {"id": 1, "email": "test@example.com"}

app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = override_get_current_user

client = TestClient(app)

def test_promo_post_no_redirect():
    payload = {
        "promo_type": "kupon",
        "params": {"discount": 15},
        "active": True,
        "cluster_id": 0
    }
    
    # POST without trailing slash (follow_redirects=False to verify redirect doesn't happen)
    response = client.post("/api/v1/promo", json=payload, follow_redirects=False)
    
    # Verify we get direct 200 OK without 307
    assert response.status_code == 200
    data = response.json()
    assert data["error"] is False
    assert data["message"] == "Campaign configured successfully"
    assert data["data"]["promo_type"] == "kupon"
    
def test_promo_get_and_delete():
    # Insert a promo config first
    payload = {
        "promo_type": "kupon",
        "params": {"discount": 15},
        "active": True,
        "cluster_id": 0
    }
    create_resp = client.post("/api/v1/promo", json=payload)
    assert create_resp.status_code == 200
    
    # Verify GET active promos works
    response = client.get("/api/v1/promo/active")
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 1
    promo_id = data["data"][0]["id"]
    
    # Verify GET by cluster ID works
    response = client.get("/api/v1/promo/cluster/0")
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["promo_type"] == "kupon"
    
    # Verify DELETE works using the cleaned path: /api/v1/promo/{promo_id}
    response = client.delete(f"/api/v1/promo/{promo_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["error"] is False
    assert data["message"] == "Campaign deleted successfully"
    
    # Verify GET active promos is now empty
    response = client.get("/api/v1/promo/active")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 0

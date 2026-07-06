from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.sql import func
from app.database.main import Base

class PromoConfig(Base):
    __tablename__ = "promo_configs"

    id          = Column(Integer, primary_key=True, index=True)
    promo_type  = Column(String, nullable=False)
    params      = Column(JSON, nullable=False, default={})
    active      = Column(Boolean, default=True)
    cluster_id  = Column(Integer, nullable=True)
    created_by  = Column(String, nullable=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), onupdate=func.now())
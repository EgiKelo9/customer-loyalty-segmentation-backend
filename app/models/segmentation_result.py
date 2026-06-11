from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB

from app.database.main import Base


class SegmentationResult(Base):
    __tablename__ = "segmentation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    batch_id = Column(String, nullable=True, index=True)
    customer_id = Column(String, nullable=True)
    cluster = Column(Integer, nullable=False)
    pattern = Column(String, nullable=False)
    segment = Column(String, nullable=False)
    recommendation = Column(String, nullable=False)
    fuzzy_membership = Column(JSONB, nullable=False)
    lrfm = Column(JSONB, nullable=True)
    applied_config_id = Column(Integer, nullable=True)
    applied_config = Column(JSONB, nullable=True)
    source = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

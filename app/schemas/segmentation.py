from datetime import datetime, date
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

class CustomerInput(BaseModel):
    L: float
    R: float
    F: float
    M: float
    transaction_date: Optional[date] = None

class TransactionInput(BaseModel):
    customer_id: str
    transaction_date: str
    invoice_id: str
    amount: float

class LRFMCalculated(BaseModel):
    L: float
    R: float
    F: float
    M: float

class RankedPromo(BaseModel):
    """
    A single promo recommendation entry from weighted fuzzy aggregation.

    promo_type  : normalized key matching frontend ALL_PROMO_TYPES value
                  e.g. "bonus_packs", "kupon", "cashback", "bogo",
                       "price_off", "sampling"
    score       : aggregated fuzzy membership weight (sum across contributing clusters)
    score_pct   : human-readable score string e.g. "28.00%"
    """
    promo_type: str
    score: float
    score_pct: str

class SegmentationResponse(BaseModel):
    customer_id: Optional[str] = None
    cluster: int
    pattern: str
    segment: str
    recommendation: str
    fuzzy_membership: Dict[str, str]
    lrfm_calculated: Optional[LRFMCalculated] = None
    batch_id: Optional[str] = None

class SegmentationHistoryItem(BaseModel):
    id: int
    customer_id: Optional[str] = None
    cluster: int
    pattern: str
    segment: str
    recommendation: str
    fuzzy_membership: Dict[str, str]
    lrfm_calculated: Optional[LRFMCalculated] = None
    applied_config_id: Optional[int] = None
    applied_config: Optional[Dict[str, Any]] = None
    source: str
    created_at: datetime

class BatchSegmentationResponse(BaseModel):
    status: str
    total_customers: int
    batch_id: str
    data: List[SegmentationResponse]

class ClusterAggregated(BaseModel):
    id: str
    name: str
    userCount: int
    avgRecency: float
    avgFrequency: float
    avgMonetary: float
    color: str
    description: str

class ScatterDataPoint(BaseModel):
    customer_id: str
    recency: float
    frequency: float
    monetary: float
    clusterId: str

class DistributionResponse(BaseModel):
    segments: List[ClusterAggregated]
    allSegmentData: List[ClusterAggregated]
    scatterData: List[ScatterDataPoint]

class BatchHistoryItem(BaseModel):
    batch_id: str
    source: str
    total_customers: int
    created_at: datetime
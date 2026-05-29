from typing import Dict, List, Optional
from pydantic import BaseModel

class CustomerInput(BaseModel):
    L: float
    R: float
    F: float
    M: float

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

class SegmentationResponse(BaseModel):
    customer_id: Optional[str] = None
    cluster: int
    pattern: str
    segment: str
    recommendation: str
    fuzzy_membership: Dict[str, str]
    lrfm_calculated: Optional[LRFMCalculated] = None

class BatchSegmentationResponse(BaseModel):
    status: str
    total_customers: int
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
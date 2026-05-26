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

class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    cluster: int
    pola: str
    segmen: str
    rekomendasi: str
    fuzzy_membership: Dict[str, str]
    lrfm_calculated: Optional[LRFMCalculated] = None

class BatchPredictionResponse(BaseModel):
    status: str
    total_pelanggan: int
    data: List[PredictionResponse]
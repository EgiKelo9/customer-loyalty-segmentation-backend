from pydantic import BaseModel
from typing import Any, Dict, Optional

class PromoConfigPayload(BaseModel):
    promo_type: str
    cluster_id: int
    params: Optional[Dict[str, Any]] = {}
    active: Optional[bool] = True
from typing import Any, Dict
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.main import get_db
from app.shared.auth import get_current_user
from app.schemas.base import StandardResponse

from app.controller.promo import (
    create_promo_config, 
    get_all_active_promos_logic, 
    get_promo_by_cluster_logic,
    delete_promo_config_logic,
    get_promo_metadata_logic,
    PromoConfigPayload
)

router = APIRouter(prefix="/promo")

# ENDPOINT 0: get metadata cluster from JSON file (for dynamic segment names in dropdown SHEET)
@router.get(
    "/metadata",
    response_model=StandardResponse[Dict[str, str]],
    summary="Get dynamic cluster segment metadata from JSON file"
)
async def get_promo_metadata():
    return await get_promo_metadata_logic()

# ENDPOINT 1: Create or Update promo config for a cluster (Upsert)
@router.post(
    "",
    response_model=StandardResponse[Dict[str, Any]],
    summary="Create or Update a Campaign Configuration"
)
async def create_promo(
    payload: PromoConfigPayload,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return await create_promo_config(payload, current_user, db)

# ENDPOINT 2: Get all active promo configs (for SHEET display)
@router.get(
    "/active",
    response_model=StandardResponse[list[Dict[str, Any]]],
    summary="Get all active promo configurations"
)
async def get_active_promos(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return await get_all_active_promos_logic(db)

# ENDPOINT 3: Get active promo config by cluster ID (for SHEET auto-fill)
@router.get(
    "/cluster/{cluster_id}",
    response_model=StandardResponse[Any],
    summary="Get active promo config by cluster ID"
)
async def get_promo_by_cluster(
    cluster_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return await get_promo_by_cluster_logic(cluster_id, db)

# ENDPOINT 4: Hapus promo config berdasarkan ID
@router.delete("/{promo_id}")
async def delete_promo_config(
    promo_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    return await delete_promo_config_logic(promo_id, db)
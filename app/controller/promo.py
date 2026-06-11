import json
import os
from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.schemas.base import StandardResponse
from app.models.promo_config import PromoConfig
from app.schemas.promo import PromoConfigPayload
from typing import Dict, List

def parse_allowed_promo_keys(promo_string: str) -> List[str]:
    mapping = {
        "kupon": "kupon",
        "voucher": "kupon",
        "cashback": "cashback",
        "buy one get one": "bogo",
        "bogo": "bogo",
        "price off": "price_off",
        "bonus packs": "bonus_packs",
        "sampling": "sampling"
    }
    allowed = []
    p_lower = promo_string.lower()
    for keyword, key in mapping.items():
        if keyword in p_lower:
            if key not in allowed:
                allowed.append(key)
    return allowed

def get_segment_name_from_json(cluster_id: int) -> str:
    try:
        path = "app/artifacts/metadata_segmentasi.json"
        if not os.path.exists(path):
            return f"Cluster {cluster_id}"
            
        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        pola_map = metadata.get("cluster_pola_map", {})
        segment_map = metadata.get("segment_map", {})
        
        pola = pola_map.get(str(cluster_id))
        if pola:
            return segment_map.get(pola, f"Cluster {cluster_id}")
        return f"Cluster {cluster_id}"
    except Exception as e:
        print(f"Error reading dynamic metadata JSON: {e}")
        return f"Cluster {cluster_id}"

def get_all_clusters_metadata() -> Dict[str, str]:
    try:
        path = "app/artifacts/metadata_segmentasi.json"
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        pola_map = metadata.get("cluster_pola_map", {})
        segment_map = metadata.get("segment_map", {})
        
        result = {}
        for cid, pola in pola_map.items():
            result[cid] = segment_map.get(pola, f"Cluster {cid}")
        return result
    except Exception:
        return {}

async def create_promo_config(payload: PromoConfigPayload, current_user: dict, db: Session):
    promo_type = payload.promo_type
    cluster_id = payload.cluster_id
    
    # Cari apakah sudah ada tipe promo yang sama di cluster tersebut
    existing = db.query(PromoConfig).filter(
        PromoConfig.cluster_id == cluster_id,
        PromoConfig.promo_type == promo_type
    ).first()

    if existing:
        # Jika ada, UPDATE
        existing.params = payload.params
        existing.active = payload.active
        existing.created_by = current_user.get("user_id")
        db.commit()
        db.refresh(existing)
        record = existing
    else:
        # Jika belum ada, INSERT baru
        record = PromoConfig(
            promo_type=promo_type,
            params=payload.params,
            active=payload.active,
            cluster_id=cluster_id,
            created_by=current_user.get("user_id"),
        )
        db.add(record)
        db.commit()
        db.refresh(record)

    return StandardResponse(
        code=200,
        error=False,
        message="Campaign configured successfully",
        data={
            "id": record.id,
            "promo_type": record.promo_type,
            "segment_name": get_segment_name_from_json(record.cluster_id),
            "params": record.params,
            "active": record.active,
            "created_by": record.created_by,
            "created_at": record.created_at.isoformat() if record.created_at else None,
        }
    )

async def get_promo_metadata_logic():
    metadata = get_all_clusters_metadata()
    return StandardResponse(code=200, error=False, message="Success", data=metadata)

# 3. Logic untuk mengambil SEMUA promo aktif (Untuk Halaman Dashboard)
async def get_all_active_promos_logic(db: Session):
    configs = db.query(PromoConfig).filter(PromoConfig.active == True).all()
    
    data_list = []
    for c in configs:
        data_list.append({
            "id": c.id, 
            "promo_type": c.promo_type, 
            "cluster_id": c.cluster_id, 
            "segment_name": get_segment_name_from_json(c.cluster_id),
            "params": c.params, 
            "active": c.active
        })
        
    return StandardResponse(
        code=200, 
        error=False, 
        message="Active campaigns retrieved successfully", 
        data=data_list
    )

# 4. Logic untuk mengambil 1 promo per cluster (Untuk Isi data otomatis di SHEET)
async def get_promo_by_cluster_logic(cluster_id: int, db: Session):
    config = db.query(PromoConfig).filter(
        PromoConfig.cluster_id == cluster_id, 
        PromoConfig.active == True
    ).first()
    
    if not config:
        return StandardResponse(code=200, error=False, message="No configuration found for this cluster", data=None)
        
    return StandardResponse(
        code=200, 
        error=False, 
        message="Cluster campaign retrieved successfully", 
        data={
            "id": config.id,
            "promo_type": config.promo_type,
            "cluster_id": config.cluster_id,
            "segment_name": get_segment_name_from_json(config.cluster_id),
            "params": config.params,
            "active": config.active
        }
    )
    
async def delete_promo_config_logic(promo_id: int, db: Session):
    config = db.query(PromoConfig).filter(PromoConfig.id == promo_id).first()
    
    if not config:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    db.delete(config)
    db.commit()
    
    return StandardResponse(
        code=200,
        error=False,
        message="Campaign deleted successfully",
        data={"id": promo_id}
    )

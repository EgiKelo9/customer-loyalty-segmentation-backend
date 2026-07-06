from typing import Any, Dict, List, Union
from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session
from app.controller.segmentation import (
	segment_from_file as segment_from_file_controller,
	segment_from_lrfm as segment_from_lrfm_controller,
	segment_from_transactions as segment_from_transactions_controller,
	get_segment_distribution as get_segment_distribution_controller,
	get_segmentation_history as get_segmentation_history_controller,
	get_segmentation_history_batches as get_segmentation_history_batches_controller,
 	get_segmentation_history_by_batch_id as get_segmentation_history_by_batch_id_controller,
)
from app.schemas.segmentation import BatchHistoryItem, BatchSegmentationResponse, CustomerInput, SegmentationResponse, TransactionInput, DistributionResponse, SegmentationHistoryItem
from app.schemas.base import StandardResponse
from app.shared.auth import get_current_user
from app.database.main import get_db

router = APIRouter(prefix="/segmentation")


@router.get(
	"/lrfm",
	response_model=StandardResponse[SegmentationResponse],
	responses={
		422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
		401: {"model": StandardResponse[dict], "description": "Unauthorized"}
	},
	summary="Segmentasi dari nilai LRFM",
)
@router.post(
	"/lrfm",
	response_model=StandardResponse[SegmentationResponse],
	responses={
		422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
		401: {"model": StandardResponse[dict], "description": "Unauthorized"}
	},
	summary="Segmentasi dari nilai LRFM",
)
async def segment_from_lrfm(
	customer: CustomerInput,
	current_user: dict = Depends(get_current_user),
	db: Session = Depends(get_db),
) -> StandardResponse[SegmentationResponse]:
	return await segment_from_lrfm_controller(customer, db, current_user)

@router.get(
	"/transactions",
	response_model=StandardResponse[SegmentationResponse],
	responses={
		422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
		401: {"model": StandardResponse[dict], "description": "Unauthorized"}
	},
	summary="Segmentasi dari transaksi JSON (1 pelanggan)",
)
@router.post(
	"/transactions",
	response_model=StandardResponse[SegmentationResponse],
	responses={
		422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
		401: {"model": StandardResponse[dict], "description": "Unauthorized"}
	},
	summary="Segmentasi dari transaksi JSON (1 pelanggan)",
)
async def segment_from_transactions(
	transactions: List[TransactionInput],
	current_user: dict = Depends(get_current_user),
	db: Session = Depends(get_db),
) -> StandardResponse[SegmentationResponse]:
	return await segment_from_transactions_controller(transactions, db, current_user)

@router.get(
	"/transactions/upload",
	response_model=StandardResponse[Union[SegmentationResponse, BatchSegmentationResponse]],
	responses={
		422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
		401: {"model": StandardResponse[dict], "description": "Unauthorized"}
	},
	summary="Segmentasi dari file CSV/Excel (1 atau banyak pelanggan)",
)
@router.post(
	"/transactions/upload",
	response_model=StandardResponse[Union[SegmentationResponse, BatchSegmentationResponse]],
	responses={
		422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
		401: {"model": StandardResponse[dict], "description": "Unauthorized"}
	},
	summary="Segmentasi dari file CSV/Excel (1 atau banyak pelanggan)",
)
async def segment_from_file(
	file: UploadFile = File(..., description="File CSV atau Excel berisi data transaksi"),
	current_user: dict = Depends(get_current_user),
	db: Session = Depends(get_db),
) -> StandardResponse[Union[SegmentationResponse, BatchSegmentationResponse]]:
	return await segment_from_file_controller(file, db, current_user)

@router.get(
    "/distribution",
    response_model=StandardResponse[DistributionResponse],
    responses={
        422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
        401: {"model": StandardResponse[dict], "description": "Unauthorized"}
    },
    summary="Mendapatkan data distribusi segmentasi (Scatter Data & Aggregate)",
)
async def get_segment_distribution(
	current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StandardResponse[DistributionResponse]:
	return await get_segment_distribution_controller(current_user, db)

@router.get(
	"/history",
	response_model=StandardResponse[List[SegmentationHistoryItem]],
	responses={
		401: {"model": StandardResponse[dict], "description": "Unauthorized"}
	},
	summary="Riwayat hasil segmentasi per user",
)
async def get_segmentation_history(
	limit: int = 50,
	current_user: dict = Depends(get_current_user),
	db: Session = Depends(get_db),
) -> StandardResponse[List[SegmentationHistoryItem]]:
	return await get_segmentation_history_controller(current_user, db, limit)

@router.get(
    "/history/batches",
    response_model=StandardResponse[List[BatchHistoryItem]],
    responses={
        401: {"model": StandardResponse[dict], "description": "Unauthorized"}
    },
    summary="Riwayat hasil segmentasi yang dikelompokkan per batch (upload/input)",
)
async def get_segmentation_history_batches(
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StandardResponse[List[BatchHistoryItem]]:
    return await get_segmentation_history_batches_controller(current_user, db, limit)

@router.get(
    "/history/batches/{batch_id}",
    response_model=StandardResponse[List[SegmentationHistoryItem]],
    summary="Ambil detail seluruh pelanggan di dalam 1 batch",
)
async def get_history_by_batch_id(
    batch_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StandardResponse[List[SegmentationHistoryItem]]:
    return await get_segmentation_history_by_batch_id_controller(batch_id, current_user, db)
from typing import Any, Dict, List, Union
from fastapi import APIRouter, Depends, File, UploadFile
from app.controller.segmentation import (
	segment_from_file as segment_from_file_controller,
	segment_from_lrfm as segment_from_lrfm_controller,
	segment_from_transactions as segment_from_transactions_controller,
)
from app.schemas.segmentation import BatchSegmentationResponse, CustomerInput, SegmentationResponse, TransactionInput
from app.schemas.base import StandardResponse
from app.shared.auth import get_current_user

router = APIRouter(prefix="/segmentation", dependencies=[Depends(get_current_user)])


@router.post(
	"/lrfm",
	response_model=StandardResponse[SegmentationResponse],
	responses={
		422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
		401: {"model": StandardResponse[dict], "description": "Unauthorized"}
	},
	summary="Segmentasi dari nilai LRFM",
)
async def segment_from_lrfm(customer: CustomerInput) -> StandardResponse[SegmentationResponse]:
	return await segment_from_lrfm_controller(customer)

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
) -> StandardResponse[SegmentationResponse]:
	return await segment_from_transactions_controller(transactions)

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
) -> StandardResponse[Union[SegmentationResponse, BatchSegmentationResponse]]:
	return await segment_from_file_controller(file)

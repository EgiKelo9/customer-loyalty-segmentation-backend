from typing import List, Union
from fastapi import APIRouter, Depends, File, UploadFile
from app.controller.predict import (
	predict_from_file as predict_from_file_controller,
	predict_from_lrfm as predict_from_lrfm_controller,
	predict_from_transactions as predict_from_transactions_controller,
)
from app.schemas.predict import BatchPredictionResponse, CustomerInput, PredictionResponse, TransactionInput
from app.shared.auth import get_current_user

router = APIRouter(prefix="/predict", dependencies=[Depends(get_current_user)])


@router.post(
	"/lrfm",
	response_model=PredictionResponse,
	summary="Prediksi dari nilai LRFM",
)
async def predict_from_lrfm(customer: CustomerInput) -> PredictionResponse:
	return await predict_from_lrfm_controller(customer)

@router.post(
	"/transactions",
	response_model=PredictionResponse,
	summary="Prediksi dari transaksi JSON (1 pelanggan)",
)
async def predict_from_transactions(
	transactions: List[TransactionInput],
) -> PredictionResponse:
	return await predict_from_transactions_controller(transactions)

@router.post(
	"/transactions/upload",
	response_model=Union[PredictionResponse, BatchPredictionResponse],
	summary="Prediksi dari file CSV/Excel (1 atau banyak pelanggan)",
)
async def predict_from_file(
	file: UploadFile = File(..., description="File CSV atau Excel berisi data transaksi"),
) -> Union[PredictionResponse, BatchPredictionResponse]:
	return await predict_from_file_controller(file)

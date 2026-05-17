import io
from typing import List, Union
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from app.schemas.predict import (
	BatchPredictionResponse,
	CustomerInput,
	LRFMCalculated,
	PredictionResponse,
	TransactionInput,
)
from app.pipeline.ml_service import predict_single
from app.pipeline.preprocessing import auto_map_columns, extract_lrfm
from app.shared.auth import get_current_user

router = APIRouter(prefix="/predict", dependencies=[Depends(get_current_user)])

def _build_prediction(row: pd.Series) -> PredictionResponse:
	result = predict_single(
		l=float(row["Length"]),
		r=float(row["Recency"]),
		f=float(row["Frequency"]),
		m=float(row["Monetary"]),
	)
	return PredictionResponse(
		customer_id=str(row.get("customer_id", "")),
		cluster=result["cluster"],
		pola=result["pola"],
		segmen=result["segmen"],
		rekomendasi=result["rekomendasi"],
		fuzzy_membership=result["fuzzy_membership"],
		lrfm_calculated=LRFMCalculated(
			L=float(row["Length"]),
			R=float(row["Recency"]),
			F=float(row["Frequency"]),
			M=float(row["Monetary"]),
		),
	)

def _parse_uploaded_file(file: UploadFile, contents: bytes) -> pd.DataFrame:
	if not file.filename:
		raise HTTPException(status_code=400, detail="Nama file tidak ditemukan.")

	ext = file.filename.split(".")[-1].lower()
	if ext == "csv":
		return pd.read_csv(io.BytesIO(contents))
	if ext in {"xlsx", "xls"}:
		return pd.read_excel(io.BytesIO(contents))

	raise HTTPException(
		status_code=400,
		detail=(
			f"Format file '{ext}' tidak didukung. "
			"Gunakan CSV atau Excel (.xlsx/.xls)."
		),
	)

@router.post(
	"/lrfm",
	response_model=PredictionResponse,
	summary="Prediksi dari nilai LRFM",
)
def predict_from_lrfm(customer: CustomerInput) -> PredictionResponse:
	try:
		result = predict_single(
			l=customer.L,
			r=customer.R,
			f=customer.F,
			m=customer.M,
		)
		return PredictionResponse(
			cluster=result["cluster"],
			pola=result["pola"],
			segmen=result["segmen"],
			rekomendasi=result["rekomendasi"],
			fuzzy_membership=result["fuzzy_membership"],
		)
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc

@router.post(
	"/transactions",
	response_model=PredictionResponse,
	summary="Prediksi dari transaksi JSON (1 pelanggan)",
)
def predict_from_transactions(
	transactions: List[TransactionInput],
) -> PredictionResponse:
	try:
		if not transactions:
			raise HTTPException(status_code=400, detail="Data transaksi tidak boleh kosong.")

		customer_ids = {t.customer_id for t in transactions}
		if len(customer_ids) > 1:
			raise HTTPException(
				status_code=400,
				detail=(
					"Endpoint ini hanya untuk 1 pelanggan. "
					f"Terdeteksi {len(customer_ids)} customer_id berbeda: {customer_ids}. "
					"Gunakan /predict/transactions/upload untuk banyak pelanggan."
				),
			)

		df_raw = pd.DataFrame([t.model_dump() for t in transactions])
		df_mapped = auto_map_columns(df_raw)
		df_lrfm = extract_lrfm(df_mapped)

		return _build_prediction(df_lrfm.iloc[0])
	except HTTPException:
		raise
	except ValueError as exc:
		raise HTTPException(status_code=422, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc

@router.post(
	"/transactions/upload",
	summary="Prediksi dari file CSV/Excel (1 atau banyak pelanggan)",
)
async def predict_from_file(
	file: UploadFile = File(..., description="File CSV atau Excel berisi data transaksi"),
) -> Union[PredictionResponse, BatchPredictionResponse]:
	try:
		contents = await file.read()
		df_raw = _parse_uploaded_file(file, contents)

		if df_raw.empty:
			raise HTTPException(status_code=400, detail="File kosong atau tidak bisa dibaca.")

		df_mapped = auto_map_columns(df_raw)
		df_lrfm = extract_lrfm(df_mapped)

		if len(df_lrfm) == 1:
			return _build_prediction(df_lrfm.iloc[0])

		results = [_build_prediction(row) for _, row in df_lrfm.iterrows()]
		return BatchPredictionResponse(
			status="success",
			total_pelanggan=len(results),
			data=results,
		)
	except HTTPException:
		raise
	except ValueError as exc:
		raise HTTPException(status_code=422, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc

import io
import pandas as pd
from typing import List, Union
from fastapi import HTTPException, UploadFile
from app.pipeline.ml_service import segment_single
from app.schemas.base import StandardResponse
from app.pipeline.preprocessing import auto_map_columns, extract_lrfm
from app.schemas.segmentation import (
	BatchSegmentationResponse,
	CustomerInput,
	LRFMCalculated,
	SegmentationResponse,
	TransactionInput,
)


def _build_segmentation_response(row: pd.Series) -> SegmentationResponse:
	result = segment_single(
		l=float(row["Length"]),
		r=float(row["Recency"]),
		f=float(row["Frequency"]),
		m=float(row["Monetary"]),
	)
	return SegmentationResponse(
		customer_id=str(row.get("customer_id", "")),
		cluster=result["cluster"],
		pattern=result["pattern"],
		segment=result["segment"],
		recommendation=result["recommendation"],
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


async def segment_from_lrfm(customer: CustomerInput) -> SegmentationResponse:
	try:
		result = segment_single(
			l=customer.L,
			r=customer.R,
			f=customer.F,
			m=customer.M,
		)
		return SegmentationResponse(
			cluster=result["cluster"],
			pattern=result["pattern"],
			segment=result["segment"],
			recommendation=result["recommendation"],
			fuzzy_membership=result["fuzzy_membership"],
		)
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


async def segment_from_transactions(
	transactions: List[TransactionInput],
) -> SegmentationResponse:
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

		segmentation_response = _build_segmentation_response(df_lrfm.iloc[0])
		return StandardResponse(
			code=200,
			error=False,
			message="Segmentation successful for single customer",
			data=segmentation_response,
		)
	except HTTPException:
		raise
	except ValueError as exc:
		raise HTTPException(status_code=422, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


async def segment_from_file(
	file: UploadFile,
) -> Union[SegmentationResponse, BatchSegmentationResponse]:
	try:
		contents = await file.read()
		df_raw = _parse_uploaded_file(file, contents)

		if df_raw.empty:
			raise HTTPException(status_code=400, detail="File kosong atau tidak bisa dibaca.")

		df_mapped = auto_map_columns(df_raw)
		df_lrfm = extract_lrfm(df_mapped)

		if len(df_lrfm) == 1:
			segmentation_response = _build_segmentation_response(df_lrfm.iloc[0])
			return StandardResponse(
				code=200,
				error=False,
				message="Segmentation successful for customer in uploaded file",
				data=segmentation_response,
			)

		results = [_build_segmentation_response(row) for _, row in df_lrfm.iterrows()]
		return StandardResponse(
			code=200,
			error=False,
			message="Segmentation successful for customers in uploaded file",
			data=BatchSegmentationResponse(
				status="success",
				total_customers=len(results),
				data=results,
			),
		)
	except HTTPException:
		raise
	except ValueError as exc:
		raise HTTPException(status_code=422, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc
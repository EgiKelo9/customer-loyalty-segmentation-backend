import io
import os
import uuid
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.pipeline.ml_service import segment_single
from app.schemas.base import StandardResponse
from app.pipeline.preprocessing import auto_map_columns, extract_lrfm
from app.schemas.segmentation import (
	BatchSegmentationResponse,
	CustomerInput,
	LRFMCalculated,
	RankedPromo,
	SegmentationResponse,
	SegmentationHistoryItem,
	TransactionInput,
	DistributionResponse,
	ClusterAggregated,
	ScatterDataPoint,
	BatchHistoryItem,
)
from app.shared.transaction_manager import TransactionManager
from app.models.segmentation_result import SegmentationResult

SEGMENTED_DATASET_PATH = os.path.join(os.getcwd(), "static", "dataset", "segmented_data.csv")


def _serialize_lrfm(lrfm: Optional[LRFMCalculated]) -> Optional[dict]:
	if not lrfm:
		return None
	return {
		"L": lrfm.L,
		"R": lrfm.R,
		"F": lrfm.F,
		"M": lrfm.M,
	}


def _persist_results(
    db: Session,
    user_id: int,
    results: List[SegmentationResponse],
    source: str,
    batch_id: str,
    lrfm_override: Optional[LRFMCalculated] = None,
) -> None:
	if db is None or user_id is None:
		return

	transaction_manager = TransactionManager(db)
	with transaction_manager.transaction() as session:
		for result in results:
			lrfm_data = _serialize_lrfm(lrfm_override or result.lrfm_calculated)
			session.add(
				SegmentationResult(
					user_id=user_id,
     				batch_id=batch_id,
					customer_id=result.customer_id,
					cluster=result.cluster,
					pattern=result.pattern,
					segment=result.segment,
					recommendation=result.recommendation,
					fuzzy_membership=result.fuzzy_membership,
					lrfm=lrfm_data,
					source=source,
				)
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


async def segment_from_lrfm(
	customer: CustomerInput,
	db: Optional[Session] = None,
	current_user: Optional[dict] = None,
) -> StandardResponse[SegmentationResponse]:
	try:
		result = segment_single(
			l=customer.L,
			r=customer.R,
			f=customer.F,
			m=customer.M,
		)
		segmentation_response = SegmentationResponse(
			cluster=result["cluster"],
			pattern=result["pattern"],
			segment=result["segment"],
			recommendation=result["recommendation"],
			fuzzy_membership=result["fuzzy_membership"],
			fuzzy_membership_raw=result["fuzzy_membership_raw"],
		)
		user_id = current_user.get("user_id") if current_user else None
		batch_id = str(uuid.uuid4())
		lrfm_override = LRFMCalculated(L=customer.L, R=customer.R, F=customer.F, M=customer.M)
		_persist_results(db, user_id, [segmentation_response], "lrfm", batch_id, lrfm_override)
		return StandardResponse(
			code=200,
			error=False,
			message="Segmentation successful for LRFM input",
			batch_id=batch_id,
			data=segmentation_response,
		)
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


async def segment_from_transactions(
	transactions: List[TransactionInput],
	db: Optional[Session] = None,
	current_user: Optional[dict] = None,
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
		user_id = current_user.get("user_id") if current_user else None
		batch_id = str(uuid.uuid4())
		_persist_results(db, user_id, [segmentation_response], "transactions", batch_id)
		return StandardResponse(
			code=200,
			error=False,
			message="Segmentation successful for single customer",
   			batch_id=batch_id,
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
	db: Optional[Session] = None,
	current_user: Optional[dict] = None,
) -> Union[SegmentationResponse, BatchSegmentationResponse]:
	try:
		contents = await file.read()
		df_raw = _parse_uploaded_file(file, contents)

		if df_raw.empty:
			raise HTTPException(status_code=400, detail="File kosong atau tidak bisa dibaca.")

		df_mapped = auto_map_columns(df_raw)
		df_lrfm = extract_lrfm(df_mapped)

		user_id = current_user.get("user_id") if current_user else None
		batch_id = str(uuid.uuid4())

		if len(df_lrfm) == 1:
			segmentation_response = _build_segmentation_response(df_lrfm.iloc[0])
			_persist_results(db, user_id, [segmentation_response], "file", batch_id)
			return StandardResponse(
				code=200,
				error=False,
				message="Segmentation successful for customer in uploaded file",
				data=segmentation_response,
			)

		results = [_build_segmentation_response(row) for _, row in df_lrfm.iterrows()]
		_persist_results(db, user_id, results, "file", batch_id)
		
		return StandardResponse(
			code=200,
			error=False,
			message="Segmentation successful for customers in uploaded file",
			data=BatchSegmentationResponse(
				status="success",
				total_customers=len(results),
    			batch_id=batch_id,
				data=results,
			),
		)
	except HTTPException:
		raise
	except ValueError as exc:
		raise HTTPException(status_code=422, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


async def get_segment_distribution() -> StandardResponse[DistributionResponse]:
	try:
		if not os.path.exists(SEGMENTED_DATASET_PATH):
			raise HTTPException(status_code=404, detail="Segmented dataset not found.")

		df = pd.read_csv(SEGMENTED_DATASET_PATH)
		if df.empty:
			raise HTTPException(status_code=400, detail="Segmented dataset is empty.")

		max_frequency_threshold = 10
		df = df[df["Frequency"] <= max_frequency_threshold]
		if df.empty:
			raise HTTPException(status_code=400, detail="No data available after filtering high frequency outliers.")

		agg_df = df.groupby(["Cluster", "Segment"]).agg({
			"customer_id": "count",
			"Recency": "mean",
			"Frequency": "mean",
			"Monetary": "mean",
		}).reset_index()

		cluster_colors = {
			0: "#ef4444",
			1: "#eab308",
			2: "#22c55e",
			3: "#3b82f6",
			4: "#a855f7",
		}

		segments = []
		for _, row in agg_df.iterrows():
			cluster_val = int(row["Cluster"])
			segments.append(
				ClusterAggregated(
					id=str(cluster_val),
					name=str(row["Segment"]),
					userCount=int(row["customer_id"]),
					avgRecency=float(row["Recency"]),
					avgFrequency=float(row["Frequency"]),
					avgMonetary=float(row["Monetary"]),
					color=cluster_colors.get(cluster_val, "#94a3b8"),
					description=(
						f"Customers belonging to the {row['Segment']} segment based on their transaction behavior."
					),
				)
			)

		all_segment = ClusterAggregated(
			id="all",
			name="All Customers",
			userCount=len(df),
			avgRecency=float(df["Recency"].mean()),
			avgFrequency=float(df["Frequency"].mean()),
			avgMonetary=float(df["Monetary"].mean()),
			color="#18181b",
			description="Global overview containing all segmented customers.",
		)

		all_segment_data = [all_segment] + segments
		target_sample_size = 1000
		total_rows = len(df)

		if total_rows <= target_sample_size:
			df_sampled = df.copy()
		else:
			fraction = target_sample_size / total_rows
			df_sampled = df.groupby("Cluster", group_keys=False).apply(
				lambda x: x.sample(frac=fraction, random_state=42)
			).copy()

		jitter_r = np.random.uniform(-0.4, 0.4, size=len(df_sampled))
		jitter_f = np.random.uniform(-0.3, 0.3, size=len(df_sampled))

		scatter_data = []
		for i, (_, row) in enumerate(df_sampled.iterrows()):
			jittered_recency = max(0.1, float(row["Recency"]) + jitter_r[i])
			jittered_frequency = max(0.1, float(row["Frequency"]) + jitter_f[i])

			scatter_data.append(
				ScatterDataPoint(
					customer_id=str(row["customer_id"]),
					recency=jittered_recency,
					frequency=jittered_frequency,
					monetary=float(row["Monetary"]),
					clusterId=str(row["Cluster"]),
				)
			)

		return StandardResponse(
			code=200,
			error=False,
			message="Distribution fetched successfully",
			data=DistributionResponse(
				segments=segments,
				allSegmentData=all_segment_data,
				scatterData=scatter_data,
			),
		)
	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


async def get_segmentation_history(
	current_user: dict,
	db: Session,
	limit: int = 50,
) -> StandardResponse[List[SegmentationHistoryItem]]:
	user_id = current_user.get("user_id")
	if not user_id:
		raise HTTPException(status_code=401, detail="Invalid token payload")

	query = (
		db.query(SegmentationResult)
		.filter(SegmentationResult.user_id == user_id)
		.order_by(SegmentationResult.created_at.desc())
		.limit(limit)
	)

	items = [
		SegmentationHistoryItem(
			id=item.id,
			customer_id=item.customer_id,
			cluster=item.cluster,
			pattern=item.pattern,
			segment=item.segment,
			recommendation=item.recommendation,
			fuzzy_membership=item.fuzzy_membership,
			lrfm_calculated=(
				LRFMCalculated(**item.lrfm) if item.lrfm else None
			),
			source=item.source,
			created_at=item.created_at,
		)
		for item in query.all()
	]

	return StandardResponse(
		code=200,
		error=False,
		message="Segmentation history fetched successfully",
		data=items,
	)

async def get_segmentation_history_batches(
    current_user: dict, db: Session, limit: int = 50
) -> StandardResponse[List[BatchHistoryItem]]:
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    query = (
        db.query(
            SegmentationResult.batch_id,
            SegmentationResult.source,
            func.count(SegmentationResult.id).label("total_customers"),
            func.min(SegmentationResult.created_at).label("created_at")
        )
        .filter(SegmentationResult.user_id == user_id)
        .filter(SegmentationResult.batch_id.isnot(None))
        .group_by(SegmentationResult.batch_id, SegmentationResult.source)
        .order_by(func.min(SegmentationResult.created_at).desc())
        .limit(limit)
    )

    items = [
        BatchHistoryItem(
            batch_id=row.batch_id,
            source=row.source,
            total_customers=row.total_customers,
            created_at=row.created_at
        ) for row in query.all()
    ]

    return StandardResponse(
        code=200, error=False, message="Batched history fetched successfully", data=items
    )
    
async def get_segmentation_history_by_batch_id(
    batch_id: str, current_user: dict, db: Session
) -> StandardResponse[List[SegmentationHistoryItem]]:
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    query = (
        db.query(SegmentationResult)
        .filter(SegmentationResult.user_id == user_id)
        .filter(SegmentationResult.batch_id == batch_id)
        .order_by(SegmentationResult.created_at.desc())
    )
    
    items = [
        SegmentationHistoryItem(
            id=item.id,
            customer_id=item.customer_id,
            cluster=item.cluster,
            pattern=item.pattern,
            segment=item.segment,
            recommendation=item.recommendation,
            fuzzy_membership=item.fuzzy_membership,
            lrfm_calculated=(LRFMCalculated(**item.lrfm) if item.lrfm else None),
            source=item.source,
            created_at=item.created_at,
        ) for item in query.all()
    ]
    return StandardResponse(code=200, error=False, message="Batch detail fetched successfully", data=items)
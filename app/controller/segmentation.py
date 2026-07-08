import io
import os
import uuid
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from datetime import date, datetime, timezone
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
    transaction_dates: Optional[List[Optional[date]]] = None,
) -> None:
    if db is None or user_id is None:
        return
 
    transaction_manager = TransactionManager(db)
    with transaction_manager.transaction() as session:
        for i, result in enumerate(results):
            lrfm_data = _serialize_lrfm(lrfm_override or result.lrfm_calculated)
 
            tx_date = None
            if transaction_dates is not None and i < len(transaction_dates):
                tx_date = transaction_dates[i]
 
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
                    transaction_date=tx_date,
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
  
		tx_date = None
		date_col = next((c for c in df_mapped.columns if "date" in c.lower()), None)
		if date_col:
			parsed = pd.to_datetime(df_mapped[date_col], errors="coerce")
			if not parsed.isna().all():
				tx_date = parsed.max().date()

		_persist_results(
			db, user_id, [segmentation_response], "transactions", batch_id,
			transaction_dates=[tx_date],
		)

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

		date_col = next((c for c in df_mapped.columns if "date" in c.lower()), None)

		if date_col and "customer_id" in df_mapped.columns:
			df_mapped["_parsed_date"] = pd.to_datetime(df_mapped[date_col], format='mixed', errors="coerce")
			date_per_cust = df_mapped.groupby("customer_id")["_parsed_date"].max()
			tx_dates = [
				date_per_cust.get(str(row.get("customer_id")), pd.NaT)
				for _, row in df_lrfm.iterrows()
			]
			tx_dates = [d.date() if pd.notna(d) else None for d in tx_dates]
		elif date_col:
			# tidak ada customer_id, pakai max global
			parsed = pd.to_datetime(df_mapped[date_col], errors="coerce")
			global_max = parsed.max()
			tx_dates = [global_max.date() if pd.notna(global_max) else None] * len(df_lrfm)
		else:
			tx_dates = [None] * len(df_lrfm)

		results = [_build_segmentation_response(row) for _, row in df_lrfm.iterrows()]
		_persist_results(db, user_id, results, "file", batch_id, transaction_dates=tx_dates)
		
		if len(results) == 1:
			segmentation_response = results[0]
			segmentation_response.batch_id = batch_id
			return StandardResponse(
				code=200,
				error=False,
				message="Segmentation successful for customer in uploaded file",
				data=segmentation_response,
			)

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


async def get_segment_distribution(
    current_user: dict,
    db: Session
) -> StandardResponse[DistributionResponse]:
	try:
		# 1. Load static dataset
		df_static = pd.DataFrame()
		if os.path.exists(SEGMENTED_DATASET_PATH):
			df_static = pd.read_csv(SEGMENTED_DATASET_PATH)
			df_static["customer_id"] = df_static["customer_id"].astype(str)

		# 2. Fetch dynamic inference dataset from DB
		user_id = current_user.get("user_id")
		db_results = db.query(SegmentationResult).filter(SegmentationResult.user_id == user_id).all()
		
		db_rows = []
		for item in db_results:
			if item.lrfm:
				db_rows.append({
					"customer_id": str(item.customer_id) if item.customer_id else f"db-anon-{item.id}",
					"Length": float(item.lrfm.get("L", 0)),
					"Recency": float(item.lrfm.get("R", 0)),
					"Frequency": float(item.lrfm.get("F", 0)),
					"Monetary": float(item.lrfm.get("M", 0)),
					"Cluster": int(item.cluster),
					"Segment": str(item.segment)
				})
		df_db = pd.DataFrame(db_rows)
		if not df_db.empty:
			df_db["customer_id"] = df_db["customer_id"].astype(str)

		# 3. Combine for ACCURATE TABLE METRICS (All 400k+ data points)
		if not df_static.empty and not df_db.empty:
			df_combined = pd.concat([df_db, df_static], ignore_index=True)
			df_combined = df_combined.drop_duplicates(subset=["customer_id"], keep="first")
		elif not df_db.empty:
			df_combined = df_db
		elif not df_static.empty:
			df_combined = df_static
		else:
			raise HTTPException(status_code=400, detail="No data available for distribution.")

		max_frequency_threshold = 10
		df_full = df_combined[df_combined["Frequency"] <= max_frequency_threshold]
		
		if df_full.empty:
			raise HTTPException(status_code=400, detail="No data available after filtering high frequency outliers.")

		# 4. Calculate aggregate metrics per segment
		agg_df = df_full.groupby(["Cluster", "Segment"]).agg({
			"customer_id": "count",
			"Recency": "mean",
			"Frequency": "mean",
			"Monetary": "mean",
		}).reset_index()

		cluster_colors = {
			0: "#ef4444",  # Uncertain Lost Customers
			1: "#06b6d4",  # Platinum Customers
			2: "#f97316",  # Dormant Lost Customers
			3: "#22c55e",  # High Value Loyal Customers
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
					description=f"Customers belonging to the {row['Segment']} segment based on their transaction behavior."
				)
			)

		all_segment = ClusterAggregated(
			id="all",
			name="All Customers",
			userCount=len(df_full),
			avgRecency=float(df_full["Recency"].mean()),
			avgFrequency=float(df_full["Frequency"].mean()),
			avgMonetary=float(df_full["Monetary"].mean()),
			color="#18181b",
			description="Global overview containing all segmented customers."
		)

		all_segment_data = [all_segment] + segments

		# 5. Prepare scatter data with sampling to max 1000 points for performance
		target_sample_size = 1000
		
		# Filter the separate datasets
		df_db_filtered = df_db[df_db["Frequency"] <= max_frequency_threshold] if not df_db.empty else pd.DataFrame()
		df_static_filtered = df_static[df_static["Frequency"] <= max_frequency_threshold] if not df_static.empty else pd.DataFrame()
		
		# Remove DB duplicates from static pool
		if not df_db_filtered.empty and not df_static_filtered.empty:
			df_static_filtered = df_static_filtered[~df_static_filtered["customer_id"].isin(df_db_filtered["customer_id"])]

		# Rule 1: Always include 100% of the live inferred DB data
		df_scatter_parts = [df_db_filtered] if not df_db_filtered.empty else []
		current_db_count = len(df_db_filtered)

		# Rule 2: Fill the remaining slots with random data from the 400k static set
		remaining_slots = max(0, target_sample_size - current_db_count)
		
		if remaining_slots > 0 and not df_static_filtered.empty:
			if len(df_static_filtered) <= remaining_slots:
				df_scatter_parts.append(df_static_filtered.copy())
			else:
				fraction = remaining_slots / len(df_static_filtered)
				df_static_sampled = df_static_filtered.groupby("Cluster", group_keys=False).apply(
					lambda x: x.sample(frac=fraction, random_state=42)
				).copy()
				df_scatter_parts.append(df_static_sampled)

		# Combine them
		df_scatter = pd.concat(df_scatter_parts, ignore_index=True) if df_scatter_parts else pd.DataFrame()

		jitter_r = np.random.uniform(-0.4, 0.4, size=len(df_scatter))
		jitter_f = np.random.uniform(-0.3, 0.3, size=len(df_scatter))

		scatter_data = []
		for i, (_, row) in enumerate(df_scatter.iterrows()):
			scatter_data.append(ScatterDataPoint(
				customer_id=str(row["customer_id"]),
				recency=max(0.1, float(row["Recency"]) + jitter_r[i]),
				frequency=max(0.1, float(row["Frequency"]) + jitter_f[i]),
				monetary=float(row["Monetary"]),
				clusterId=str(row["Cluster"]),
			))

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
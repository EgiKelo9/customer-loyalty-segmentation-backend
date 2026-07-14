import io
import os
import uuid
import json
import numpy as np
import pandas as pd
import gc
from typing import List, Optional, Union
from datetime import date, datetime, timezone
from fastapi import HTTPException, UploadFile, BackgroundTasks
from app.database.main import SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.pipeline.ml_service import segment_single, segment_batch
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
CHUNK_SIZE = 50_000               # rows per chunk for background / distribution
MAX_SCATTER_POINTS = 1000         # max points shown in scatter plot
DIRECT_PROCESS_CUSTOMERS = 500    # threshold: ≤ this → inline, > → background


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

def process_file_background(
    contents: bytes, 
    filename: str, 
    mapping_str: str, 
    batch_id: str, 
    user_id: Optional[int]
):
    """
    Fungsi worker ini berjalan di background thread setelah response HTTP dikembalikan.
    Menggunakan Vectorized Batch Prediction dan Bulk Insert untuk performa maksimal.
    """
    db = SessionLocal()
    try:
        # 1. Parse File
        ext = filename.split(".")[-1].lower()
        if ext == "csv":
            df_raw = pd.read_csv(io.BytesIO(contents), low_memory=False)
        elif ext in ("xlsx", "xls"):
            df_raw = pd.read_excel(io.BytesIO(contents))
        else:
            print(f"Format tidak didukung: {ext}")
            return

        # 2. Proses Mapping dari Frontend
        if mapping_str:
            mapping_dict = json.loads(mapping_str)
            rename_rules = {v: k for k, v in mapping_dict.items() if v}
            df_raw.rename(columns=rename_rules, inplace=True)
        else:
            df_raw = auto_map_columns(df_raw)

        # 3. Compute max transaction date per customer
        date_col = next((c for c in df_raw.columns if "date" in c.lower()), None)
        if date_col:
            df_raw["_parsed_date"] = pd.to_datetime(
                df_raw[date_col], format="mixed", errors="coerce"
            )
            date_per_cust = df_raw.groupby("customer_id")["_parsed_date"].max()
        else:
            date_per_cust = pd.Series(dtype="datetime64[ns]")

        # 4. Build LRFM and free raw data
        df_lrfm = extract_lrfm(df_raw)
        del df_raw
        gc.collect()

        # Attach transaction dates
        df_lrfm["transaction_date"] = df_lrfm.index.map(date_per_cust)
        df_lrfm["transaction_date"] = df_lrfm["transaction_date"].apply(
            lambda d: d.date() if pd.notna(d) else None
        )

        # 5. Chunked inference + DB insert
        total_rows = len(df_lrfm)
        for start in range(0, total_rows, CHUNK_SIZE):
            chunk = df_lrfm.iloc[start : start + CHUNK_SIZE].copy()
            df_results = segment_batch(chunk)

            db_records = []
            for idx, row in df_results.iterrows():
                cust_id = str(idx) if idx else f"anon-{start}"
                tx_date = row.get("transaction_date")
                if pd.isna(tx_date):
                    tx_date = None

                lrfm_dict = {
                    "L": float(row["Length"]),
                    "R": float(row["Recency"]),
                    "F": float(row["Frequency"]),
                    "M": float(row["Monetary"]),
                }

                db_records.append(
                    SegmentationResult(
                        user_id=user_id,
                        batch_id=batch_id,
                        customer_id=cust_id,
                        cluster=int(row["cluster"]),
                        pattern=str(row["pattern"]),
                        segment=str(row["segment"]),
                        recommendation=str(row["recommendation"]),
                        fuzzy_membership=row["fuzzy_membership"],
                        lrfm=lrfm_dict,
                        source="file",
                        transaction_date=tx_date,
                    )
                )

            db.bulk_save_objects(db_records)
            db.commit()
            del db_records, df_results, chunk
            gc.collect()

        print(f"Background batch {batch_id} finished successfully.")

    except Exception as exc:
        print(f"Error in background task {batch_id}: {exc}")
        db.rollback()
    finally:
        db.close()


# ---------------------------------------------------------------------
#  Inline processing for small files (returns results immediately)
# ---------------------------------------------------------------------
def _process_file_inline(
    contents: bytes,
    filename: str,
    mapping_str: str,
    batch_id: str,
    user_id: Optional[int],
    db: Session,
) -> Union[SegmentationResponse, BatchSegmentationResponse]:
    """Process file synchronously, store in DB, and return segmentation data."""
    ext = filename.split(".")[-1].lower()
    if ext == "csv":
        df_raw = pd.read_csv(io.BytesIO(contents), low_memory=False)
    elif ext in ("xlsx", "xls"):
        df_raw = pd.read_excel(io.BytesIO(contents))
    else:
        raise HTTPException(status_code=400, detail=f"Format {ext} tidak didukung")

    # Mapping
    if mapping_str:
        mapping_dict = json.loads(mapping_str)
        rename_rules = {v: k for k, v in mapping_dict.items() if v}
        df_raw.rename(columns=rename_rules, inplace=True)
    else:
        df_raw = auto_map_columns(df_raw)

    # Extract LRFM
    df_lrfm = extract_lrfm(df_raw)
    del df_raw; gc.collect()

    # Single customer?
    if len(df_lrfm) == 1:
        row = df_lrfm.iloc[0]
        result = segment_single(
            l=float(row["Length"]),
            r=float(row["Recency"]),
            f=float(row["Frequency"]),
            m=float(row["Monetary"]),
        )
        seg_resp = SegmentationResponse(
            customer_id=str(row.get("customer_id", row.name)),
            cluster=result["cluster"],
            pattern=result["pattern"],
            segment=result["segment"],
            recommendation=result["recommendation"],
            fuzzy_membership=result["fuzzy_membership"],
            lrfm_calculated=LRFMCalculated(
                L=float(row["Length"]), R=float(row["Recency"]),
                F=float(row["Frequency"]), M=float(row["Monetary"]),
            ),
        )
        _persist_results(db, user_id, [seg_resp], "file", batch_id)
        return seg_resp

    # Batch inference
    df_results = segment_batch(df_lrfm)
    responses = []
    for idx, row in df_results.iterrows():
        cust_id = str(idx) if idx else f"anon-{batch_id}"
        seg = SegmentationResponse(
            customer_id=cust_id,
            cluster=int(row["cluster"]),
            pattern=str(row["pattern"]),
            segment=str(row["segment"]),
            recommendation=str(row["recommendation"]),
            fuzzy_membership=row["fuzzy_membership"],
            lrfm_calculated=LRFMCalculated(
                L=float(row["Length"]), R=float(row["Recency"]),
                F=float(row["Frequency"]), M=float(row["Monetary"]),
            ),
        )
        responses.append(seg)

    _persist_results(db, user_id, responses, "file", batch_id)
    return BatchSegmentationResponse(
        status="success",
        total_customers=len(responses),
        batch_id=batch_id,
        data=responses,
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
	mapping: Optional[str] = None, 
    background_tasks: BackgroundTasks = None,
    db: Optional[Session] = None,
	current_user: Optional[dict] = None,
) -> StandardResponse[dict]:
    try:
        contents = await file.read()
        
        user_id = current_user.get("user_id") if current_user else None
        batch_id = str(uuid.uuid4())
        ext = file.filename.split(".")[-1].lower()

        # Quick estimation of customer count – read first chunks
        if ext == "csv":
            # Use chunks to count rows without loading entire file into memory
            reader = pd.read_csv(io.BytesIO(contents), chunksize=1000)
            total_rows = 0
            for chunk in reader:
                total_rows += len(chunk)
                if total_rows > DIRECT_PROCESS_CUSTOMERS:
                    break
        else:
            # For Excel we cannot estimate easily; treat as small file (or you could default to background)
            # Here we'll fallback to inline; if it's huge it may OOM – adjust threshold accordingly.
            total_rows = 0  # will force inline; you may want to add a size check

        if total_rows <= DIRECT_PROCESS_CUSTOMERS:
            # Process immediately
            result = _process_file_inline(contents, file.filename, mapping, batch_id, user_id, db)
            return StandardResponse(
                code=200,
                error=False,
                message="Segmentasi selesai",
                data=result.model_dump() if hasattr(result, "model_dump") else result,
            )

        # Large file → background
        if background_tasks:
            background_tasks.add_task(
                process_file_background,
                contents=contents,
                filename=file.filename,
                mapping_str=mapping,
                batch_id=batch_id,
                user_id=user_id,
            )

        # Langsung balas sukses tanpa perlu nunggu komputasi selesai
        return StandardResponse(
            code=202,
            error=False,
            message="Data file sebesar itu diterima! Sistem sedang memproses segmen pelanggan di latar belakang.",
            data={
                "batch_id": batch_id,
                "status": "processing"
            }
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def get_segment_distribution(
    current_user: dict,
    db: Session,
) -> StandardResponse[DistributionResponse]:
    try:
        user_id = current_user.get("user_id")

        # 1. Load DB inferences (small)
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
                    "Segment": str(item.segment),
                })
        df_db = pd.DataFrame(db_rows) if db_rows else pd.DataFrame(
            columns=["customer_id", "Length", "Recency", "Frequency", "Monetary", "Cluster", "Segment"]
        )
        if not df_db.empty:
            df_db["customer_id"] = df_db["customer_id"].astype("string")
            for col in ["Length", "Recency", "Frequency", "Monetary"]:
                df_db[col] = pd.to_numeric(df_db[col], downcast="float")
            df_db["Cluster"] = df_db["Cluster"].astype("int8")

        # 2. Read static CSV in chunks, accumulate aggregates and reservoir sample
        agg_dict = {}
        total_count = 0
        sumR = sumF = sumM = 0.0
        reservoir = []
        rng = np.random.default_rng(42)

        if not os.path.exists(SEGMENTED_DATASET_PATH):
            # No static data? Just use DB
            df_static_empty = pd.DataFrame()
        else:
            csv_sample = pd.read_csv(SEGMENTED_DATASET_PATH, nrows=1)
            has_cluster = "Cluster" in csv_sample.columns
            dtype_dict = {
                "customer_id": "string",
                "Segment": "category",
                "Frequency": "float32",
                "Monetary": "float32",
                "Length": "float32",
                "Recency": "float32",
            }
            if has_cluster:
                dtype_dict["Cluster"] = "Int8"

            reader = pd.read_csv(
                SEGMENTED_DATASET_PATH,
                dtype=dtype_dict,
                chunksize=CHUNK_SIZE,
                low_memory=False,
            )

            for chunk in reader:
                chunk = chunk[chunk["Frequency"] <= 10]
                if not df_db.empty:
                    chunk = chunk[~chunk["customer_id"].isin(df_db["customer_id"])]
                if chunk.empty:
                    continue

                if not has_cluster:
                    chunk["Cluster"] = 0

                for (cluster, seg), grp in chunk.groupby(["Cluster", "Segment"]):
                    cnt = len(grp)
                    key = (int(cluster), str(seg))
                    if key not in agg_dict:
                        agg_dict[key] = {"count": 0, "sumR": 0.0, "sumF": 0.0, "sumM": 0.0}
                    agg_dict[key]["count"] += cnt
                    agg_dict[key]["sumR"] += grp["Recency"].sum()
                    agg_dict[key]["sumF"] += grp["Frequency"].sum()
                    agg_dict[key]["sumM"] += grp["Monetary"].sum()

                total_count += len(chunk)
                sumR += chunk["Recency"].sum()
                sumF += chunk["Frequency"].sum()
                sumM += chunk["Monetary"].sum()

                # Reservoir sampling
                for _, row in chunk.iterrows():
                    row_dict = row.to_dict()
                    if not has_cluster:
                        row_dict["Cluster"] = 0
                    if len(reservoir) < MAX_SCATTER_POINTS:
                        reservoir.append(row_dict)
                    else:
                        j = rng.integers(0, total_count)
                        if j < MAX_SCATTER_POINTS:
                            reservoir[j] = row_dict

        # 3. Add DB data to aggregates (DB data is small)
        if not df_db.empty:
            df_db_filtered = df_db[df_db["Frequency"] <= 10]
            for (cluster, seg), grp in df_db_filtered.groupby(["Cluster", "Segment"]):
                cnt = len(grp)
                key = (int(cluster), str(seg))
                if key not in agg_dict:
                    agg_dict[key] = {"count": 0, "sumR": 0.0, "sumF": 0.0, "sumM": 0.0}
                agg_dict[key]["count"] += cnt
                agg_dict[key]["sumR"] += grp["Recency"].sum()
                agg_dict[key]["sumF"] += grp["Frequency"].sum()
                agg_dict[key]["sumM"] += grp["Monetary"].sum()

            total_count += len(df_db_filtered)
            sumR += df_db_filtered["Recency"].sum()
            sumF += df_db_filtered["Frequency"].sum()
            sumM += df_db_filtered["Monetary"].sum()

            for _, row in df_db_filtered.iterrows():
                if len(reservoir) < MAX_SCATTER_POINTS:
                    reservoir.append(row.to_dict())
                else:
                    j = rng.integers(0, MAX_SCATTER_POINTS)
                    reservoir[j] = row.to_dict()

        if total_count == 0:
            raise HTTPException(status_code=400, detail="No data available for distribution.")

        # 4. Build segment summaries
        cluster_colors = {0: "#ef4444", 1: "#06b6d4", 2: "#f97316", 3: "#22c55e"}
        segments = []
        for (cluster, seg), vals in agg_dict.items():
            cnt = vals["count"]
            segments.append(ClusterAggregated(
                id=str(cluster),
                name=seg,
                userCount=cnt,
                avgRecency=round(vals["sumR"] / cnt, 2),
                avgFrequency=round(vals["sumF"] / cnt, 2),
                avgMonetary=round(vals["sumM"] / cnt, 2),
                color=cluster_colors.get(cluster, "#94a3b8"),
                description=f"Customers in the {seg} segment.",
            ))

        all_segment = ClusterAggregated(
            id="all", name="All Customers",
            userCount=total_count,
            avgRecency=round(sumR / total_count, 2),
            avgFrequency=round(sumF / total_count, 2),
            avgMonetary=round(sumM / total_count, 2),
            color="#18181b",
            description="Global overview of all segmented customers.",
        )
        all_segment_data = [all_segment] + segments

        # 5. Build scatter data from reservoir (with jitter)
        scatter_data = []
        jitter_r = np.random.uniform(-0.4, 0.4, size=len(reservoir))
        jitter_f = np.random.uniform(-0.3, 0.3, size=len(reservoir))
        for i, row in enumerate(reservoir):
            scatter_data.append(ScatterDataPoint(
                customer_id=str(row["customer_id"]),
                recency=max(0.1, float(row["Recency"]) + jitter_r[i]),
                frequency=max(0.1, float(row["Frequency"]) + jitter_f[i]),
                monetary=float(row["Monetary"]),
                clusterId=str(row.get("Cluster", "")),
            ))

        return StandardResponse(
            code=200, error=False,
            message="Distribution fetched successfully",
            data=DistributionResponse(segments=segments, allSegmentData=all_segment_data, scatterData=scatter_data),
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
    batch_id: str, 
    current_user: dict, 
    db: Session,
    limit: int = 500,
    skip: int = 0
) -> StandardResponse[List[SegmentationHistoryItem]]:
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    query = (
        db.query(SegmentationResult)
        .filter(SegmentationResult.user_id == user_id)
        .filter(SegmentationResult.batch_id == batch_id)
        .order_by(SegmentationResult.created_at.desc())
        .offset(skip)
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
            lrfm_calculated=(LRFMCalculated(**item.lrfm) if item.lrfm else None),
            source=item.source,
            created_at=item.created_at,
        ) for item in query.all()
    ]
    return StandardResponse(code=200, error=False, message="Batch detail fetched successfully", data=items)

async def get_customer_detail_in_batch_controller(
    batch_id: str, customer_id: str, current_user: dict, db: Session
) -> StandardResponse[SegmentationHistoryItem]:
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Cari spesifik 1 pelanggan di dalam batch tersebut
    item = (
        db.query(SegmentationResult)
        .filter(SegmentationResult.user_id == user_id)
        .filter(SegmentationResult.batch_id == batch_id)
        .filter(SegmentationResult.customer_id == customer_id)
        .first()
    )
    
    if not item:
        raise HTTPException(status_code=404, detail="Pelanggan tidak ditemukan di dalam batch ini.")
        
    result = SegmentationHistoryItem(
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
    )
    
    return StandardResponse(code=200, error=False, message="Detail fetched successfully", data=result)
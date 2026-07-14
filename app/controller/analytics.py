import os
import math
import pandas as pd
from sqlalchemy import select, func, cast, Float, distinct, text
from sqlalchemy.dialects.postgresql import JSONB
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Literal, Optional
from fastapi import HTTPException
from app.schemas.base import StandardResponse
from app.schemas.analytics import (
    CustomerChartData,
    CustomerData,
    KeyPerformanceIndicator,
    KPIResponse,
    ChartDataResponse,
    CustomerDataResponse,
    PaginationMetadata,
    SegmentTrendResponse
)
from sqlalchemy.orm import Session
from app.models.segmentation_result import SegmentationResult

RAW_DATASET_PATH = os.path.join(os.getcwd(), "static", "dataset", "raw_data.csv")
SEGMENTED_DATASET_PATH = os.path.join(os.getcwd(), "static", "dataset", "segmented_data.csv")

_STATIC_SEGMENTED_DF: Optional[pd.DataFrame] = None
_STATIC_RAW_DF: Optional[pd.DataFrame] = None

def _get_static_segmented_df() -> pd.DataFrame:
    global _STATIC_SEGMENTED_DF
    if _STATIC_SEGMENTED_DF is None:
        if os.path.exists(SEGMENTED_DATASET_PATH):
            df = pd.read_csv(SEGMENTED_DATASET_PATH)
            if not df.empty:
                df["customer_id"] = df["customer_id"].astype(str)
                # Bersihkan NaN sekali di sini, bukan berulang kali di tiap fungsi
                for col in ["Frequency", "Monetary", "Length", "Recency"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                if "Segment" in df.columns:
                    df["Segment"] = df["Segment"].fillna("Unknown").astype(str)
            _STATIC_SEGMENTED_DF = df
        else:
            _STATIC_SEGMENTED_DF = pd.DataFrame(
                columns=["customer_id", "Segment", "Frequency", "Monetary", "Length", "Recency"]
            )
    return _STATIC_SEGMENTED_DF.copy()


def _get_static_raw_df() -> pd.DataFrame:
    global _STATIC_RAW_DF
    if _STATIC_RAW_DF is None:
        _STATIC_RAW_DF = _load_dataset(RAW_DATASET_PATH)
    return _STATIC_RAW_DF.copy()


def invalidate_static_cache():
    """Panggil ini kalau file CSV statis diganti/diupdate saat runtime."""
    global _STATIC_SEGMENTED_DF, _STATIC_RAW_DF
    _STATIC_SEGMENTED_DF = None
    _STATIC_RAW_DF = None


def _load_dataset(dataset_path: str) -> pd.DataFrame:
    """Helper untuk membaca file dataset statis."""
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset tidak ditemukan di path: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membaca dataset: {str(e)}")

def calculate_trend(current_value: float, previous_value: float) -> float:
    """Helper untuk menghitung persentase tren."""
    if previous_value == 0:
        return 100.0 if current_value > 0 else 0.0
    return round(((current_value - previous_value) / previous_value) * 100, 2)

def _prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Helper untuk menyiapkan data transaksi dengan kolom yang diperlukan."""
    required_columns = {"user_ID", "order_ID", "order_date", "quantity", "final_unit_price"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise HTTPException(status_code=422, detail=f"Kolom transaksi yang hilang: {', '.join(missing)}")
    
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce").dt.date
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["final_unit_price"] = pd.to_numeric(df["final_unit_price"], errors="coerce").fillna(0)
    df["revenue"] = df["quantity"] * df["final_unit_price"]
    df = df.dropna(subset=["order_date", "user_ID", "order_ID"])
    return df


async def get_kpis(db: Session = None, current_user: dict = None) -> StandardResponse[KPIResponse]:
    try:
        # 1. Ambil Inferences Dinamis dari Database
        df_db = pd.DataFrame()
        if db and current_user:
            user_id = current_user.get("user_id")
            if user_id:
                sql = text("""
                    SELECT customer_id, segment, (lrfm->>'M')::float AS monetary
                    FROM (
                        SELECT
                            customer_id,
                            segment,
                            lrfm,
                            ROW_NUMBER() OVER (
                                PARTITION BY customer_id
                                ORDER BY created_at DESC
                            ) AS rn
                        FROM segmentation_results
                        WHERE user_id = :user_id
                    ) sub
                    WHERE rn = 1
                """)
                rows = db.execute(sql, {"user_id": user_id}).all()
                if rows:
                    df_db = pd.DataFrame(
                        rows, columns=["customer_id", "Segment", "Monetary"]
                    )
                    df_db["customer_id"] = df_db["customer_id"].astype(str)
                    df_db["Monetary"] = pd.to_numeric(df_db["Monetary"], errors="coerce").fillna(0)
                    df_db["Segment"] = df_db["Segment"].fillna("Unknown").astype(str)

        df_static = _get_static_segmented_df()
        if not df_static.empty and "Monetary" not in df_static.columns:
            df_static["Monetary"] = 0.0

        if not df_db.empty and not df_static.empty:
            df = pd.concat([df_db, df_static[["customer_id", "Segment", "Monetary"]]], ignore_index=True)
            df = df.drop_duplicates(subset=["customer_id"], keep="first")
        elif not df_db.empty:
            df = df_db
        elif not df_static.empty:
            df = df_static
        else:
            df = pd.DataFrame(columns=["customer_id", "Segment", "Monetary"])

        if df.empty:
            return StandardResponse(
                code=200, error=False, message="No data available",
                data=KPIResponse(data=[])
            )

        # 4. Hitung Metrik Cerdas
        total_customers = len(df)
        avg_monetary = df["Monetary"].mean() if not df["Monetary"].isna().all() else 0.0

        dominant_segment = "Belum Ada"
        segment_series = df["Segment"].dropna()
        if not segment_series.empty:
            dominant_segment = segment_series.mode()[0]

        # 5. Susun Response KPI
        kpis = [
            KeyPerformanceIndicator(
                title="Total Pelanggan Tersegmen",
                value=float(total_customers),
                trend=0.0
            ),
            KeyPerformanceIndicator(
                title="Rata-rata Nilai Pelanggan (M)",
                value=float(avg_monetary),
                trend=0.0
            ),
            KeyPerformanceIndicator(
                title="Segmen Paling Dominan",
                value=dominant_segment,
                trend=0.0
            ),
        ]
        
        return StandardResponse(
            code=200, error=False, message="KPI fetched successfully",
            data=KPIResponse(data=kpis),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

def get_customer_chart_data(
    target_date: Optional[date] = None,
    date_range: Literal["today", "last 7 days", "this month"] = "last 7 days"
) -> ChartDataResponse:
    try:
        df = _get_static_raw_df()  # dari cache, bukan re-read disk
        df = _prepare_transactions(df)

        min_date_env = os.getenv("MIN_DATE", "2018-03-01")
        max_date_env = os.getenv("MAX_DATE", "2018-03-31")

        try:
            min_date = datetime.strptime(min_date_env, "%Y-%m-%d").date()
            max_date = datetime.strptime(max_date_env, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=500,
                detail="Konfigurasi MIN_DATE atau MAX_DATE pada env tidak valid. Gunakan format YYYY-MM-DD."
            )

        # target_date fallback + clamp
        if target_date is None:
            target_date = max_date
        if target_date < min_date:
            target_date = min_date
        if target_date > max_date:
            target_date = max_date

        # Parse timestamp transaksi (utama: order_time, fallback: order_date)
        if "order_time" in df.columns:
            df["order_ts"] = pd.to_datetime(df["order_time"], errors="coerce")
        else:
            df["order_ts"] = pd.to_datetime(df["order_date"], errors="coerce")

        df = df.dropna(subset=["order_ts", "user_ID"]).copy()
        if df.empty:
            return StandardResponse(
                code=200,
                error=False,
                message="Chart data fetched successfully",
                data=ChartDataResponse(data=[]),
            )

        # Hitung first seen customer untuk newCustomers
        first_seen = (
            df.groupby("user_ID")["order_ts"]
            .min()
            .rename("first_seen_ts")
            .reset_index()
        )

        # Tentukan rentang + granularitas
        target_dt_start = datetime.combine(target_date, datetime.min.time())
        target_dt_end = datetime.combine(target_date, datetime.max.time())

        if date_range == "today":
            start_dt = target_dt_start
            end_dt = target_dt_end
            freq = "1h"
            label_fmt = "%H:%M"
        elif date_range == "this month":
            month_start = target_date.replace(day=1)
            start_dt = datetime.combine(max(min_date, month_start), datetime.min.time())
            end_dt = target_dt_end
            freq = "1D"
            label_fmt = "%d %b"
        else:
            # default / last 7 days, tidak boleh melewati min_date_env
            start_7 = target_date - timedelta(days=6)
            start_dt = datetime.combine(max(min_date, start_7), datetime.min.time())
            end_dt = target_dt_end
            freq = "6h"
            label_fmt = "%d %b %H:%M"

        # Filter range
        df_range = df[(df["order_ts"] >= start_dt) & (df["order_ts"] <= end_dt)].copy()

        # Active accounts per bucket
        active = (
            df_range.set_index("order_ts")
            .groupby(pd.Grouper(freq=freq))["user_ID"]
            .nunique()
            .rename("activeAccounts")
        )

        # New customers per bucket (berdasarkan first_seen_ts)
        first_seen_range = first_seen[
            (first_seen["first_seen_ts"] >= start_dt) &
            (first_seen["first_seen_ts"] <= end_dt)
        ].copy()

        new_cust = (
            first_seen_range.set_index("first_seen_ts")
            .groupby(pd.Grouper(freq=freq))["user_ID"]
            .nunique()
            .rename("newCustomers")
        )

        # Pastikan bucket lengkap (termasuk yg 0)
        idx = pd.date_range(start=start_dt, end=end_dt, freq=freq)
        merged = pd.DataFrame(index=idx)
        merged = merged.join(active, how="left").join(new_cust, how="left").fillna(0)

        chart_data = [
            CustomerChartData(
                date=ts.strftime(label_fmt),
                activeAccounts=int(row["activeAccounts"]),
                newCustomers=int(row["newCustomers"]),
            )
            for ts, row in merged.iterrows()
        ]

        return StandardResponse(
            code=200,
            error=False,
            message="Chart data fetched successfully",
            data=ChartDataResponse(data=chart_data),
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

async def get_customer_data_list(
    page: int = 1,
    per_page: int = 10,
    search: Optional[str] = None,
    segment: Optional[str] = None,
    db: Session = None,
    current_user: dict = None,
) -> CustomerDataResponse:
    try:
        df_static = _get_static_segmented_df()  # dari cache

        df_db = pd.DataFrame()
        if db and current_user:
            user_id = current_user.get("user_id")

            def _lrfm_field(key: str, label: str):
                return cast(SegmentationResult.lrfm[key].astext, Float).label(label)

            latest_subq = (
                select(
                    SegmentationResult.customer_id,
                    SegmentationResult.segment,
                    _lrfm_field("F", "frequency"),
                    _lrfm_field("M", "monetary"),
                    _lrfm_field("L", "length"),
                    _lrfm_field("R", "recency"),
                )
                .where(SegmentationResult.user_id == user_id)
                .distinct(SegmentationResult.customer_id)
                .order_by(SegmentationResult.customer_id, SegmentationResult.created_at.desc())
            ).subquery()

            rows = db.execute(select(latest_subq)).all()
            if rows:
                df_db = pd.DataFrame(
                    rows,
                    columns=["customer_id", "Segment", "Frequency", "Monetary", "Length", "Recency"],
                )
                df_db["customer_id"] = df_db["customer_id"].astype(str)
                for col in ["Frequency", "Monetary", "Length", "Recency"]:
                    df_db[col] = pd.to_numeric(df_db[col], errors="coerce").fillna(0)
                df_db["Segment"] = df_db["Segment"].fillna("Unknown").astype(str)

        # 3. Combine DataFrames
        if not df_db.empty and not df_static.empty:
            df = pd.concat([df_db, df_static], ignore_index=True)
            df = df.drop_duplicates(subset=["customer_id"], keep="first")
        elif not df_db.empty:
            df = df_db
        else:
            df = df_static

        # Ekstrak segments SETELAH cleaning (df_static & df_db sudah bersih dari cache/query)
        segments = sorted(df["Segment"].dropna().unique().tolist()) if "Segment" in df.columns else []
            
        # 4. Apply Search Query
        if search:
            df = df[df["customer_id"].astype(str).str.contains(search, case=False, na=False)]
        
        # 5. Apply Segment Filter
        if segment and segment.lower() != 'all':
            df = df[df["Segment"].str.lower() == segment.lower()]
        
        # 6. Pagination Logic
        total_data = len(df)
        total_page = math.ceil(total_data / per_page) if total_data > 0 else 1
        
        # Validasi batas halaman agar tidak keluar dari rentang yang tersedia
        if page > total_page and total_page > 0:
            page = total_page
        if page < 1:
            page = 1
            
        # Hitung indeks awal dan akhir untuk slicing DataFrame
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        # Ambil data sesuai potongan halaman
        df_sliced = df.iloc[start_idx:end_idx]
        
        # 7. Format final response
        reference_date = pd.to_datetime(os.getenv("MAX_DATE", "2018-03-31")).date() + timedelta(days=1)
        
        customers = []
        for _, row in df_sliced.iterrows():
            length_days = int(pd.to_numeric(row.get("Length", 0), errors="coerce") or 0)
            recency_days = int(pd.to_numeric(row.get("Recency", 0), errors="coerce") or 0)
            joined_date = reference_date - timedelta(days=length_days + recency_days)
            
            customers.append(
                CustomerData(
                    id=str(row["customer_id"]),
                    segment=str(row["Segment"]), 
                    orderCount=int(row["Frequency"]),
                    orderAmount=float(row["Monetary"]),
                    joinedDate=joined_date.isoformat()
                )
            )
            
        # Menyusun Metadata Paginasi
        metadata = PaginationMetadata(
            currentPage=page,
            perPage=per_page,
            totalPages=total_page,
            totalData=total_data,
            allSegments=segments
        )
            
        return StandardResponse(
            code=200,
            error=False,
            message="Customer data fetched successfully",
            data=CustomerDataResponse(
                metadata=metadata,
                data=customers,
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

def get_segment_trends(
    db: Session,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: Optional[dict] = None,
) -> StandardResponse[SegmentTrendResponse]:
    try:
        user_id = current_user.get("user_id") if current_user else None

        stmt = select(
            SegmentationResult.segment,
            SegmentationResult.transaction_date,
            SegmentationResult.created_at,
        )
        if user_id:
            stmt = stmt.where(SegmentationResult.user_id == user_id)

        rows = db.execute(stmt).all()
        db_rows = [
            {
                "segment": str(seg),
                "date": tx_date if tx_date is not None else created_at.date(),
            }
            for seg, tx_date, created_at in rows
        ]

        df_db = pd.DataFrame(db_rows)
        if not df_db.empty:
            df_db["date"] = pd.to_datetime(df_db["date"], utc=True)

        df_static = pd.DataFrame()
        df_raw_static = _get_static_segmented_df()  # dari cache
        if not df_raw_static.empty and "Segment" in df_raw_static.columns:
            date_col_candidates = ["last_transaction_date", "LastTransactionDate", "JoinedDate"]
            date_col = next((c for c in date_col_candidates if c in df_raw_static.columns), None)
            if date_col:
                df_raw_static["date"] = pd.to_datetime(df_raw_static[date_col], errors="coerce").dt.tz_localize("UTC")
                df_static = df_raw_static[["Segment", "date"]].rename(columns={"Segment": "segment"})
                df_static = df_static.dropna(subset=["date"])

        if not df_static.empty and not df_db.empty:
            df = pd.concat([df_db, df_static], ignore_index=True)
        elif not df_db.empty:
            df = df_db
        elif not df_static.empty:
            df = df_static
        else:
            df = pd.DataFrame(columns=["segment", "date"])
 
        # 4. Filter rentang tanggal
        now = datetime.now(timezone.utc)
 
        if end_date:
            end_dt = pd.to_datetime(end_date).tz_localize("UTC").replace(hour=23, minute=59, second=59)
        else:
            end_dt = now
 
        if start_date:
            start_dt = pd.to_datetime(start_date).tz_localize("UTC").replace(hour=0, minute=0, second=0)
        else:
            start_dt = end_dt - pd.Timedelta(days=30)
 
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
 
        if df.empty:
            return StandardResponse(
                code=200,
                error=False,
                message="Tidak ada data pada rentang waktu ini",
                data=SegmentTrendResponse(data=[], segments=[]),
            )
 
        # 5. Tentukan granularitas berdasarkan rentang hari
        delta_days = (end_dt - start_dt).days
 
        if delta_days <= 31:
            freq = "D"
            date_format = "%d %b"
        elif delta_days <= 365:
            freq = "W"
            date_format = "%d %b"
        else:
            freq = "ME"
            date_format = "%b %Y"
 
        # 6. Agregasi: GROUP BY period & segment → pivot
        df["period"] = df["date"].dt.floor(freq)
        grouped = df.groupby(["period", "segment"]).size().reset_index(name="count")
 
        pivot_df = (
            grouped
            .pivot(index="period", columns="segment", values="count")
            .fillna(0)
            .astype(int)
        )
 
        all_segments = list(pivot_df.columns)
        chart_data = []
        for period, row in pivot_df.iterrows():
            data_point = {"date": period.strftime(date_format)}
            for seg in all_segments:
                data_point[seg] = int(row[seg])
            chart_data.append(data_point)
 
        return StandardResponse(
            code=200,
            error=False,
            message="Segment trends fetched successfully",
            data=SegmentTrendResponse(data=chart_data, segments=all_segments),
        )
 
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Format tanggal tidak valid. Gunakan YYYY-MM-DD.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
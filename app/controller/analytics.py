import os
import math
import calendar
import pandas as pd
from datetime import date, datetime, timedelta
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
    PaginationMetadata
)

RAW_DATASET_PATH = os.path.join(os.getcwd(), "static", "dataset", "raw_data.csv")
SEGMENTED_DATASET_PATH = os.path.join(os.getcwd(), "static", "dataset", "segmented_data.csv")

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


async def get_kpis(target_date: Optional[date] = None) -> KPIResponse:
    try:
        df = _load_dataset(RAW_DATASET_PATH)
        df = _prepare_transactions(df)
        
        # Validasi target tanggal, fallback ke hari ini jika tidak ada
        if target_date is None:
            target_date = os.getenv("MAX_DATE", "2018-03-31")
            
        previous_date = target_date - timedelta(days=1)
        
        # Pisahkan data untuk Hari Ini (T) dan Kemarin (T-1)
        df_today = df[df["order_date"] == target_date]
        df_yesterday = df[df["order_date"] == previous_date]
        
        # Total Revenue (Monetary)
        revenue_today = float(df_today["revenue"].sum())
        revenue_yesterday = float(df_yesterday["revenue"].sum())
        trend_revenue = calculate_trend(revenue_today, revenue_yesterday)
        
        # Total Customers (Jumlah Baris)
        customers_today = int(df_today["user_ID"].nunique())
        customers_yesterday = int(df_yesterday["user_ID"].nunique())
        trend_customers = calculate_trend(customers_today, customers_yesterday)

        # Average Orders (Rata-rata Frequency)
        orders_today = int(df_today["order_ID"].nunique())
        orders_yesterday = int(df_yesterday["order_ID"].nunique())
        avg_orders_today = orders_today / customers_today if customers_today > 0 else 0.0
        avg_orders_yesterday = orders_yesterday / customers_yesterday if customers_yesterday > 0 else 0.0
        trend_avg_orders = calculate_trend(avg_orders_today, avg_orders_yesterday)

        kpis = [
            KeyPerformanceIndicator(
                title="Daily Revenue",
                value=revenue_today,
                trend=trend_revenue
            ),
            KeyPerformanceIndicator(
                title="Daily Active Customers",
                value=float(customers_today),
                trend=trend_customers
            ),
            KeyPerformanceIndicator(
                title="Average Orders",
                value=round(avg_orders_today, 2),
                trend=trend_avg_orders
            ),
        ]
        
        return StandardResponse(
            code=200,
            error=False,
            message="KPI fetched successfully",
            data=KPIResponse(data=kpis),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

async def get_customer_chart_data(
    target_date: Optional[date] = None,
    date_range: Literal["today", "last 7 days", "this month"] = "last 7 days"
) -> ChartDataResponse:
    try:
        df = _load_dataset(RAW_DATASET_PATH)
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
            label_fmt = "%Y-%m-%d %H:%M"
        elif date_range == "this month":
            month_start = target_date.replace(day=1)
            start_dt = datetime.combine(max(min_date, month_start), datetime.min.time())
            end_dt = target_dt_end
            freq = "1D"
            label_fmt = "%Y-%m-%d"
        else:
            # default / last 7 days, tidak boleh melewati min_date_env
            start_7 = target_date - timedelta(days=6)
            start_dt = datetime.combine(max(min_date, start_7), datetime.min.time())
            end_dt = target_dt_end
            freq = "6h"
            label_fmt = "%Y-%m-%d %H:%M"

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

async def get_customer_data_list(page: int) -> CustomerDataResponse:
    try:
        df = _load_dataset(SEGMENTED_DATASET_PATH)
        
        # Konfigurasi Paginasi (per_page dikunci senilai 10)
        per_page = 10
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
        
        # atur tanggal bergabung (joinedDate)
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
            totalPage=total_page,
            totalData=total_data
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
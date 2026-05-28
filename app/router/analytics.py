from fastapi import APIRouter, Depends, Query
from datetime import date
from typing import Any, Dict, Literal, Optional
from app.controller.analytics import (
    get_kpis,
    get_customer_chart_data,
    get_customer_data_list
)
from app.schemas.analytics import (
    ChartDataResponse,
    CustomerDataResponse,
    KPIResponse
)
from app.schemas.base import StandardResponse
from app.shared.auth import get_current_user

router = APIRouter(prefix="/analytics", dependencies=[Depends(get_current_user)])

@router.get(
    "/kpi",
    response_model=StandardResponse[KPIResponse],
    responses={
        422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
        401: {"model": StandardResponse[dict], "description": "Unauthorized"}
    },
    summary="Mendapatkan ringkasan KPI dan tren harian berdasarkan tanggal acuan",
)
async def kpi_endpoint(
    target_date: Optional[date] = Query(
        None, 
        description="Tanggal acuan untuk menghitung tren harian (Format: YYYY-MM-DD)"
    )
):
    return await get_kpis(target_date=target_date)

@router.get(
    "/charts",
    response_model=StandardResponse[ChartDataResponse],
    responses={
        422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
        401: {"model": StandardResponse[dict], "description": "Unauthorized"}
    },
    summary="Mendapatkan data grafik aktivitas pelanggan berdasarkan filter rentang",
)
async def chart_endpoint(
    target_date: Optional[date] = Query(
        None, 
        description="Tanggal acuan utama analisis data (Format: YYYY-MM-DD)"
    ),
    date_range: Literal["today", "last 7 days", "this month"] = Query(
        "last 7 days", 
        description="Rentang distribusi rentang waktu chart yang ingin diambil dari tanggal acuan"
    )
):
    return await get_customer_chart_data(target_date=target_date, date_range=date_range)

@router.get(
    "/customers",
    response_model=StandardResponse[CustomerDataResponse],
    responses={
        422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
        401: {"model": StandardResponse[dict], "description": "Unauthorized"}
    },
    summary="Mendapatkan daftar pelanggan beserta segmentasinya",
)
async def customers_endpoint(
    page: int = Query(1, ge=1, description="Nomor halaman untuk paginasi"),
):
    return await get_customer_data_list(page=page)
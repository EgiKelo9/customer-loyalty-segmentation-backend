from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import date
from typing import Any, Dict, Literal, Optional
from app.controller.analytics import (
    get_kpis,
    get_customer_chart_data,
    get_customer_data_list,
    get_segment_trends
)
from app.schemas.analytics import (
    ChartDataResponse,
    CustomerDataResponse,
    KPIResponse,
    SegmentTrendResponse,
)
from app.schemas.base import StandardResponse
from app.database.main import get_db
from app.shared.auth import get_current_user

router = APIRouter(prefix="/analytics", dependencies=[Depends(get_current_user)])

@router.get(
    "/kpis",
    response_model=StandardResponse[KPIResponse],
    responses={
        401: {"model": StandardResponse[dict], "description": "Unauthorized"},
        500: {"model": StandardResponse[dict], "description": "Internal Server Error"}
    },
    summary="Dapatkan ringkasan metrik profil pelanggan",
)
def kpi_endpoint(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> StandardResponse[KPIResponse]:
    return get_kpis(db, current_user)

@router.get(
    "/charts",
    response_model=StandardResponse[ChartDataResponse],
    responses={
        422: {"model": StandardResponse[Dict[str, Any]], "description": "Validation Error"},
        401: {"model": StandardResponse[dict], "description": "Unauthorized"}
    },
    summary="Mendapatkan data grafik aktivitas pelanggan berdasarkan filter rentang",
)
def chart_endpoint(
    target_date: Optional[date] = Query(
        None, 
        description="Tanggal acuan utama analisis data (Format: YYYY-MM-DD)"
    ),
    date_range: Literal["today", "last 7 days", "this month"] = Query(
        "last 7 days", 
        description="Rentang distribusi rentang waktu chart yang ingin diambil dari tanggal acuan"
    )
):
    return get_customer_chart_data(target_date=target_date, date_range=date_range)

@router.get(
    "/customers",
    response_model=StandardResponse[CustomerDataResponse],
    summary="Get customer data list with pagination and filters",
)
def get_customers(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by Customer ID"),
    segment: Optional[str] = Query(None, description="Filter by Segment name"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> StandardResponse[CustomerDataResponse]:
    # Pass db and current_user to the controller
    return get_customer_data_list(page, per_page, search, segment, db, current_user)

@router.get(
    "/segment-trends",
    response_model=StandardResponse[SegmentTrendResponse],
    summary="Get aggregated segment trends over a specific date range",
)
def get_segment_trends_route(
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> StandardResponse[SegmentTrendResponse]:
    return get_segment_trends(db, start_date, end_date, current_user)
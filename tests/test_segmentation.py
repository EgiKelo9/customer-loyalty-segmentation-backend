import asyncio
import io
import os
import sys

import pandas as pd
import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.controller import segmentation as segmentation_controller
from app.schemas.segmentation import CustomerInput, TransactionInput


def run(coro):
    return asyncio.run(coro)


def fake_segment_single(l, r, f, m):
    return {
        "cluster": 2,
        "pattern": "L↑R↓F↑M↑",
        "segment": "Champions",
        "recommendation": "Maintain engagement with premium offers",
        "fuzzy_membership": {
            "0": "5.00%",
            "1": "15.00%",
            "2": "80.00%",
        },
    }


def fake_extract_lrfm(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in df.iterrows():
        rows.append(
            {
                "customer_id": str(row.get("customer_id", f"C{index + 1}")),
                "Length": float(index + 10),
                "Recency": float(index + 2),
                "Frequency": float(index + 3),
                "Monetary": float((index + 1) * 1000),
            }
        )
    return pd.DataFrame(rows)


def make_upload_file(filename: str, csv_text: str) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(csv_text.encode("utf-8")))


def test_segment_from_lrfm_success(monkeypatch):
    monkeypatch.setattr(segmentation_controller, "segment_single", fake_segment_single)

    customer = CustomerInput(L=12, R=4, F=3, M=1_500_000)
    response = run(segmentation_controller.segment_from_lrfm(customer))

    assert response.customer_id is None
    assert response.cluster == 2
    assert response.pattern == "L↑R↓F↑M↑"
    assert response.segment == "Champions"
    assert response.recommendation == "Maintain engagement with premium offers"
    assert response.fuzzy_membership == {"0": "5.00%", "1": "15.00%", "2": "80.00%"}
    assert response.lrfm_calculated is None


def test_segment_from_transactions_success(monkeypatch):
    monkeypatch.setattr(segmentation_controller, "auto_map_columns", lambda df: df)
    monkeypatch.setattr(segmentation_controller, "extract_lrfm", fake_extract_lrfm)
    monkeypatch.setattr(segmentation_controller, "segment_single", fake_segment_single)

    transactions = [
        TransactionInput(
            customer_id="C001",
            transaction_date="2024-01-01",
            invoice_id="INV-001",
            amount=150_000,
        )
    ]

    response = run(segmentation_controller.segment_from_transactions(transactions))

    assert response.code == 200
    assert response.error is False
    assert response.message == "Segmentation successful for single customer"
    assert response.data.customer_id == "C001"
    assert response.data.cluster == 2
    assert response.data.segment == "Champions"


def test_segment_from_transactions_rejects_empty_list():
    with pytest.raises(HTTPException) as exc_info:
        run(segmentation_controller.segment_from_transactions([]))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Data transaksi tidak boleh kosong."


def test_segment_from_transactions_rejects_multiple_customers():
    transactions = [
        TransactionInput(
            customer_id="C001",
            transaction_date="2024-01-01",
            invoice_id="INV-001",
            amount=150_000,
        ),
        TransactionInput(
            customer_id="C002",
            transaction_date="2024-01-02",
            invoice_id="INV-002",
            amount=200_000,
        ),
    ]

    with pytest.raises(HTTPException) as exc_info:
        run(segmentation_controller.segment_from_transactions(transactions))

    assert exc_info.value.status_code == 400
    assert "Endpoint ini hanya untuk 1 pelanggan" in exc_info.value.detail


def test_segment_from_file_success_single_customer(monkeypatch):
    monkeypatch.setattr(segmentation_controller, "auto_map_columns", lambda df: df)
    monkeypatch.setattr(segmentation_controller, "extract_lrfm", fake_extract_lrfm)
    monkeypatch.setattr(segmentation_controller, "segment_single", fake_segment_single)

    upload_file = make_upload_file(
        "single.csv",
        "customer_id,transaction_date,invoice_id,amount\nC001,2024-01-01,INV-001,150000\n",
    )

    response = run(segmentation_controller.segment_from_file(upload_file))

    assert response.code == 200
    assert response.error is False
    assert response.message == "Segmentation successful for customer in uploaded file"
    assert response.data.customer_id == "C001"
    assert response.data.segment == "Champions"


def test_segment_from_file_success_batch_customer(monkeypatch):
    monkeypatch.setattr(segmentation_controller, "auto_map_columns", lambda df: df)
    monkeypatch.setattr(segmentation_controller, "extract_lrfm", fake_extract_lrfm)
    monkeypatch.setattr(segmentation_controller, "segment_single", fake_segment_single)

    upload_file = make_upload_file(
        "batch.csv",
        (
            "customer_id,transaction_date,invoice_id,amount\n"
            "C001,2024-01-01,INV-001,150000\n"
            "C002,2024-01-02,INV-002,250000\n"
        ),
    )

    response = run(segmentation_controller.segment_from_file(upload_file))

    assert response.code == 200
    assert response.error is False
    assert response.message == "Segmentation successful for customers in uploaded file"
    assert response.data.status == "success"
    assert response.data.total_customers == 2
    assert len(response.data.data) == 2
    assert {item.customer_id for item in response.data.data} == {"C001", "C002"}


def test_segment_from_file_rejects_unsupported_extension():
    upload_file = make_upload_file(
        "invalid.txt",
        "customer_id,transaction_date,invoice_id,amount\nC001,2024-01-01,INV-001,150000\n",
    )

    with pytest.raises(HTTPException) as exc_info:
        run(segmentation_controller.segment_from_file(upload_file))

    assert exc_info.value.status_code == 400
    assert "tidak didukung" in exc_info.value.detail


def test_get_segment_distribution_success(monkeypatch):
    dataset = pd.DataFrame(
        [
            {
                "customer_id": "C001",
                "Cluster": 0,
                "Segment": "Loyal Customers",
                "Recency": 10,
                "Frequency": 5,
                "Monetary": 100_000,
            },
            {
                "customer_id": "C002",
                "Cluster": 1,
                "Segment": "At Risk",
                "Recency": 20,
                "Frequency": 3,
                "Monetary": 200_000,
            },
        ]
    )

    monkeypatch.setattr(segmentation_controller.os.path, "exists", lambda path: True)
    monkeypatch.setattr(segmentation_controller.pd, "read_csv", lambda path: dataset.copy())

    response = run(segmentation_controller.get_segment_distribution())

    assert response.code == 200
    assert response.error is False
    assert response.message == "Distribution fetched successfully"
    assert len(response.data.segments) == 2
    assert response.data.allSegmentData[0].id == "all"
    assert response.data.allSegmentData[0].userCount == 2
    assert len(response.data.scatterData) == 2


def test_get_segment_distribution_missing_file(monkeypatch):
    monkeypatch.setattr(segmentation_controller.os.path, "exists", lambda path: False)

    with pytest.raises(HTTPException) as exc_info:
        run(segmentation_controller.get_segment_distribution())

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Segmented dataset not found."
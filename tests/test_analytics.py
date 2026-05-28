import os
import asyncio
from datetime import date

import app.controller.analytics as analytics

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")
RAW = os.path.join(DATA_DIR, "sample_raw_for_analytics.csv")
SEG = os.path.join(DATA_DIR, "sample_segmented_for_analytics.csv")


def setup_module(module):
    # override dataset paths used by analytics module to point to test samples
    analytics.RAW_DATASET_PATH = RAW
    analytics.SEGMENTED_DATASET_PATH = SEG
    os.environ.setdefault("MIN_DATE", "2018-03-01")
    os.environ.setdefault("MAX_DATE", "2018-03-31")


def test_get_kpis_basic():
    resp = asyncio.run(analytics.get_kpis(target_date=date(2018, 3, 1)))
    assert resp.code == 200
    assert resp.error is False
    kpis = resp.data.data
    # convert to dict by title
    values = {k.title: k.value for k in kpis}
    # revenue from sample: 100 + (2*50) = 200
    assert abs(values.get("Daily Revenue", 0) - 200.0) < 0.01
    assert int(values.get("Daily Active Customers", 0)) == 2
    assert abs(values.get("Average Orders", 0) - 1.0) < 0.01


def test_get_customer_chart_today():
    resp = asyncio.run(analytics.get_customer_chart_data(target_date=date(2018, 3, 1), date_range="today"))
    assert resp.code == 200
    assert resp.error is False
    data = resp.data.data
    # expect two active events across hourly buckets (09:00 and 10:00)
    total_active = sum(item.activeAccounts for item in data)
    total_new = sum(item.newCustomers for item in data)
    assert total_active == 2
    # only one new customer on that day in sample data
    assert total_new == 1


def test_get_customer_data_list_basic():
    resp = asyncio.run(analytics.get_customer_data_list(page=1))
    assert resp.code == 200
    assert resp.error is False
    # our sample segmented file has 2 customers
    assert resp.data.metadata.totalData == 2
    assert len(resp.data.data) == 2
    # each returned item should have joinedDate attribute (string)
    for c in resp.data.data:
        assert getattr(c, "joinedDate", None) is not None

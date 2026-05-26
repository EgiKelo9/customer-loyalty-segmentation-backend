import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from app.pipeline.preprocessing import auto_map_columns, clean_data, extract_lrfm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

passed = 0
failed = 0

def run_case(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}")
        print(f"     → {e}")
        failed += 1

SAMPLE = pd.DataFrame({
    "customer_id": ["C001", "C001", "C001", "C002", "C002"],
    "transaction_date": [
        "2024-01-01", "2024-02-15", "2024-03-20",
        "2024-01-10", "2024-01-10",
    ],
    "invoice_id": ["I1", "I2", "I3", "I4", "I4"],
    "amount": [100000, 200000, 150000, 50000, 50000],
})

def test_map_standard():
    df = pd.DataFrame(columns=["customer_id", "transaction_date", "invoice_id", "amount"])
    assert "customer_id" in auto_map_columns(df).columns

def test_map_alias():
    df = pd.DataFrame(columns=["user_id", "order_date", "order_id", "grand_total"])
    result = auto_map_columns(df)
    for col in ["customer_id", "transaction_date", "invoice_id", "amount"]:
        assert col in result.columns, f"Kolom '{col}' tidak ditemukan"

def test_map_uppercase():
    df = pd.DataFrame(columns=["Customer_ID", "Transaction_Date", "Invoice_ID", "Amount"])
    assert "customer_id" in auto_map_columns(df).columns

def test_map_with_spaces():
    df = pd.DataFrame(columns=[" customer_id ", " amount "])
    assert "customer_id" in auto_map_columns(df).columns

def test_map_unmatched_column_kept():
    df = pd.DataFrame(columns=["random_field", "customer_id"])
    result = auto_map_columns(df)
    assert "random_field" in result.columns

def test_clean_null_customer():
    df = pd.DataFrame({
        "customer_id": ["C001", None, "C003"],
        "transaction_date": ["2024-01-01"] * 3,
        "invoice_id": ["I1", "I2", "I3"],
        "amount": [100, 200, 300],
    })
    assert len(clean_data(df)) == 2

def test_clean_negative_amount():
    df = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "transaction_date": ["2024-01-01"] * 3,
        "invoice_id": ["I1", "I2", "I3"],
        "amount": [100, -50, 0],
    })
    assert len(clean_data(df)) == 1

def test_clean_non_numeric_amount():
    df = pd.DataFrame({
        "customer_id": ["C001", "C002"],
        "transaction_date": ["2024-01-01"] * 2,
        "invoice_id": ["I1", "I2"],
        "amount": ["abc", 500],
    })
    assert len(clean_data(df)) == 1

def test_clean_missing_column():
    df = pd.DataFrame({
        "customer_id": ["C001"],
        "transaction_date": ["2024-01-01"],
        "amount": [100],
    })
    try:
        clean_data(df)
        assert False, "Harusnya raise ValueError"
    except ValueError:
        pass

def test_clean_all_invalid_raises():
    df = pd.DataFrame({
        "customer_id": ["C001", "C002"],
        "transaction_date": ["2024-01-01", "2024-01-02"],
        "invoice_id": ["I1", "I2"],
        "amount": [0, -5],
    })
    try:
        clean_data(df)
        assert False, "Harusnya raise ValueError"
    except ValueError:
        pass

def test_lrfm_columns():
    result = extract_lrfm(SAMPLE.copy())
    for col in ["customer_id", "Length", "Recency", "Frequency", "Monetary"]:
        assert col in result.columns

def test_lrfm_row_count():
    assert len(extract_lrfm(SAMPLE.copy())) == 2

def test_lrfm_frequency_unique_invoice():
    # C002: 2 baris tapi invoice sama → Frequency = 1
    c002 = extract_lrfm(SAMPLE.copy())
    c002 = c002[c002["customer_id"] == "C002"].iloc[0]
    assert c002["Frequency"] == 1, f"Expected 1, got {c002['Frequency']}"

def test_lrfm_monetary_sum():
    # C001: 100000 + 200000 + 150000 = 450000
    c001 = extract_lrfm(SAMPLE.copy())
    c001 = c001[c001["customer_id"] == "C001"].iloc[0]
    assert c001["Monetary"] == 450000

def test_lrfm_length_days():
    # C001: 2024-01-01 → 2024-03-20 = 79 hari
    c001 = extract_lrfm(SAMPLE.copy())
    c001 = c001[c001["customer_id"] == "C001"].iloc[0]
    assert c001["Length"] == 79, f"Expected 79, got {c001['Length']}"

def test_lrfm_single_transaction():
    df = pd.DataFrame({
        "customer_id": ["C999"],
        "transaction_date": ["2024-06-01"],
        "invoice_id": ["I99"],
        "amount": [75000],
    })
    row = extract_lrfm(df).iloc[0]
    assert row["Length"] == 0
    assert row["Frequency"] == 1

def test_lrfm_invalid_dates_raises():
    df = pd.DataFrame({
        "customer_id": ["C001"],
        "transaction_date": ["not-a-date"],
        "invoice_id": ["I1"],
        "amount": [100],
    })
    try:
        extract_lrfm(df)
        assert False, "Harusnya raise ValueError"
    except ValueError:
        pass

def test_csv_valid():
    df = pd.read_csv(os.path.join(DATA_DIR, "sample_valid.csv"))
    result = extract_lrfm(auto_map_columns(df))
    assert len(result) == 3  # C001, C002, C003

def test_csv_bad_columns():
    # Kolom alias → harus bisa di-map otomatis
    df = pd.read_csv(os.path.join(DATA_DIR, "sample_bad_columns.csv"))
    result = extract_lrfm(auto_map_columns(df))
    assert len(result) == 2

def test_csv_dirty_cleaned():
    # Dari sample_dirty.csv → setelah cleaning tersisa C001 + C005
    df = pd.read_csv(os.path.join(DATA_DIR, "sample_dirty.csv"))
    result = extract_lrfm(auto_map_columns(df))
    assert len(result) == 2

def test_csv_lrfm_values_correct():
    # C002: INV004 + INV005 → Frequency=2, Monetary=125000, Length=51
    df = pd.read_csv(os.path.join(DATA_DIR, "sample_valid.csv"))
    result = extract_lrfm(auto_map_columns(df))
    c002 = result[result["customer_id"] == "C002"].iloc[0]
    assert c002["Frequency"] == 2
    assert c002["Monetary"] == 125000
    assert c002["Length"] == 51


if __name__ == "__main__":
    print("\n📦 TEST PREPROCESSING")
    print("─" * 40)

    print("\nauto_map_columns:")
    run_case("kolom standar", test_map_standard)
    run_case("alias kolom (user_id, order_date, dll)", test_map_alias)
    run_case("uppercase kolom", test_map_uppercase)
    run_case("kolom dengan spasi", test_map_with_spaces)

    print("\nclean_data:")
    run_case("hapus baris null customer_id", test_clean_null_customer)
    run_case("hapus amount negatif & nol", test_clean_negative_amount)
    run_case("hapus amount non-numerik", test_clean_non_numeric_amount)
    run_case("error kalau kolom wajib tidak ada", test_clean_missing_column)

    print("\nextract_lrfm (hardcoded):")
    run_case("kolom output lengkap", test_lrfm_columns)
    run_case("jumlah baris = pelanggan unik", test_lrfm_row_count)
    run_case("frequency hitung invoice unik", test_lrfm_frequency_unique_invoice)
    run_case("monetary = sum amount", test_lrfm_monetary_sum)
    run_case("length = selisih hari", test_lrfm_length_days)
    run_case("1 transaksi → length = 0", test_lrfm_single_transaction)

    print("\nextract_lrfm (dari CSV):")
    run_case("sample_valid.csv → 3 pelanggan", test_csv_valid)
    run_case("sample_bad_columns.csv → kolom alias ter-map", test_csv_bad_columns)
    run_case("sample_dirty.csv → data kotor dibersihkan", test_csv_dirty_cleaned)
    run_case("sample_valid.csv → nilai LRFM C002 benar", test_csv_lrfm_values_correct)

    print(f"\n{'═' * 40}")
    print(f"  Passed : {passed}")
    print(f"  Failed : {failed}")
    print(f"  Total  : {passed + failed}")
    print(f"{'═' * 40}\n")

    if failed > 0:
        sys.exit(1)
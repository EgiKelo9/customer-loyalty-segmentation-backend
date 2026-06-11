import re
import pandas as pd
from datetime import datetime
from typing import Optional

FIXED_REFERENCE_DATE = pd.to_datetime("2018-04-01")

def _normalize(name: str) -> str:
    """
    Normalisasi nama kolom:
    - lowercase
    - camelCase/PascalCase → pisah dengan spasi
    - ganti karakter non-alphanumeric dengan spasi
    - strip & deduplicate spasi

    Contoh:
    'CustomerID' → 'customer id'
    'transactionDate'→ 'transaction date'
    'grand_total' → 'grand total'
    'ID-Pelanggan' → 'id pelanggan'
    'AMOUNT' → 'amount'
    """
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(name))
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    name = re.sub(r'[^a-zA-Z0-9]', ' ', name)
    return re.sub(r'\s+', ' ', name.lower()).strip()

KEYWORD_RULES = {
    "customer_id": {
        "required": [
            {"customer", "id"},
            {"pelanggan", "id"},
            {"pembeli", "id"},
            {"member", "id"},
            {"user", "id"},
            {"userid"},
            {"customerid"},
        ],
        "exclude": {"invoice", "order", "transaction", "trx"},
    },
    "transaction_date": {
        "required": [
            {"date"},
            {"tanggal"},
            {"tgl"},
            {"purchase", "date"},
            {"order", "date"},
            {"transaction", "date"},
        ],
        "exclude": set(),
    },
    "invoice_id": {
        "required": [
            {"invoice"},
            {"order", "id"},
            {"order", "no"},
            {"no", "pesanan"},
            {"trx", "id"},
            {"transaction", "id"},
            {"id", "transaksi"},
            {"invoiceid"},
            {"orderid"},
        ],
        "exclude": {"customer", "pelanggan", "user", "date", "tanggal"},
    },
    "amount": {
        "required": [
            {"amount"},
            {"total"},
            {"nominal"},
            {"revenue"},
            {"price"},
            {"harga"},
            {"sales"},
            {"belanja"},
        ],
        "exclude": {"customer", "pelanggan", "date", "invoice", "order", "id"},
    },
}

def _matches_rule(tokens: set, rules: dict) -> bool:
    """
    Cek apakah token memenuhi salah satu required group
    dan tidak mengandung exclude keyword.
    """
    # Cek exclude dulu
    if tokens & rules.get("exclude", set()):
        return False

    # Cek apakah ada required group yang terpenuhi
    for required_group in rules["required"]:
        if required_group.issubset(tokens):
            return True

    return False

def auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Otomatis mapping nama kolom ke nama standar.

    Mendukung berbagai format:
    - snake_case : customer_id, grand_total
    - camelCase : customerId, grandTotal
    - PascalCase : CustomerId, GrandTotal
    - UPPERCASE : CUSTOMER_ID, AMOUNT
    - dibalik : id_customer, id_transaction
    - dengan spasi : Customer ID, Grand Total
    - bahasa indo : id_pelanggan, tgl_transaksi
    """
    df = df.copy()
    original_cols = list(df.columns)

    col_mapping = {}
    used_targets = set()

    for col in original_cols:
        normalized = _normalize(col)
        tokens = set(normalized.split())

        for target, rules in KEYWORD_RULES.items():
            if target in used_targets:
                continue
            if _matches_rule(tokens, rules):
                col_mapping[col] = target
                used_targets.add(target)
                break

    df.rename(columns=col_mapping, inplace=True)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["customer_id", "transaction_date", "invoice_id", "amount"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {missing_cols}. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    df = df.copy()
    df = df.dropna(subset=["customer_id", "transaction_date", "amount"])

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])
    df = df[df["amount"] > 0]

    if df.empty:
        raise ValueError("Tidak ada data valid setelah proses cleaning.")

    return df

def extract_lrfm(
    df: pd.DataFrame,
    reference_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Hitung nilai LRFM dari data transaksi:
    - L (Length) : Jarak hari antara transaksi pertama dan terakhir
    - R (Recency) : Hari sejak transaksi terakhir
    - F (Frequency) : Jumlah invoice unik
    - M (Monetary) : Total nilai transaksi

    Returns DataFrame dengan kolom:
    [customer_id, Length, Recency, Frequency, Monetary]
    """
    df_clean = clean_data(df)

    df_clean["transaction_date"] = pd.to_datetime(
        df_clean["transaction_date"],
        format='mixed',
        errors='coerce'
    )
    df_clean = df_clean.dropna(subset=["transaction_date"])

    if df_clean.empty:
        raise ValueError("Tidak ada tanggal transaksi yang valid.")

    if reference_date is None:
        reference_date = FIXED_REFERENCE_DATE

    lrfm_df = df_clean.groupby("customer_id").agg(
        First_Transaction=("transaction_date", "min"),
        Last_Transaction=("transaction_date", "max"),
        Frequency=("invoice_id", "nunique"),
        Monetary=("amount", "sum"),
    ).reset_index()

    lrfm_df["Length"] = (
        lrfm_df["Last_Transaction"] - lrfm_df["First_Transaction"]
    ).dt.days

    lrfm_df["Recency"] = (
        reference_date - lrfm_df["Last_Transaction"]
    ).dt.days

    return lrfm_df[["customer_id", "First_Transaction", "Length", "Recency", "Frequency", "Monetary"]]
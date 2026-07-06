import os
import json
import datetime as dt
import numpy as np
import pandas as pd
import joblib
import skfuzzy as fuzz

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RAW_DATA     = os.path.join(BASE_DIR, "static", "dataset", "raw_data.csv")
OUTPUT_PATH  = os.path.join(BASE_DIR, "static", "dataset", "segmented_data.csv")
ARTIFACTS    = os.path.join(BASE_DIR, "app", "artifacts")

SCALER_PATH   = os.path.join(ARTIFACTS, "scaler_lrfm.joblib")
CENTERS_PATH  = os.path.join(ARTIFACTS, "fcm_centers.joblib")
METADATA_PATH = os.path.join(ARTIFACTS, "metadata_segmentasi.json")

print("Loading artifacts...")
scaler      = joblib.load(SCALER_PATH)
fcm_centers = joblib.load(CENTERS_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

M_PARAM = float(metadata["m"])
CLUSTER_POLA_MAP = metadata["cluster_pola_map"]   # {"0": "L↓R↑F↓M↓", ...}
SEGMENT_MAP = metadata["segment_map"]
PROMO_MAP = metadata["promo_map"]

print(f"  m={M_PARAM}, clusters={len(CLUSTER_POLA_MAP)}")

print("Loading raw data...")
df = pd.read_csv(RAW_DATA)
df["order_date"] = pd.to_datetime(df["order_date"])

df = df[(df["quantity"] > 0) & (df["final_unit_price"] > 0)].copy()

df["total_price"] = df["quantity"] * df["final_unit_price"]

print("Computing LRFM...")
snapshot_date = df["order_date"].max() + dt.timedelta(days=1)

lrfm_df = df.groupby("user_ID").agg(
    first_order=("order_date", "min"),
    last_order=("order_date", "max"),
    frequency=("order_ID", "nunique"),
    monetary=("total_price", "sum"),
).reset_index()

lrfm_df["length"]  = (lrfm_df["last_order"] - lrfm_df["first_order"]).dt.days
lrfm_df["recency"] = (snapshot_date - lrfm_df["last_order"]).dt.days
lrfm_df = lrfm_df[["user_ID", "length", "recency", "frequency", "monetary", "last_order"]]

print(f"  Total customers: {len(lrfm_df)}")

print("Transforming features...")
features = ["length", "recency", "frequency", "monetary"]

lrfm_log = lrfm_df.copy()
for col in features:
    lrfm_log[col] = np.log1p(lrfm_log[col])

# Note: artifacts/scaler_lrfm.joblib is the one used in production inference
features_log = lrfm_log[features].values
features_scaled = scaler.transform(features_log)  # shape: (n_customers, 4)

# FCM predict requires shape (4, n_customers)
X = features_scaled.T

print("Running FCM prediction (this may take a while for large datasets)...")
u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
    test_data=X,
    cntr_trained=fcm_centers,
    m=M_PARAM,
    error=0.005,
    maxiter=1000,
)

# Hard assignment: cluster with highest membership
cluster_labels = np.argmax(u, axis=0)
lrfm_df["Cluster"] = cluster_labels

print("Mapping clusters to segments...")
lrfm_df["Pola"] = lrfm_df["Cluster"].apply(
    lambda c: CLUSTER_POLA_MAP.get(str(c), "Unknown")
)
lrfm_df["Segment"] = lrfm_df["Pola"].apply(
    lambda p: SEGMENT_MAP.get(p, f"Unknown ({p})")
)

output_df = pd.DataFrame({
    "customer_id": lrfm_df["user_ID"].astype(str),
    "Length": lrfm_df["length"].round(4),
    "Recency": lrfm_df["recency"].round(4),
    "Frequency": lrfm_df["frequency"].round(4),
    "Monetary": lrfm_df["monetary"].round(4),
    "Cluster": lrfm_df["Cluster"],
    "Segment": lrfm_df["Segment"],
    "last_transaction_date":  lrfm_df["last_order"].dt.date,
})

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
output_df.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Done! Saved to: {OUTPUT_PATH}")
print(f"   Total rows: {len(output_df)}")
print(f"\nCluster distribution:")
print(output_df.groupby(["Cluster", "Segment"]).size().reset_index(name="count").to_string(index=False))
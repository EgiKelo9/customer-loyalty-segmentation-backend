import joblib
import json
import numpy as np
import pandas as pd
import os
import skfuzzy as fuzz

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app", "artifacts")

try:
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler_lrfm.joblib"))
    fcm_centers = joblib.load(os.path.join(MODEL_PATH, "fcm_centers.joblib"))

    with open(os.path.join(MODEL_PATH, "metadata_segmentasi.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    M_PARAM = float(metadata.get("m", 1.5))

    CLUSTER_POLA_MAP: dict = metadata.get("cluster_pola_map", {})

    SEGMENT_MAP: dict = metadata.get("segment_map", {})
    PROMO_MAP: dict = metadata.get("promo_map", {})

except FileNotFoundError as e:
    raise RuntimeError(
        f"Model file tidak ditemukan: {e}. "
        f"Pastikan folder 'app/artifacts' berisi: scaler_lrfm.joblib, "
        f"fcm_centers.joblib, metadata_segmentasi.json"
    )

def segment_single(l: float, r: float, f: float, m: float) -> dict:
    """
    Prediksi segmen loyalitas untuk 1 pelanggan.

    Alur:
    1. Log1p transform (sesuai preprocessing notebook)
    2. StandardScaler transform
    3. FCM predict → dapat cluster_id & fuzzy membership
    4. Lookup pattern dari cluster_pola_map (sudah fix dari training)
    5. Lookup segment & recommendation dari metadata

    Args:
        l: Length  - hari antara transaksi pertama & terakhir
        r: Recency - hari sejak transaksi terakhir
        f: Frequency - jumlah transaksi unik
        m: Monetary  - total nilai transaksi

    Returns:
        dict dengan cluster, pola, segmen, rekomendasi, fuzzy_membership
    """
    # 1. Log1p transform
    features_log = np.array([[
        np.log1p(l),
        np.log1p(r),
        np.log1p(f),
        np.log1p(m),
    ]])
    
    features_df = pd.DataFrame(
        features_log,
        columns=["length", "recency", "frequency", "monetary"]
    )

    # 2. Scale
    features_scaled = scaler.transform(features_df).T

    # 3. FCM predict
    u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        test_data=features_scaled,
        cntr_trained=fcm_centers,
        m=M_PARAM,
        error=0.005,
        maxiter=1000,
    )

    probs = u[:, 0]
    cluster_id = int(np.argmax(probs))

    # 4. Lookup pattern dari cluster_pola_map
    pattern = CLUSTER_POLA_MAP.get(str(cluster_id), "Tidak diketahui")

    # 5. Lookup segment & recommendation
    segment = SEGMENT_MAP.get(pattern, "Segmen tidak diketahui")
    recommendation = PROMO_MAP.get(segment, "Tidak ada rekomendasi")

    return {
        "cluster": cluster_id,
        "pattern": pattern,
        "segment": segment,
        "recommendation": recommendation,
        "fuzzy_membership": {
            f"Cluster {i}": f"{p * 100:.2f}%"
            for i, p in enumerate(probs)
        },
    }
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

PROMO_KEYWORD_MAP = {
    "sampling": "sampling",
    "cashback": "cashback",
    "buy one get one free": "bogo",
    "buy one get one": "bogo",
    "price off deals": "price_off",
    "price off": "price_off",
    "bonus packs": "bonus_packs",
    "kupon": "kupon",
}

def segment_single(l: float, r: float, f: float, m: float) -> dict:
    """
    Prediksi segmen loyalitas untuk 1 pelanggan.

    Alur:
    1. Log1p transform (sesuai preprocessing notebook)
    2. MinMax Scaler transform
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
        np.log1p(np.maximum(r, 0)),
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

    def get_segment_name(idx: int) -> str:
        pat = CLUSTER_POLA_MAP.get(str(idx), "Tidak diketahui")
        return SEGMENT_MAP.get(pat, f"Cluster {idx}")

    # 4. Lookup pattern dari cluster_pola_map
    pattern = CLUSTER_POLA_MAP.get(str(cluster_id), "Tidak diketahui")

    # 5. Lookup segment & recommendation
    segment = SEGMENT_MAP.get(pattern, "Segmen tidak diketahui")
    recommendation = PROMO_MAP.get(segment, "Tidak ada rekomendasi")

    # fuzzy_membership: string version for display (backward compat)
    fuzzy_membership_str = {
        get_segment_name(i): f"{p * 100:.2f}%"
        for i, p in enumerate(probs)
    }

    return {
        "cluster": cluster_id,
        "pattern": pattern,
        "segment": segment,
        "recommendation": recommendation,
        "fuzzy_membership": fuzzy_membership_str,
    }


def segment_batch(df_lrfm: pd.DataFrame) -> pd.DataFrame:
    """
    Prediksi segmen untuk ratusan ribu pelanggan sekaligus (Vectorized).
    """
    # 1. Ekstrak kolom sebagai Numpy Array (Sangat Cepat)
    l = df_lrfm['Length'].values
    r = np.maximum(df_lrfm['Recency'].values, 0)
    f = df_lrfm['Frequency'].values
    m = df_lrfm['Monetary'].values

    # 2. Log1p Transform sekaligus ke seluruh baris
    features_log = np.column_stack([np.log1p(l), np.log1p(r), np.log1p(f), np.log1p(m)])
    
    features_df = pd.DataFrame(features_log, columns=["length", "recency", "frequency", "monetary"])

    # 3. Scaling sekaligus
    features_scaled = scaler.transform(features_df).T

    # 4. Prediksi FCM sekaligus (Numpy akan bekerja maksimal di sini)
    u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        test_data=features_scaled,
        cntr_trained=fcm_centers,
        m=M_PARAM,
        error=0.005,
        maxiter=1000,
    )

    # probs bentuknya (n_clusters, n_samples). Kita ubah (transpose) biar per baris
    probs = u.T 
    cluster_ids = np.argmax(probs, axis=1)

    # 5. Mapping massal menggunakan List Comprehension (Jauh lebih cepat dari apply pandas)
    patterns = [CLUSTER_POLA_MAP.get(str(cid), "Tidak diketahui") for cid in cluster_ids]
    segments = [SEGMENT_MAP.get(pat, "Segmen tidak diketahui") for pat in patterns]
    recs = [PROMO_MAP.get(seg, "Tidak ada rekomendasi") for seg in segments]

    # Helper function nama segmen (persis seperti fungsi lu sebelumnya)
    def get_segment_name(idx: int) -> str:
        pat = CLUSTER_POLA_MAP.get(str(idx), "Tidak diketahui")
        return SEGMENT_MAP.get(pat, f"Cluster {idx}")
        
    cluster_names = [get_segment_name(i) for i in range(probs.shape[1])]
    
    # Bikin kamus fuzzy membership massal
    fuzzy_memberships = [
        {name: f"{p * 100:.2f}%" for name, p in zip(cluster_names, row_probs)}
        for row_probs in probs
    ]

    # 6. Kembalikan data lengkap ke DataFrame
    df_out = df_lrfm.copy()
    df_out['cluster'] = cluster_ids
    df_out['pattern'] = patterns
    df_out['segment'] = segments
    df_out['recommendation'] = recs
    df_out['fuzzy_membership'] = fuzzy_memberships
    
    return df_out
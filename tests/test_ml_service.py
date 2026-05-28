import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

passed = 0
failed = 0

def run_case(name: str, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}")
        print(f"     → {e}")
        failed += 1

try:
    from app.pipeline.ml_service import segment_single, scaler, fcm_centers, metadata, CLUSTER_POLA_MAP
    MODEL_AVAILABLE = True
except RuntimeError as e:
    MODEL_AVAILABLE = False
    MODEL_ERROR = str(e)

if not MODEL_AVAILABLE:
    import pytest

    pytest.skip(
        f"Model belum tersedia: {MODEL_ERROR}",
        allow_module_level=True,
    )

def test_model_scaler_loaded():
    assert scaler is not None

def test_model_centers_loaded():
    assert fcm_centers is not None
    assert fcm_centers.ndim == 2  # shape (n_clusters, 4)
    assert fcm_centers.shape[1] == 4  # 4 fitur LRFM

def test_metadata_keys():
    for key in ["m", "segment_map", "promo_map", "cluster_pola_map"]:
        assert key in metadata, f"Key '{key}' tidak ada di metadata"

def test_cluster_pola_map_not_empty():
    assert len(CLUSTER_POLA_MAP) > 0

def test_cluster_pola_map_format():
    # Setiap nilai harus format L↑/↓R↑/↓F↑/↓M↑/↓
    for cluster_id, pola in CLUSTER_POLA_MAP.items():
        assert pola.startswith("L"), f"Pola '{pola}' harus dimulai dengan L"
        assert "R" in pola, f"Pola '{pola}' harus ada R"
        assert "F" in pola, f"Pola '{pola}' harus ada F"
        assert "M" in pola, f"Pola '{pola}' harus ada M"

def test_predict_output_keys():
    result = segment_single(l=25, r=3, f=10, m=1500000)
    for key in ["cluster", "pattern", "segment", "recommendation", "fuzzy_membership"]:
        assert key in result, f"Key '{key}' tidak ada di output"

def test_predict_cluster_valid():
    result = segment_single(l=25, r=3, f=10, m=1500000)
    n_clusters = fcm_centers.shape[0]
    assert 0 <= result["cluster"] < n_clusters

def test_predict_membership_count():
    result = segment_single(l=25, r=3, f=10, m=1500000)
    n_clusters = fcm_centers.shape[0]
    assert len(result["fuzzy_membership"]) == n_clusters

def test_predict_membership_sums_100():
    result = segment_single(l=25, r=3, f=10, m=1500000)
    total = sum(
        float(v.replace("%", ""))
        for v in result["fuzzy_membership"].values()
    )
    assert abs(total - 100.0) < 0.1, f"Total membership {total:.2f}% bukan 100%"

def test_predict_pattern_from_map():
    # Pola harus berasal dari cluster_pola_map, bukan dihitung ulang
    result = segment_single(l=25, r=3, f=10, m=1500000)
    cluster_id = str(result["cluster"])
    expected_pattern = CLUSTER_POLA_MAP.get(cluster_id)
    assert result["pattern"] == expected_pattern, (
        f"Pattern '{result['pattern']}' tidak sesuai cluster_pola_map '{expected_pattern}'"
    )

def test_predict_segment_not_unknown():
    result = segment_single(l=25, r=3, f=10, m=1500000)
    assert result["segment"] != "Segmen tidak diketahui", (
        f"Segmen 'Unknown' — cek cluster_pola_map & segment_map"
    )

def test_predict_minimum_values():
    # Nilai minimum tidak boleh error
    result = segment_single(l=0, r=1, f=1, m=1)
    assert "cluster" in result

def test_predict_large_values():
    # Nilai ekstrem besar tidak boleh error
    result = segment_single(l=3650, r=1, f=500, m=999_999_999)
    assert "cluster" in result

def test_predict_deterministic():
    # Hasil harus sama kalau input sama
    r1 = segment_single(l=30, r=5, f=8, m=750000)
    r2 = segment_single(l=30, r=5, f=8, m=750000)
    assert r1["cluster"] == r2["cluster"]
    assert r1["pattern"] == r2["pattern"]

def test_missing_model_files_raises(monkeypatch):
    import importlib.util
    import app.pipeline.ml_service as ml_service

    module_path = os.path.abspath(ml_service.__file__)
    spec = importlib.util.spec_from_file_location("ml_service_missing", module_path)
    module = importlib.util.module_from_spec(spec)

    def _raise(*args, **kwargs):
        raise FileNotFoundError("missing")

    import joblib

    monkeypatch.setattr(joblib, "load", _raise)
    with pytest.raises(RuntimeError):
        spec.loader.exec_module(module)


if __name__ == "__main__":
    print("\n🤖 TEST ML SERVICE")
    print("─" * 40)

    if not MODEL_AVAILABLE:
        print(f"\n  ⚠️  Model belum tersedia — semua test di-skip")
        print(f"     → {MODEL_ERROR}")
        print(f"\n  Langkah selanjutnya:")
        print(f"  1. Jalankan notebook → export model ke folder model/")
        print(f"  2. Pastikan ada: scaler_lrfm.joblib, fcm_centers.joblib, metadata_segmentasi.json")
        print(f"  3. Jalankan test ini lagi\n")
        sys.exit(0)

    print("\nModel Loading:")
    run_case("scaler berhasil di-load", test_model_scaler_loaded)
    run_case("fcm_centers shape benar (n_clusters × 4)", test_model_centers_loaded)
    run_case("metadata punya semua key yang diperlukan", test_metadata_keys)
    run_case("cluster_pola_map tidak kosong", test_cluster_pola_map_not_empty)
    run_case("cluster_pola_map format pola benar", test_cluster_pola_map_format)

    print("\npredict_single:")
    run_case("output punya semua key", test_predict_output_keys)
    run_case("cluster_id valid (0 sampai n_clusters-1)", test_predict_cluster_valid)
    run_case("jumlah membership = jumlah cluster", test_predict_membership_count)
    run_case("total fuzzy membership ~100%", test_predict_membership_sums_100)
    run_case("pola diambil dari cluster_pola_map (bukan dihitung ulang)", test_predict_pola_from_map)
    run_case("segmen bukan 'Segmen tidak diketahui'", test_predict_segmen_not_unknown)
    run_case("tidak crash untuk nilai minimum (L=0, F=1, M=1)", test_predict_minimum_values)
    run_case("tidak crash untuk nilai ekstrem besar", test_predict_large_values)
    run_case("hasil deterministik (input sama → output sama)", test_predict_deterministic)

    print(f"\n{'═' * 40}")
    print(f"  Passed : {passed}")
    print(f"  Failed : {failed}")
    print(f"  Total  : {passed + failed}")
    print(f"{'═' * 40}\n")

    if failed > 0:
        sys.exit(1)
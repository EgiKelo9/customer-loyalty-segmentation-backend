import os
import sys
import pandas as pd

# Pastikan root direktori masuk ke dalam sys.path agar module 'app' bisa diimport
sys.path.append(os.getcwd())

from app.pipeline.preprocessing import auto_map_columns, extract_lrfm
from app.pipeline.ml_service import segment_single

def main():
    # Menentukan path file berdasarkan struktur folder root
    BASE_DIR = os.getcwd()
    INPUT_PATH = os.path.join(BASE_DIR, "static", "dataset", "raw_data.csv")
    OUTPUT_PATH = os.path.join(BASE_DIR, "static", "dataset", "segmented_data.csv")
    
    print(f"[*] Membaca dataset dari: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print(f"[!] Error: File dataset tidak ditemukan di {INPUT_PATH}")
        sys.exit(1)
        
    try:
        # 1. Load data transaksi mentah
        df_raw = pd.read_csv(INPUT_PATH)
        print(f"[*] Berhasil memuat {len(df_raw)} baris data transaksi mentah.")
        
        # 2. Preprocessing & Ekstraksi LRFM
        print("[*] Melakukan preprocessing dan ekstraksi metrik LRFM...")
        df_mapped = auto_map_columns(df_raw)
        df_lrfm = extract_lrfm(df_mapped)
        print(f"[*] Berhasil mengekstrak LRFM untuk {len(df_lrfm)} pelanggan.")
        
        # 3. Proses Segmentasi menggunakan model yang dimuat dari joblib
        print("[*] Memulai proses clustering FCM...")
        results = []
        
        # Reset index agar customer_id bisa diakses sebagai kolom jika sebelumnya menjadi index
        df_lrfm_iterable = df_lrfm.reset_index() if 'customer_id' not in df_lrfm.columns else df_lrfm
        
        for _, row in df_lrfm_iterable.iterrows():
            # Memanggil fungsi segment_single (yang di dalamnya memuat model via joblib)
            res = segment_single(
                l=float(row['Length']),
                r=float(row['Recency']),
                f=float(row['Frequency']),
                m=float(row['Monetary'])
            )
            
            # Menyusun dictionary hasil untuk tiap pelanggan
            customer_id = row.get('customer_id', getattr(row, 'name', 'Unknown'))
            result_row = {
                'customer_id': customer_id,
                'Length': row['Length'],
                'Recency': row['Recency'],
                'Frequency': row['Frequency'],
                'Monetary': row['Monetary'],
                'Cluster': res['cluster'],
                'Segment': res['segment'],
                'Pattern': res['pattern'],
                'Recommendation': res['recommendation'],
                'JoinedDate': (pd.to_datetime(row.get('First_Transaction')).date().isoformat() if pd.notna(row.get('First_Transaction')) else None)
            }
            results.append(result_row)
            
        # 4. Menyimpan output
        final_df = pd.DataFrame(results)
        final_df.to_csv(OUTPUT_PATH, index=False)
        print(f"[+] File hasil segmentasi berhasil disimpan di:\n    -> {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"[!] Terjadi kesalahan saat memproses data: {str(e)}")

if __name__ == "__main__":
    main()

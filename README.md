# Kecerdasan-Komputasional
## Deskripsi
Kecerdasan-Komputasional adalah kumpulan skrip dan modul Python untuk eksperimen dengan algoritma pembelajaran mesin dan metode optimasi (mis. neural network, evolutionary algorithms, dan fuzzy logic). Repo ini dirancang untuk cepat melakukan pelatihan, inferensi, dan evaluasi pada dataset tabular atau file CSV.

## Fitur
- Pipeline sederhana untuk pelatihan, validasi, dan inferensi
- Skrip contoh untuk training, evaluasi, dan prediksi
- Konfigurasi lewat file YAML/JSON
- Logging dan checkpoint model

## Persyaratan
- Python 3.13
- Paket akan dicantumkan di requirements.txt (contoh: numpy, pandas, scikit-learn, torch/tensorflow jika diperlukan)

## Instalasi
1. Clone repo:
    git clone <repo-url>
2. Buat virtual environment dan aktifkan:
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate     # Windows
3. Install dependency:
    pip install -r requirements.txt

## Struktur proyek (singkat)
- ann.py            
- reggesion.py      
- datasheet.py      
- ann_output/       
- datasheet/        
- diagram_output/   

## Cara penggunaan
Menyalakan env:
```bash
.venv\Scripts\activate
```

Instal dependency:
```bash
pip install -r requirements.txt
```

Menjalankan script python:
```bash
python ann.py
```

## Format data
- Format umum: CSV
- Kolom fitur (numerik/one-hot), kolom target bernama sesuai config (mis. "label")
- Pastikan missing value ditangani atau script preprocessing dijalankan

## Evaluasi & metrik
- Binary/Multiclass: accuracy, precision, recall, f1
- Regression: MAE, MSE, R2
- Hasil evaluasi dicatat ke log dan file CSV di folder results/

## Tips & troubleshooting
- Pastikan versi Python dan dependency sesuai requirements.txt
- Periksa path file di config
- Cek log untuk stacktrace jika proses berhenti
- Untuk GPU, pastikan driver dan framework (CUDA) cocok

## Kontribusi
- Buka issue untuk bug/fitur
- Fork repo, buat branch fitur, kirim pull request
- Ikuti format test dan code style (black/isort jika disediakan)

## Lisensi
Distribusi di bawah lisensi MIT. Lihat file LICENSE untuk detail.

## Kontak
Untuk pertanyaan, buka issue di repository.

# Tugas Kecerdasan Komputasional (Python)

Repository ini berisi kumpulan kode implementasi algoritma untuk mata kuliah **Kecerdasan Komputasional**.
**Author**: Zainul Mutawakkil

## Struktur Direktori

### 1. Algoritma Genetika (Genetic Algorithm)
Folder: `Algoritma Genetika/`

Berisi implementasi Algoritma Genetika untuk menyelesaikan masalah optimasi:

- **[jadwal_ga.py](./Algoritma%20Genetika/jadwal_ga.py)**: 
  - **Tujuan**: Menyusun jadwal 3 mata kuliah (A, B, C) dalam 1 ruangan tanpa bentrok.
  - **Constraints**: 
    1. Tidak boleh ada jadwal yang overlap.
    2. Mata Kuliah A (Prof. X) harus selesai sebelum jam 10.00 pagi.
  - **Metode**: Evolusi populasi waktu mulai (start time) hingga ditemukan solusi valid.

- **[Evo.py](./Algoritma%20Genetika/Evo.py)**:
  - **Tujuan**: Mendemonstrasikan evolusi string sederhana.
  - **Target**: Membentuk string "ZAINUL MUTAWAKKIL" dari string acak.
  - **Metode**: Mutasi karakter acak dan seleksi berdasarkan kesamaan huruf dengan target.

### 2. Jaringan Syaraf Tiruan (Artificial Neural Network)
File Root Directory

Berisi implementasi ANN menggunakan TensorFlow/Keras untuk kasus regresi (Prediksi Superchat):

- **[ann.py](./ann.py)**:
  - **Deskripsi**: Script utama untuk melatih model ANN.
  - **Fitur**: 
    - Load dataset (`channels.csv`, `chat_stats.csv`, `superchat_stats.csv`).
    - Preprocessing & Feature Engineering.
    - Arsitektur ANN (Dense layers + Dropout).
    - Visualisasi Training Loss & MAE.
    - Evaluasi Model (R-squared, MAE).
  - **Output**: Model disimpan sebagai `ann_superchat_model.h5` di folder `ann_outputs/`.

- **[reggesion.py](./reggesion.py)**:
  - **Deskripsi**: Versi alternatif/awal untuk prediksi regresi Superchat.
  - **Fitur**: Mirip dengan `ann.py`, menggunakan Sequential Model untuk memprediksi `totalSC`.

## Cara Menjalankan

### Menjalankan Algoritma Genetika
Masuk ke folder dan jalankan script:
```bash
cd "Algoritma Genetika"
python3 jadwal_ga.py
# atau
python3 Evo.py
```

### Menjalankan ANN
Pastikan dependency terinstall (`tensorflow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`).
```bash
python3 ann.py
```

import kagglehub
import shutil
import os
import pandas as pd

#Download dataset dari Kaggle
print("Downloading dataset...")
path = kagglehub.dataset_download("uetchy/vtuber-livechat-elements")
print("Dataset downloaded to:", path)

#Tentukan folder tujuan
target_dir = r"C:\Users\Zee\Documents\Python\datasheet"

#Salin dataset ke folder tujuan
print("Copying dataset files...")
shutil.copytree(path, target_dir, dirs_exist_ok=True)
print("Dataset copied to:", target_dir)

#Cek semua file CSV di folder tersebut
csv_files = [f for f in os.listdir(target_dir) if f.endswith(".csv")]
if not csv_files:
    print("⚠️ Tidak ditemukan file CSV di folder:", target_dir)
else:
    print("✅ Ditemukan file CSV berikut:")
    for file in csv_files:
        print("-", file)

    #(Opsional) Baca file CSV pertama
    first_csv = os.path.join(target_dir, csv_files[0])
    print("\nMembaca file:", first_csv)
    df = pd.read_csv(first_csv)
    print("Jumlah baris:", len(df))
    print("Kolom yang tersedia:", list(df.columns))

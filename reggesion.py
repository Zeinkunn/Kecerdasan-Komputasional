import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shutil

# ---------------------------------------------------------
#LOAD DATASET
# ---------------------------------------------------------
DATA_DIR = r"C:\Users\Zee\Documents\Python\datasheet"

def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    return pd.read_csv(path)

print("📂 Memuat dataset...")
df_channels = load_csv('channels.csv')
df_chat_stats = load_csv('chat_stats.csv')
df_superchat_stats = load_csv('superchat_stats.csv')
print("✅ Semua file CSV berhasil dimuat.")

# Cek kolom tiap file
for name, df in [('channels', df_channels), ('chat_stats', df_chat_stats), ('superchat_stats', df_superchat_stats)]:
    print(f"{name} columns:", list(df.columns))

# ---------------------------------------------------------
#GABUNG DATA
# ---------------------------------------------------------
df_stats = pd.merge(
    df_chat_stats, 
    df_superchat_stats, 
    on=['channelId', 'period'], 
    how='inner'
)

df_merged = pd.merge(
    df_stats,
    df_channels,
    on='channelId',
    how='inner'
)

print(f"\n✅ Data berhasil digabung. Total baris: {len(df_merged)}")
print(f"Total kolom: {len(df_merged.columns)}")

# ---------------------------------------------------------
#PREPROCESSING
# ---------------------------------------------------------
TARGET_COLUMN = 'totalSC'
NUMERICAL_FEATURES = [
    'chats', 'memberChats', 'uniqueChatters', 'uniqueMembers',
    'superChats', 'uniqueSuperChatters', 'subscriptionCount', 'videoCount'
]
CATEGORICAL_FEATURES = ['affiliation']

# Bersihkan afiliasi
top_5_affiliations = df_merged['affiliation'].value_counts().nlargest(5).index
df_merged['affiliation_clean'] = df_merged['affiliation'].apply(
    lambda x: x if x in top_5_affiliations else 'Lainnya'
)
CATEGORICAL_FEATURES = ['affiliation_clean']

# Siapkan X & y
y = df_merged[TARGET_COLUMN]
X = df_merged[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

# Encoding kategorikal
X_encoded = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True)
print(f"✅ Data di-encode. Jumlah fitur input: {X_encoded.shape[1]}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Scaling fitur
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scaling target
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

print("✅ Data siap untuk training.")

# ---------------------------------------------------------
#BANGUN MODEL ANN
# ---------------------------------------------------------
n_features = X_train_scaled.shape[1]
model = Sequential([
    Dense(128, input_dim=n_features, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()

# ---------------------------------------------------------
#TRAINING DENGAN EARLY STOPPING
# ---------------------------------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\n🚀 Training model...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print("✅ Training selesai!")

# ---------------------------------------------------------
#VISUALISASI LOSS
# ---------------------------------------------------------
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# ---------------------------------------------------------
#EVALUASI MODEL
# ---------------------------------------------------------
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_asli = y_test.values

mae = mean_absolute_error(y_test_asli, y_pred)
r2 = r2_score(y_test_asli, y_pred)
mean_sc_value = y_test_asli.mean()

print("\n📊 HASIL EVALUASI MODEL REGRESI")
print(f"Rata-rata totalSC di data tes: {mean_sc_value:,.2f}")
print(f"Mean Absolute Error (MAE):     {mae:,.2f}")
print(f"R-squared (R²):                {r2:.4f}")

df_hasil = pd.DataFrame({
    'Nilai Aktual (totalSC)': y_test_asli[:5],
    'Tebakan Model': y_pred.flatten()[:5]
})
df_hasil['Selisih'] = df_hasil['Nilai Aktual (totalSC)'] - df_hasil['Tebakan Model']
print("\n🧾 Contoh Tebakan vs Aktual:")
print(df_hasil.to_string(formatters={
    'Nilai Aktual (totalSC)': '{:,.0f}'.format,
    'Tebakan Model': '{:,.0f}'.format,
    'Selisih': '{:,.0f}'.format
}))

# ---------------------------------------------------------
#SIMPAN MODEL
# ---------------------------------------------------------
model_path = os.path.join(DATA_DIR, "vtuber_sc_predictor.h5")
model.save(model_path)
print(f"\n💾 Model disimpan di: {model_path}")

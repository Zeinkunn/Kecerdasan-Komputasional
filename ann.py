# ann_superchat_regression.py
# Prediksi totalSC (Superchat revenue) menggunakan ANN (regresi)
# Output: plots via matplotlib, predictions.csv, evaluation_summary.txt
# Author: Zainul Mutawakkil (23EO10003)
# Run: python ann.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ML / DL libs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Config / Paths
# -------------------------
DATA_DIR = r"C:\Users\Zee\Documents\Python\datasheet"  # files uploaded here
CHANNELS_CSV = os.path.join(DATA_DIR, "channels.csv")
CHAT_STATS_CSV = os.path.join(DATA_DIR, "chat_stats.csv")
SUPERCHAT_STATS_CSV = os.path.join(DATA_DIR, "superchat_stats.csv")

OUTPUT_DIR = os.path.join(".", "ann_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Utility functions
# -------------------------
def safe_read_csv(path):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    return pd.read_csv(path)

def print_df_head_tail(df, name, n=3):
    print(f"\n== {name} (head) ==")
    print(df.head(n).to_string())
    print(f"\n== {name} (tail) ==")
    print(df.tail(n).to_string())

# -------------------------
# 1) Load CSVs
# -------------------------
print("Loading CSV files...")
df_channels = safe_read_csv(CHANNELS_CSV)
df_chat_stats = safe_read_csv(CHAT_STATS_CSV)
df_superchat_stats = safe_read_csv(SUPERCHAT_STATS_CSV)
print("Files loaded.")

# quick column overview
print("\n--- Columns ---")
for name, df in [("channels", df_channels), ("chat_stats", df_chat_stats), ("superchat_stats", df_superchat_stats)]:
    print(f"{name}: {list(df.columns)}")

# -------------------------
# 2) Merge datasets
# -------------------------
print("\nMerging datasets on ['channelId', 'period'] where applicable...")
# Merge chat_stats & superchat_stats on channelId + period
if 'channelId' not in df_chat_stats.columns or 'period' not in df_chat_stats.columns:
    print("[ERROR] chat_stats.csv missing 'channelId' or 'period' columns")
    sys.exit(1)
if 'channelId' not in df_superchat_stats.columns or 'period' not in df_superchat_stats.columns:
    print("[ERROR] superchat_stats.csv missing 'channelId' or 'period' columns")
    sys.exit(1)

df_stats = pd.merge(df_chat_stats, df_superchat_stats, on=['channelId', 'period'], how='inner', suffixes=('_chat', '_sc'))
print(f"After merging chat & superchat stats: {len(df_stats)} rows")

# Merge with channels metadata
if 'channelId' not in df_channels.columns:
    print("[ERROR] channels.csv missing 'channelId' column")
    sys.exit(1)

df_merged = pd.merge(df_stats, df_channels, on='channelId', how='inner')
print(f"After merging with channels metadata: {len(df_merged)} rows")

# Preview merged
print_df_head_tail(df_merged, "merged")

# -------------------------
# 3) Initial analysis & visualization prep
# -------------------------
print("\nBasic statistics (numeric features):")
numeric = df_merged.select_dtypes(include=[np.number])
print(numeric.describe().T[['count','mean','std','min','25%','50%','75%','max']])

# Save a correlation heatmap data (we'll plot with matplotlib)
corr = numeric.corr()
corr_path = os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
corr.to_csv(corr_path)
print(f"Correlation matrix saved to {corr_path}")

# Plot distribution of target 'totalSC' if present
TARGET = "totalSC"
if TARGET not in df_merged.columns:
    print(f"[ERROR] Target column '{TARGET}' not found in merged data. Available numeric columns: {list(numeric.columns)}")
    sys.exit(1)

plt.figure(figsize=(8,4))
plt.hist(df_merged[TARGET].dropna(), bins=60)
plt.title(f"Distribution of target: {TARGET}")
plt.xlabel(TARGET)
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# also show log-distribution if skewed
plt.figure(figsize=(8,4))
plt.hist(np.log1p(df_merged[TARGET].clip(lower=0)), bins=60)
plt.title(f"Log(1 + {TARGET}) Distribution")
plt.xlabel(f"log1p({TARGET})")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# -------------------------
# 4) Preprocessing / Feature Engineering
# -------------------------
print("\nPreprocessing & feature engineering...")

# Choose numerical features likely to be predictive (modify if absent)
candidate_numerical = [
    'chats', 'memberChats', 'uniqueChatters', 'uniqueMembers',
    'superChats', 'uniqueSuperChatters', 'subscriptionCount', 'videoCount',
    'averageSC'  # include if present
]
numerical_features = [c for c in candidate_numerical if c in df_merged.columns]
print("Numerical features used:", numerical_features)

# Categorical: affiliation (clean top 5)
cat_col = 'affiliation'
if cat_col in df_merged.columns:
    top_aff = df_merged[cat_col].fillna("Unknown").value_counts().nlargest(5).index.tolist()
    df_merged['affiliation_clean'] = df_merged[cat_col].fillna("Unknown").apply(lambda x: x if x in top_aff else "Other")
    categorical_features = ['affiliation_clean']
    print("Categorical features used:", categorical_features, " (top 5 preserved)")
else:
    categorical_features = []
    print("No affiliation column found; skipping categorical encoding.")

# Build feature dataframe
features = numerical_features.copy()
df_features = df_merged[features].copy()
if categorical_features:
    df_features = pd.concat([df_features, pd.get_dummies(df_merged[categorical_features], drop_first=True)], axis=1)

# Drop rows with NA in features or target
mask_ok = df_features.notna().all(axis=1) & df_merged[TARGET].notna()
df_features = df_features[mask_ok]
df_target = df_merged.loc[mask_ok, TARGET]
print(f"After dropping NA rows, samples = {len(df_features)}")

# Quick check
print("Feature matrix shape:", df_features.shape)
print("Target vector shape:", df_target.shape)

# -------------------------
# 5) Train/test split & scaling
# -------------------------
X = df_features.values
y = df_target.values.reshape(-1,1)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f"Train samples: {len(X_train)}  Test samples: {len(X_test)}")

# scale X and y (scaling y helps training stability for regression with ANN)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# -------------------------
# 6) Build ANN model (complex)
# -------------------------
print("\nBuilding ANN model...")

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.30))

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.25))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.20))

    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(1, activation='linear'))  # regression output
    return model

model = build_model(X_train_scaled.shape[1])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
model.summary()

# -------------------------
# 7) Callbacks: early stopping & LR reduce
# -------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1)
]

# -------------------------
# 8) Train
# -------------------------
EPOCHS = 100
BATCH = 32

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=2
)

# -------------------------
# 9) Plot training curves: loss and MAE (scaled y metrics are MSE/MAE on scaled y)
# -------------------------
# Convert history to dataframe and save
hist_df = pd.DataFrame(history.history)
hist_csv = os.path.join(OUTPUT_DIR, "training_history.csv")
hist_df.to_csv(hist_csv, index=False)
print(f"Training history saved to {hist_csv}")

# Plot MSE (loss) and MAE (note: MAE here for scaled y)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss (MSE)')
plt.plot(history.history['val_loss'], label='val_loss (MSE)')
plt.title('Loss (MSE) per epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['mae'], label='train_mae (scaled)')
plt.plot(history.history['val_mae'], label='val_mae (scaled)')
plt.title('MAE (scaled y) per epoch')
plt.xlabel('Epoch')
plt.ylabel('MAE (scaled)')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# 10) Evaluate on test set (inverse transform predictions)
# -------------------------
print("\nEvaluating on test set...")
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_test_orig = y_test.flatten()

mae = mean_absolute_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)
mse = np.mean((y_test_orig - y_pred)**2)

print(f"Test MAE : {mae:,.4f}")
print(f"Test MSE : {mse:,.4f}")
print(f"Test R2  : {r2:.4f}")

# Save evaluation summary
eval_txt_path = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")
with open(eval_txt_path, "w") as f:
    f.write(f"ANN Superchat Regression Evaluation\n")
    f.write(f"Date: {datetime.now()}\n")
    f.write(f"Samples (train/test): {len(X_train)}/{len(X_test)}\n")
    f.write(f"Features used: {df_features.columns.tolist()}\n\n")
    f.write(f"Test MAE: {mae}\n")
    f.write(f"Test MSE: {mse}\n")
    f.write(f"Test R2:  {r2}\n")
print(f"Evaluation summary saved to {eval_txt_path}")

# -------------------------
# 11) Scatter plot: actual vs predicted
# -------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test_orig, y_pred, alpha=0.6, s=20)
mn = min(np.min(y_test_orig), np.min(y_pred))
mx = max(np.max(y_test_orig), np.max(y_pred))
plt.plot([mn,mx], [mn,mx], color='red', linestyle='--', label='y = y_pred')
plt.xlabel('Actual totalSC')
plt.ylabel('Predicted totalSC')
plt.title('Actual vs Predicted (test set)')
plt.legend()
plt.tight_layout()
plt.show()

# Also show first 10 comparisons in a table
comp_df = pd.DataFrame({
    'actual_totalSC': y_test_orig,
    'predicted_totalSC': y_pred,
    'abs_error': np.abs(y_test_orig - y_pred)
}).sort_values('abs_error').reset_index(drop=True)

print("\nTop 10 best predictions (smallest absolute error):")
print(comp_df.head(10).to_string(index=False, formatters={
    'actual_totalSC': '{:,.2f}'.format,
    'predicted_totalSC': '{:,.2f}'.format,
    'abs_error': '{:,.2f}'.format
}))

print("\nTop 10 worst predictions (largest absolute error):")
print(comp_df.tail(10).sort_values('abs_error', ascending=False).head(10).to_string(index=False, formatters={
    'actual_totalSC': '{:,.2f}'.format,
    'predicted_totalSC': '{:,.2f}'.format,
    'abs_error': '{:,.2f}'.format
}))

# -------------------------
# 12) Save predictions & model
# -------------------------
predictions_path = os.path.join(OUTPUT_DIR, "predictions_test_set.csv")
comp_df.to_csv(predictions_path, index=False)
print(f"Predictions saved to {predictions_path}")

model_path = os.path.join(OUTPUT_DIR, "ann_superchat_model.h5")
model.save(model_path)
print(f"Model saved to {model_path}")

# -------------------------
# 13) Final notes & end
# -------------------------
print("\nALL DONE ✅")
print(f"Outputs folder: {os.path.abspath(OUTPUT_DIR)}")
print("Files produced:")
for fname in os.listdir(OUTPUT_DIR):
    print(" -", fname)



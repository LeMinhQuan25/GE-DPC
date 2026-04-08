# covtype_to_gedpc_full.py

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# 1) CONFIG
# =========================================================
DATA_FILE = Path("/Users/quanle/Downloads/covertype/covtype.data.gz")
OUT_DIR = Path("/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/data")

OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_OUT = OUT_DIR / "covtype_54d.txt"
LABEL_OUT   = OUT_DIR / "covtype_label.txt"
INFO_OUT    = OUT_DIR / "covtype_info.txt"

# =========================================================
# 2) LOAD FULL DATA
# =========================================================
# File gốc không có header
df = pd.read_csv(
    DATA_FILE,
    header=None,
    compression="gzip"
)

print("Loaded shape:", df.shape)

# Chuẩn Covertype: 581012 rows, 55 cols = 54 features + 1 label
expected_rows = 581012
expected_cols = 55

if df.shape[1] != expected_cols:
    raise ValueError(f"Số cột hiện tại = {df.shape[1]}, expected = {expected_cols}")

if df.shape[0] != expected_rows:
    print(f"Warning: số dòng hiện tại = {df.shape[0]}, expected = {expected_rows}")

# =========================================================
# 3) SPLIT X, y
# =========================================================
X = df.iloc[:, :-1].copy()   # 54 feature
y = df.iloc[:, -1].copy()    # label Cover_Type

print("Feature shape:", X.shape)
print("Label shape  :", y.shape)

if X.shape[1] != 54:
    raise ValueError(f"Số chiều feature hiện tại = {X.shape[1]}, không phải 54")

# =========================================================
# 4) CLEAN / TYPE CAST
# =========================================================
X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

if X.isna().any().any():
    raise ValueError("Feature có NaN, dữ liệu không đúng như mong đợi")

if y.isna().any():
    raise ValueError("Label có NaN")

# Label gốc của Covertype là 1..7
# đổi về 0..6 cho tiện đánh giá
unique_labels = sorted(y.unique().tolist())
label_map = {lab: idx for idx, lab in enumerate(unique_labels)}
y_encoded = y.map(label_map).astype(int).to_numpy()

print("Unique raw labels:", unique_labels)
print("Label mapping    :", label_map)

# =========================================================
# 5) MIN-MAX NORMALIZE
# =========================================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.astype(np.float64))
X_scaled = np.clip(X_scaled, 0.0, 1.0)

# =========================================================
# 6) FINAL CHECK
# =========================================================
n_rows, n_cols = X_scaled.shape

print("Final X shape:", X_scaled.shape)
print("Final y shape:", y_encoded.shape)
print("X min/max    :", X_scaled.min(), X_scaled.max())

if n_cols != 54:
    raise ValueError(f"Lỗi: số chiều sau scale = {n_cols}, không phải 54")

if n_rows != len(y_encoded):
    raise ValueError("Lỗi: số dòng feature và label không khớp")

# =========================================================
# 7) EXPORT TXT FOR GE-DPC
# =========================================================
np.savetxt(FEATURE_OUT, X_scaled, fmt="%.10f")
np.savetxt(LABEL_OUT, y_encoded, fmt="%d")

with open(INFO_OUT, "w", encoding="utf-8") as f:
    f.write("Covertype full dataset -> GE-DPC\n")
    f.write(f"Input file: {DATA_FILE}\n")
    f.write(f"Rows: {n_rows}\n")
    f.write(f"Dims: {n_cols}\n")
    f.write(f"Unique raw labels: {unique_labels}\n")
    f.write(f"Label mapping: {label_map}\n")
    f.write(f"Feature output: {FEATURE_OUT}\n")
    f.write(f"Label output: {LABEL_OUT}\n")

print("\nDONE")
print("Feature txt:", FEATURE_OUT)
print("Label txt  :", LABEL_OUT)
print("Info txt   :", INFO_OUT)
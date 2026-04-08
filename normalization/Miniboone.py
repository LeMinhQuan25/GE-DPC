# miniboone_to_gedpc.py

from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# 1) CONFIG
# =========================================================
DATA_FILE = Path("/Users/quanle/Downloads/MiniBooNE_PID.txt")
OUT_DIR = Path("/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/data/miniboone")

OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_OUT = OUT_DIR / "miniboone.txt"
LABEL_OUT   = OUT_DIR / "miniboone_label.txt"
INFO_OUT    = OUT_DIR / "miniboone_info.txt"

# =========================================================
# 2) LOAD RAW FILE
# =========================================================
with open(DATA_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Dòng đầu: số mẫu của 2 lớp
header = lines[0].split()
if len(header) < 2:
    raise ValueError("Dòng đầu của MiniBooNE không đúng format mong đợi.")

n_signal = int(float(header[0]))
n_background = int(float(header[1]))

data_lines = lines[1:]
expected_rows = n_signal + n_background

if len(data_lines) != expected_rows:
    raise ValueError(
        f"Số dòng dữ liệu = {len(data_lines)} nhưng expected = {expected_rows}"
    )

# =========================================================
# 3) PARSE FEATURES
# =========================================================
X_list = []
for i, line in enumerate(data_lines):
    parts = line.split()
    row = [float(x) for x in parts]
    X_list.append(row)

X = np.array(X_list, dtype=np.float64)

if X.ndim != 2:
    raise ValueError("X không phải ma trận 2 chiều.")

n_rows, n_cols = X.shape
print("Raw X shape:", X.shape)

if n_cols != 50:
    raise ValueError(f"Số chiều hiện tại = {n_cols}, expected = 50")

# =========================================================
# 4) BUILD LABELS
# =========================================================
# Theo format UCI:
# - n_signal dòng đầu là class 1
# - n_background dòng sau là class 0
# Ta map về 0,1 cho gọn:
# signal -> 1
# background -> 0
y = np.array([1] * n_signal + [0] * n_background, dtype=int)

if len(y) != n_rows:
    raise ValueError("Số dòng X và y không khớp.")

# =========================================================
# 5) SANITY CHECK
# =========================================================
if np.isnan(X).any():
    raise ValueError("Feature có NaN.")

if np.isinf(X).any():
    raise ValueError("Feature có Inf.")

# =========================================================
# 6) MIN-MAX NORMALIZE
# =========================================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.clip(X_scaled, 0.0, 1.0)

# =========================================================
# 7) EXPORT TXT FOR GE-DPC
# =========================================================
np.savetxt(FEATURE_OUT, X_scaled, fmt="%.10f")
np.savetxt(LABEL_OUT, y, fmt="%d")

with open(INFO_OUT, "w", encoding="utf-8") as f:
    f.write("MiniBooNE -> GE-DPC\n")
    f.write(f"Input file: {DATA_FILE}\n")
    f.write(f"Rows: {n_rows}\n")
    f.write(f"Dims: {n_cols}\n")
    f.write(f"Signal: {n_signal}\n")
    f.write(f"Background: {n_background}\n")
    f.write("Label mapping:\n")
    f.write("  background -> 0\n")
    f.write("  signal -> 1\n")
    f.write(f"Feature output: {FEATURE_OUT}\n")
    f.write(f"Label output: {LABEL_OUT}\n")

print("\nDONE")
print("Feature txt:", FEATURE_OUT)
print("Label txt  :", LABEL_OUT)
print("Info txt   :", INFO_OUT)
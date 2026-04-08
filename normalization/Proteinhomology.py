# kddcup04_bio_train_to_gedpc.py

from pathlib import Path
import tarfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# 1) CONFIG
# =========================================================
ARCHIVE_FILE = Path("/Users/quanle/Downloads/data_kddcup04.tar.gz")
INNER_FILE = "bio_train.dat"   # dùng file này vì có label và >100k dòng

OUT_DIR = Path("/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/data/homology")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_OUT = OUT_DIR / "kddcup04_biotxt"
LABEL_OUT   = OUT_DIR / "kddcup04_bio_label.txt"
INFO_OUT    = OUT_DIR / "kddcup04_bio_info.txt"

# =========================================================
# 2) LOAD FROM TAR.GZ
# =========================================================
with tarfile.open(ARCHIVE_FILE, "r:gz") as tar:
    members = tar.getnames()
    if INNER_FILE not in members:
        raise ValueError(f"Không tìm thấy {INNER_FILE} trong archive. Có: {members}")

    f = tar.extractfile(INNER_FILE)
    if f is None:
        raise ValueError(f"Không thể extract {INNER_FILE}")

    # file là tab-separated, không có header
    df = pd.read_csv(f, sep="\t", header=None)

print("Loaded shape:", df.shape)

# bio_train.dat chuẩn: 145751 rows, 77 cols
# = 2 ID + 1 label + 74 features
expected_cols = 77
if df.shape[1] != expected_cols:
    raise ValueError(f"Số cột hiện tại = {df.shape[1]}, expected = {expected_cols}")

# =========================================================
# 3) SPLIT COLUMNS
# =========================================================
# Cột:
# 0 = id1
# 1 = id2
# 2 = label
# 3..76 = 74 features
id_cols = [0, 1]
label_col = 2
feature_cols = list(range(3, 77))

X = df.iloc[:, feature_cols].copy()
y_raw = df.iloc[:, label_col].copy()

print("Feature shape:", X.shape)
print("Label shape  :", y_raw.shape)

if X.shape[1] != 74:
    raise ValueError(f"Số chiều feature hiện tại = {X.shape[1]}, không phải 74")

# =========================================================
# 4) NUMERIC CHECK
# =========================================================
X = X.apply(pd.to_numeric, errors="coerce")
y_raw = pd.to_numeric(y_raw, errors="coerce")

if X.isna().any().any():
    bad_cols = X.columns[X.isna().any()].tolist()
    raise ValueError(f"Feature có NaN ở các cột: {bad_cols}")

if y_raw.isna().any():
    raise ValueError("Label có NaN")

# =========================================================
# 5) LABEL MAP
# =========================================================
unique_labels = sorted(y_raw.unique().tolist())
label_map = {lab: idx for idx, lab in enumerate(unique_labels)}
y = y_raw.map(label_map).astype(int).to_numpy()

print("Unique raw labels:", unique_labels)
print("Label mapping    :", label_map)

# =========================================================
# 6) MIN-MAX NORMALIZE
# =========================================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.astype(np.float64))
X_scaled = np.clip(X_scaled, 0.0, 1.0)

# =========================================================
# 7) FINAL CHECK
# =========================================================
n_rows, n_cols = X_scaled.shape

print("Final X shape:", X_scaled.shape)
print("Final y shape:", y.shape)
print("X min/max    :", X_scaled.min(), X_scaled.max())

if n_cols != 74:
    raise ValueError(f"Lỗi: số chiều sau scale = {n_cols}, không phải 74")

if n_rows != len(y):
    raise ValueError("Lỗi: số dòng X và y không khớp")

# =========================================================
# 8) EXPORT TXT FOR GE-DPC
# =========================================================
np.savetxt(FEATURE_OUT, X_scaled, fmt="%.10f")
np.savetxt(LABEL_OUT, y, fmt="%d")

with open(INFO_OUT, "w", encoding="utf-8") as f:
    f.write("KDD Cup 2004 Protein Homology (bio_train.dat) -> GE-DPC\n")
    f.write(f"Archive file: {ARCHIVE_FILE}\n")
    f.write(f"Inner file: {INNER_FILE}\n")
    f.write(f"Rows: {n_rows}\n")
    f.write(f"Dims: {n_cols}\n")
    f.write(f"Label column: {label_col}\n")
    f.write(f"Feature cols: 3..76\n")
    f.write(f"Unique raw labels: {unique_labels}\n")
    f.write(f"Label mapping: {label_map}\n")
    f.write(f"Feature output: {FEATURE_OUT}\n")
    f.write(f"Label output: {LABEL_OUT}\n")

print("\nDONE")
print("Feature txt:", FEATURE_OUT)
print("Label txt  :", LABEL_OUT)
print("Info txt   :", INFO_OUT)
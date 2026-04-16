# covertype_to_gedpc.py

from pathlib import Path
import gzip
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
DATA_PATH = Path("/Users/quanle/Downloads/covertype/covtype.data.gz")
OUT_DIR = Path("/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/data/covtype")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_OUT = OUT_DIR / "covertype.txt"
LABEL_OUT = OUT_DIR / "covertype_label.txt"
LABEL_MAP_OUT = OUT_DIR / "covertype_label_mapping.txt"

# =========================
# LOAD
# =========================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy file: {DATA_PATH}")

# covtype.data.gz là file CSV nén gzip, không có header
with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
    df = pd.read_csv(f, header=None)

# UCI Covertype:
# - 54 feature
# - 1 label cuối cùng (Cover_Type, từ 1..7)
expected_cols = 55
if df.shape[1] != expected_cols:
    raise ValueError(f"Expected {expected_cols} columns, got {df.shape[1]}")

X = df.iloc[:, :-1].astype(np.float64).to_numpy()
y_raw = df.iloc[:, -1].astype(int).to_numpy()

# đưa label từ 1..7 về 0..6 cho ổn định khi tính metric
unique_labels = sorted(np.unique(y_raw))
label_map = {old: new for new, old in enumerate(unique_labels)}
y = np.array([label_map[v] for v in y_raw], dtype=int)

# kiểm tra shape chuẩn
assert X.shape[0] == len(y), "Số dòng X và y không khớp"
assert X.shape[1] == 54, f"Expected 54 features, got {X.shape[1]}"

# =========================
# SAVE
# =========================
np.savetxt(FEATURE_OUT, X, fmt="%.10f")
np.savetxt(LABEL_OUT, y, fmt="%d")

with open(LABEL_MAP_OUT, "w", encoding="utf-8") as f:
    for old, new in label_map.items():
        f.write(f"{new} = original_cover_type_{old}\n")

print("Done: Covertype")
print("Feature file:", FEATURE_OUT)
print("Label file  :", LABEL_OUT)
print("X shape     :", X.shape)
print("y shape     :", y.shape)
print("Labels      :", sorted(np.unique(y).tolist()))
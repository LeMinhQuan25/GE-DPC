# miniboone_to_gedpc.py

from pathlib import Path
import numpy as np

# =========================
# CONFIG
# =========================
DATA_PATH = Path("/Users/quanle/Downloads/MiniBooNE_PID.txt")
OUT_DIR = Path("/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/data/miniboone")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_OUT = OUT_DIR / "miniboone.txt"
LABEL_OUT = OUT_DIR / "miniboone_label.txt"
LABEL_MAP_OUT = OUT_DIR / "miniboone_label_mapping.txt"

# =========================
# LOAD
# =========================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy file: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    # Dòng đầu: số signal và số background
    first_line = f.readline().strip()
    parts = first_line.split()

    if len(parts) != 2:
        raise ValueError(f"Dòng đầu không đúng format 'n_signal n_background': {first_line}")

    n_signal = int(parts[0])
    n_background = int(parts[1])

    # Các dòng còn lại: mỗi dòng có 50 feature
    X_list = []
    for line_num, line in enumerate(f, start=2):
        line = line.strip()
        if not line:
            continue

        vals = line.split()
        if len(vals) != 50:
            raise ValueError(f"Line {line_num}: expected 50 features, got {len(vals)}")

        row = [float(v) for v in vals]
        X_list.append(row)

X = np.array(X_list, dtype=np.float64)

expected_rows = n_signal + n_background
if X.shape[0] != expected_rows:
    raise ValueError(f"Expected {expected_rows} rows, got {X.shape[0]}")

if X.shape[1] != 50:
    raise ValueError(f"Expected 50 features, got {X.shape[1]}")

# Theo UCI:
# - signal ở trước
# - background ở sau
# Gán:
#   signal = 1
#   background = 0
y = np.concatenate([
    np.ones(n_signal, dtype=int),
    np.zeros(n_background, dtype=int)
])

assert X.shape[0] == len(y), "Số dòng X và y không khớp"

# =========================
# SAVE
# =========================
np.savetxt(FEATURE_OUT, X, fmt="%.10f")
np.savetxt(LABEL_OUT, y, fmt="%d")

with open(LABEL_MAP_OUT, "w", encoding="utf-8") as f:
    f.write("1 = signal (electron neutrino)\n")
    f.write("0 = background (muon neutrino)\n")

print("Done: MiniBooNE")
print("Feature file:", FEATURE_OUT)
print("Label file  :", LABEL_OUT)
print("X shape     :", X.shape)
print("y shape     :", y.shape)
print("Labels      :", sorted(np.unique(y).tolist()))
print("Signal      :", n_signal)
print("Background  :", n_background)
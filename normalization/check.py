from pathlib import Path
import numpy as np

INPUT_FILE = Path("/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/dataset/unlabel/rice+cammeo.txt")
OUTPUT_DIR = Path("/Users/quanle/Documents/Master/Thesis/Dataset/unlabel")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12

def load_feature_txt(file_path: Path):
    data = np.loadtxt(file_path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data

def minmax_normalize(X: np.ndarray):
    col_min = np.min(X, axis=0)
    col_max = np.max(X, axis=0)
    col_range = col_max - col_min

    constant_mask = col_range < EPS
    safe_range = np.where(constant_mask, 1.0, col_range)

    X_norm = (X - col_min) / safe_range
    X_norm[:, constant_mask] = 0.0
    return X_norm

X = load_feature_txt(INPUT_FILE)
X_norm = minmax_normalize(X)

out_file = OUTPUT_DIR / f"{INPUT_FILE.stem}.txt"
np.savetxt(out_file, X_norm, fmt="%.12f")

print(f"Done: {out_file}")
print(f"Shape: {X.shape}")
print(f"Raw min/max: {X.min():.6f} / {X.max():.6f}")
print(f"Norm min/max: {X_norm.min():.6f} / {X_norm.max():.6f}")
from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import inspect
import sys
import tempfile
from types import ModuleType

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


"""
GE-DPC wrapper for Covertype and MiniBooNE.

Mục tiêu:
- giữ nguyên logic gốc của source phát triển
- chỉ vá 2 điểm bên ngoài:
    1) preprocessing theo dataset
    2) center selection ổn định hơn bằng log-gamma

Cách hoạt động:
- import file source gốc
- monkey-patch auto_select_centers(...)
- preprocess feature file ra file tạm
- gọi hàm run gốc hoặc main() của source gốc
"""


# =====================================================================
# 1) USER CONFIG
# =====================================================================
# PHẢI trỏ tới file source gốc thật sự, KHÔNG phải file wrapper này
ORIGINAL_SOURCE_PATH = Path("/Users/quanle/Documents/Master/Thesis/Code/GE-DPC-main/GE-DPC-15-04.py")

THIS_FILE = Path(__file__).resolve()
if ORIGINAL_SOURCE_PATH.resolve() == THIS_FILE:
    raise ValueError(
        "ORIGINAL_SOURCE_PATH is pointing to this wrapper file itself. "
        "Please change it to the real original GE-DPC source file."
    )

DATASET_CONFIG = {
    "covertype": {
        "auto_center_k": 7,
        "min_centers": 7,
        "max_centers": 7,
    },
    "miniboone": {
        "auto_center_k": 2,
        "min_centers": 2,
        "max_centers": 2,
    },
}


# =====================================================================
# 2) PREPROCESSING PATCH
# =====================================================================
def preprocess_features_by_dataset(X: np.ndarray, dataset_name: str) -> np.ndarray:
    """
    Chỉ scale dữ liệu theo đúng đặc tính từng dataset.
    Không thay đổi logic gốc của thuật toán GE-DPC.
    """
    name = str(dataset_name).lower()
    X = np.asarray(X, dtype=np.float64)

    if name == "covertype":
        if X.shape[1] != 54:
            print(f"[WARN] Covertype expected 54 dims, got {X.shape[1]}. Skip special preprocessing.")
            return X

        # Covertype:
        # 10 cột đầu là continuous
        # 44 cột sau là binary indicators
        X_cont = X[:, :10]
        X_bin = X[:, 10:]

        scaler = MinMaxScaler()
        X_cont_scaled = scaler.fit_transform(X_cont)

        X_new = np.hstack([X_cont_scaled, X_bin.astype(np.float64)])
        print("[INFO] Applied Covertype preprocessing: MinMax on first 10 continuous dims; binary dims unchanged.")
        return X_new

    if name == "miniboone":
        scaler = StandardScaler()
        X_new = scaler.fit_transform(X)
        print("[INFO] Applied MiniBooNE preprocessing: StandardScaler on all dims.")
        return X_new

    print(f"[INFO] No special preprocessing for dataset: {dataset_name}")
    return X


# =====================================================================
# 3) CENTER-SELECTION PATCH (LOG-GAMMA)
# =====================================================================
def auto_select_centers_loggamma(
    densities,
    min_dists,
    dist_matrix=None,
    auto_center_mode="knee",
    auto_center_k=None,
    min_centers=2,
    max_centers=None,
    **kwargs,
):
    """
    Drop-in replacement for auto_select_centers(...)

    Giữ nguyên đầu ra:
    - trả về list index center

    Chỉ thay điểm tính score:
    - gamma cũ: density * delta
    - gamma mới: log(1+density) + log(1+delta)

    Mục đích:
    - giảm nổ số cho MiniBooNE
    - ổn định hơn khi xếp hạng center
    """
    densities = np.asarray(densities, dtype=np.float64)
    min_dists = np.asarray(min_dists, dtype=np.float64)

    score = np.log1p(np.maximum(densities, 0.0)) + np.log1p(np.maximum(min_dists, 0.0))
    ranked = np.argsort(-score)

    if auto_center_k is not None:
        k = int(auto_center_k)
    else:
        k = max(int(min_centers), 2)
        if max_centers is not None:
            k = min(k, int(max_centers))

    if max_centers is not None:
        k = min(k, int(max_centers))
    k = max(int(min_centers), int(k))
    k = min(k, len(ranked))

    selected = ranked[:k].tolist()

    print("[PATCH] Using log-gamma center selection")
    print("[PATCH] Selected centers:", selected)
    print("[PATCH] Selected log-gamma values:", [float(score[i]) for i in selected])

    return selected


# =====================================================================
# 4) LOAD ORIGINAL MODULE
# =====================================================================
def load_original_module(source_path: Path) -> ModuleType:
    if not source_path.exists():
        raise FileNotFoundError(f"Original source file not found: {source_path}")

    spec = importlib.util.spec_from_file_location("gedpc_original_module", str(source_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {source_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["gedpc_original_module"] = module
    spec.loader.exec_module(module)
    return module


# =====================================================================
# 5) PATCH ORIGINAL MODULE
# =====================================================================
def patch_original_module(module: ModuleType) -> None:
    if hasattr(module, "auto_select_centers"):
        module.auto_select_centers = auto_select_centers_loggamma
        print("[PATCH] Replaced original auto_select_centers(...) with log-gamma version.")
    else:
        print("[WARN] Original module has no auto_select_centers(...). Center patch was not applied.")


# =====================================================================
# 6) WRITE TEMP PREPROCESSED FEATURE FILE
# =====================================================================
def write_preprocessed_feature_file(feature_file: Path, dataset_name: str) -> Path:
    X = np.loadtxt(feature_file)
    X = preprocess_features_by_dataset(X, dataset_name)

    tmp = tempfile.NamedTemporaryFile(prefix=f"{dataset_name}_pre_", suffix=".txt", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    np.savetxt(tmp_path, X, fmt="%.10f")
    print(f"[INFO] Wrote temporary preprocessed feature file: {tmp_path}")
    return tmp_path


# =====================================================================
# 7) CHOOSE ORIGINAL RUN FUNCTION
# =====================================================================
def choose_run_function(module: ModuleType):
    candidates = [
        "run_ge_dpc_real",
        "run_ge_dpc_highdim_fast",
        "main_run",
        "run",
        "main",
    ]

    for name in candidates:
        if hasattr(module, name) and callable(getattr(module, name)):
            fn = getattr(module, name)
            print(f"[INFO] Using original run function: {name}(...)")
            return fn

    available = [
        name for name in dir(module)
        if callable(getattr(module, name)) and not name.startswith("_")
    ]
    raise AttributeError(
        "Could not find a supported run function in the original source. "
        f"Available callables: {available}"
    )


# =====================================================================
# 8) CALL ORIGINAL RUN FUNCTION
# =====================================================================
def call_run_function(
    module: ModuleType,
    run_fn,
    feature_file: Path,
    label_file: Path,
    dataset_name: str,
    user_kwargs: dict,
):
    sig = inspect.signature(run_fn)
    params = sig.parameters
    cfg = DATASET_CONFIG[dataset_name]

    # ---------------------------------------------------------
    # Case A: source gốc là API style, hàm có tham số
    # ---------------------------------------------------------
    if len(params) > 0:
        call_kwargs = {}

        if "feature_file" in params:
            call_kwargs["feature_file"] = str(feature_file)
        if "label_file" in params:
            call_kwargs["label_file"] = str(label_file)
        if "dataset_name" in params:
            call_kwargs["dataset_name"] = dataset_name

        for key in ["auto_center_k", "min_centers", "max_centers"]:
            if key in params:
                call_kwargs[key] = cfg[key]

        for key, value in user_kwargs.items():
            if key in params:
                call_kwargs[key] = value

        print("[INFO] Final call kwargs:")
        for k, v in call_kwargs.items():
            print(f"  - {k} = {v}")

        return run_fn(**call_kwargs)

    # ---------------------------------------------------------
    # Case B: source gốc là script-style main() không tham số
    # ---------------------------------------------------------
    print("[INFO] Detected script-style main() with no parameters. Setting module globals...")

    setattr(module, "feature_file", str(feature_file))
    setattr(module, "label_file", str(label_file))
    setattr(module, "dataset_name", dataset_name)

    setattr(module, "auto_center_k", cfg["auto_center_k"])
    setattr(module, "min_centers", cfg["min_centers"])
    setattr(module, "max_centers", cfg["max_centers"])

    for key, value in user_kwargs.items():
        setattr(module, key, value)

    print("[INFO] Injected globals into original module:")
    print(f"  - feature_file   = {feature_file}")
    print(f"  - label_file     = {label_file}")
    print(f"  - dataset_name   = {dataset_name}")
    print(f"  - auto_center_k  = {cfg['auto_center_k']}")
    print(f"  - min_centers    = {cfg['min_centers']}")
    print(f"  - max_centers    = {cfg['max_centers']}")

    return run_fn()


# =====================================================================
# 9) MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["covertype", "miniboone"])
    parser.add_argument("--feature_file", required=True)
    parser.add_argument("--label_file", required=True)

    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--outlier_t", type=float, default=2.0)
    parser.add_argument("--approx_rank", type=int, default=16)
    parser.add_argument("--svd_oversamples", type=int, default=8)
    parser.add_argument("--svd_n_iter", type=int, default=2)
    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    feature_file = Path(args.feature_file)
    label_file = Path(args.label_file)

    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    original_module = load_original_module(ORIGINAL_SOURCE_PATH)
    patch_original_module(original_module)

    preprocessed_feature_file = write_preprocessed_feature_file(feature_file, dataset_name)
    run_fn = choose_run_function(original_module)

    user_kwargs = {
        "epsilon": args.epsilon,
        "outlier_t": args.outlier_t,
        "approx_rank": args.approx_rank,
        "svd_oversamples": args.svd_oversamples,
        "svd_n_iter": args.svd_n_iter,
    }

    call_run_function(
        module=original_module,
        run_fn=run_fn,
        feature_file=preprocessed_feature_file,
        label_file=label_file,
        dataset_name=dataset_name,
        user_kwargs=user_kwargs,
    )


if __name__ == "__main__":
    main()
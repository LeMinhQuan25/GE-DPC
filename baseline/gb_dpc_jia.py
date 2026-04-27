"""
Run GB-DPC code on 4 datasets in one batch.

Datasets:
- breast_cancer
- hcv_data
- dry_bean
- rice_cammeo

This file keeps the original GB-DPC logic:
1) Generate granular balls by repeated 2-means splitting.
2) Compute GB center, radius, quality, mean radius.
3) Compute density, distance, delta / nearest higher-density GB.
4) Select cluster centers automatically by top-k gamma = density * delta,
   where k is inferred from the number of ground-truth classes.
5) Assign GB labels by DPC propagation.
6) Map GB labels back to samples and evaluate ACC/NMI/ARI.

Notes:
- Plotting and manual rectangle selection are disabled so the script can run all datasets automatically.
- Ground-truth labels are NOT used during clustering, only to infer k and calculate final metrics.
- Index tracking is used only to map GB labels back to original samples efficiently; it does not change clustering logic.
"""

import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import k_means
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder


# ============================================================
# Evaluation
# ============================================================
def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    return cm[row_ind, col_ind].sum() / y_true.size


# ============================================================
# Original GB-DPC helper functions
# ============================================================
def get_num(gb: np.ndarray) -> int:
    return gb.shape[0]


def calculate_center_and_radius(gb: np.ndarray) -> Tuple[np.ndarray, float]:
    center = gb.mean(axis=0)
    radius = np.max((((gb - center) ** 2).sum(axis=1) ** 0.5))
    return center, float(radius)


def splits_ball_with_indices(
    gb: np.ndarray,
    indices: np.ndarray,
    splitting_method: str = "2-means",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits_k = 2
    unique_points = np.unique(gb, axis=0)

    if splitting_method == "2-means":
        if unique_points.shape[0] < splits_k:
            splits_k = unique_points.shape[0]
        if splits_k <= 1:
            return [(gb, indices)]
        labels = k_means(X=gb, n_clusters=splits_k, n_init=3, random_state=8)[1]
    else:
        raise ValueError("Only splitting_method='2-means' is supported.")

    ball_list: List[Tuple[np.ndarray, np.ndarray]] = []
    for single_label in range(splits_k):
        mask = labels == single_label
        if np.any(mask):
            ball_list.append((gb[mask, :], indices[mask]))
    return ball_list


def splits_with_indices(
    gb_items: Sequence[Tuple[np.ndarray, np.ndarray]],
    num: int,
    splitting_method: str = "2-means",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    gb_list_new: List[Tuple[np.ndarray, np.ndarray]] = []
    for gb, idx in gb_items:
        p = get_num(gb)
        if p < num:
            gb_list_new.append((gb, idx))
        else:
            gb_list_new.extend(splits_ball_with_indices(gb, idx, splitting_method))
    return gb_list_new


def get_ball_quality(gb: np.ndarray, center: np.ndarray) -> Tuple[int, float]:
    n = gb.shape[0]
    ball_quality = n
    mean_r = np.mean(((gb - center) ** 2) ** 0.5)
    return ball_quality, float(mean_r)


def ball_density2(radiusAD: np.ndarray, ball_qualitysA: np.ndarray, mean_rs: np.ndarray) -> np.ndarray:
    n = radiusAD.shape[0]
    ball_dens2 = np.zeros(shape=n, dtype=float)
    for i in range(n):
        if radiusAD[i] == 0:
            ball_dens2[i] = 0.0
        else:
            ball_dens2[i] = ball_qualitysA[i] / (radiusAD[i] * radiusAD[i] * mean_rs[i])
    return ball_dens2


def ball_distance(centersAD: np.ndarray) -> np.ndarray:
    return squareform(pdist(centersAD))


def ball_min_dist(ball_distS: np.ndarray, ball_densS: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = ball_distS.shape[0]
    ball_min_distAD = np.zeros(shape=n, dtype=float)
    ball_nearestAD = np.zeros(shape=n, dtype=int)

    index_ball_dens = np.argsort(-ball_densS)
    for i3, index in enumerate(index_ball_dens):
        if i3 == 0:
            continue
        index_ball_higher_dens = index_ball_dens[:i3]
        dists = [ball_distS[index, j] for j in index_ball_higher_dens]
        ball_min_distAD[index] = np.min(dists)
        ball_index_near = np.argmin(dists)
        ball_nearestAD[index] = int(index_ball_higher_dens[ball_index_near])

    ball_min_distAD[index_ball_dens[0]] = np.max(ball_min_distAD)
    if np.max(ball_min_distAD) < 1:
        ball_min_distAD = ball_min_distAD * 10
    return ball_min_distAD, ball_nearestAD


def auto_find_centers_by_gamma(ball_densS: np.ndarray, ball_min_distS: np.ndarray, k: int) -> np.ndarray:
    """Automatic replacement for manual rectangle selection, for batch runs."""
    gamma = ball_densS * ball_min_distS
    k = int(max(1, min(k, len(gamma))))
    return np.argsort(-gamma)[:k]


def ball_cluster(ball_densS: np.ndarray, ball_centers: np.ndarray, ball_nearest: np.ndarray) -> np.ndarray:
    k = len(ball_centers)
    if k == 0:
        raise RuntimeError("No centers were selected.")

    n = ball_densS.shape[0]
    ball_labs = -1 * np.ones(n, dtype=int)

    for i5, cen1 in enumerate(ball_centers):
        ball_labs[cen1] = int(i5)

    ball_index_density = np.argsort(-ball_densS)
    for index2 in ball_index_density:
        if ball_labs[index2] == -1:
            ball_labs[index2] = ball_labs[int(ball_nearest[index2])]

    return ball_labs


# ============================================================
# Dataset registry
# ============================================================
def maybe_extract_dataset_zip(base_dir: Path) -> None:
    dataset_dir = base_dir / "dataset"
    zip_path = base_dir / "dataset.zip"

    if dataset_dir.exists():
        return
    if zip_path.exists():
        print(f"[INFO] Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base_dir)


def get_4_dataset_registry(base_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    return {
        "breast_cancer": (
            base_dir / "dataset" / "unlabel" / "breast_cancer.txt",
            base_dir / "dataset" / "label" / "breast_cancer_label.txt",
        ),
        "hcv_data": (
            base_dir / "dataset" / "unlabel" / "hcv_data.txt",
            base_dir / "dataset" / "label" / "hcv_data_label.txt",
        ),
        "dry_bean": (
            base_dir / "dataset" / "unlabel" / "dry_bean.txt",
            base_dir / "dataset" / "label" / "dry_bean_label.txt",
        ),
        "rice_cammeo": (
            base_dir / "dataset" / "unlabel" / "rice+cammeo.txt",
            base_dir / "dataset" / "label" / "rice+cammeo_label.txt",
        ),
    }


# ============================================================
# Main run functions
# ============================================================
def run_one_dataset(dataset_name: str, feature_file: Path, label_file: Path) -> Dict[str, object]:
    X = np.loadtxt(feature_file, dtype=float)
    y = np.loadtxt(label_file)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y = y.astype(int)
    y_true = LabelEncoder().fit_transform(y)
    k = len(np.unique(y_true))

    print("=" * 100)
    print(f"Running dataset: {dataset_name}")
    print(f"Feature file   : {feature_file}")
    print(f"Label file     : {label_file}")
    print(f"Data shape     : n={X.shape[0]}, d={X.shape[1]}, k={k}")
    print("=" * 100)

    # 1) Generate granular balls
    time_gb_start = time.time()
    num = int(np.ceil(np.sqrt(X.shape[0])))
    splitting_method = "2-means"
    gb_items: List[Tuple[np.ndarray, np.ndarray]] = [(X, np.arange(X.shape[0]))]

    iteration = 0
    while True:
        iteration += 1
        current_count = len(gb_items)
        gb_items = splits_with_indices(gb_items, num, splitting_method)
        print(f"GB count after split iteration {iteration}: {len(gb_items)}")
        if len(gb_items) == current_count:
            break

    time_gb_gen = time.time() - time_gb_start

    # 2) Compute GB attributes
    time_attr_start = time.time()
    centersAD = []
    radiusAD = []
    ball_qualitysA = []
    mean_rs = []

    for gb, _ in gb_items:
        center, radius = calculate_center_and_radius(gb)
        ball_quality, mean_r = get_ball_quality(gb, center)
        centersAD.append(center)
        radiusAD.append(radius)
        ball_qualitysA.append(ball_quality)
        mean_rs.append(mean_r)

    centersAD = np.array(centersAD)
    radiusAD = np.array(radiusAD)
    ball_qualitysA = np.array(ball_qualitysA)
    mean_rs = np.array(mean_rs)

    ball_dens2 = ball_density2(radiusAD, ball_qualitysA, mean_rs)
    ball_distAD = ball_distance(centersAD)
    ball_min_distAD, ball_nearestAD = ball_min_dist(ball_distAD, ball_dens2)
    time_attr_calc = time.time() - time_attr_start

    # 3) Select centers and cluster
    time_core_start = time.time()
    ball_centers = auto_find_centers_by_gamma(ball_dens2, ball_min_distAD, k=k)
    labels = ball_cluster(ball_dens2, ball_centers, ball_nearestAD)
    time_core_cluster = time.time() - time_core_start

    # 4) Map GB labels back to original data points, excluded from runtime
    y_pred = np.zeros(len(X), dtype=int)
    for i, (_, indices) in enumerate(gb_items):
        y_pred[indices] = labels[i]

    # 5) Evaluate, excluded from runtime
    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    time_total_cluster = time_attr_calc + time_core_cluster
    time_total_effective = time_gb_gen + time_total_cluster

    print("-" * 100)
    print(f"Selected centers: {ball_centers.tolist()}")
    print(f"ACC: {acc:.3f}")
    print(f"NMI: {nmi:.3f}")
    print(f"ARI: {ari:.3f}")
    print("-" * 100)
    print("Runtime statistics")
    print(f"1. GB generation time      : {time_gb_gen * 1000:.6f} ms")
    print(f"2. Attribute compute time  : {time_attr_calc * 1000:.6f} ms")
    print(f"3. Core clustering time    : {time_core_cluster * 1000:.6f} ms")
    print(f"Total clustering time      : {time_total_cluster * 1000:.6f} ms")
    print(f"Total effective runtime    : {time_total_effective * 1000:.6f} ms")
    print("=" * 100)

    return {
        "dataset": dataset_name,
        "n": X.shape[0],
        "d": X.shape[1],
        "k": k,
        "acc": acc,
        "nmi": nmi,
        "ari": ari,
        "gb_count": len(gb_items),
        "generation_time": time_gb_gen,
        "attribute_time": time_attr_calc,
        "cluster_time": time_core_cluster,
        "total_cluster_time": time_total_cluster,
        "total_time": time_total_effective,
    }


def run_all_4_datasets(base_dir: Path, dataset_names: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, object]]:
    maybe_extract_dataset_zip(base_dir)
    registry = get_4_dataset_registry(base_dir)

    if dataset_names is None:
        dataset_names = ["breast_cancer", "hcv_data", "dry_bean", "rice_cammeo"]

    results: Dict[str, Dict[str, object]] = {}

    for name in dataset_names:
        try:
            feature_file, label_file = registry[name]
            if not feature_file.exists() or not label_file.exists():
                raise FileNotFoundError(f"Missing files: {feature_file} or {label_file}")
            results[name] = run_one_dataset(name, feature_file, label_file)
        except Exception as exc:
            results[name] = {"error": str(exc)}
            print(f"[ERROR] {name}: {exc}")

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(
        f"{'Dataset':<18} {'n':>8} {'d':>5} {'k':>4} "
        f"{'ACC':>8} {'NMI':>8} {'ARI':>8} {'GBs':>8} "
        f"{'Gen(ms)':>12} {'Attr(ms)':>12} {'Clus(ms)':>12} {'Total(ms)':>12}"
    )
    print("-" * 120)

    for name, res in results.items():
        if "error" in res:
            print(f"{name:<18} ERROR: {res['error']}")
        else:
            print(
                f"{name:<18} "
                f"{res['n']:>8} "
                f"{res['d']:>5} "
                f"{res['k']:>4} "
                f"{res['acc']:>8.3f} "
                f"{res['nmi']:>8.3f} "
                f"{res['ari']:>8.3f} "
                f"{res['gb_count']:>8} "
                f"{res['generation_time'] * 1000:>12.6f} "
                f"{res['attribute_time'] * 1000:>12.6f} "
                f"{res['cluster_time'] * 1000:>12.6f} "
                f"{res['total_time'] * 1000:>12.6f}"
            )

    print("=" * 120)
    return results


if __name__ == "__main__":
    # Case 1: Put this file directly inside GE-DPC-main, alongside the dataset/ folder.
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Case 2: If this file is inside GE-DPC-main/scripts, use this instead:
    # BASE_DIR = Path(__file__).resolve().parent.parent

    run_all_4_datasets(BASE_DIR)

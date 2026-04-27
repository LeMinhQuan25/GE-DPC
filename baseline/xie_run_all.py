"""
GB-DPC / Granular Ball DPC - Run All Datasets
=============================================
Purpose:
- Keep the original granular-ball generation and DPC clustering logic.
- Remove manual decision-graph selection so the code can run all datasets in one pass.
- Print detailed result per dataset and one SUMMARY table at the end.

Folder assumption:
GE-DPC-main/
  real_dataset_and_label/
    real_datasets/
      Iris.txt, Seed.txt, segment_3.txt, ...
    real_datasets_label/
      Iris_label.txt, Seed_label.txt, segment_3_label.txt, ...
  dataset/
    unlabel/
      breast_cancer.txt, hcv_data.txt, dry_bean.txt, rice+cammeo.txt
    label/
      breast_cancer_label.txt, hcv_data_label.txt, dry_bean_label.txt, rice+cammeo_label.txt

Run:
python3 gb_dpc_run_all_datasets.py
"""

import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder


# ============================================================
# Granular Ball - same core logic as original code
# Difference: keep point indices for fast and correct label mapping
# ============================================================
class GranularBall:
    def __init__(self, points: np.ndarray, indices: np.ndarray):
        self.points = np.asarray(points, dtype=float)
        self.indices = np.asarray(indices, dtype=int)
        self.size = len(self.points)

        if self.size > 0:
            self.center = np.mean(self.points, axis=0)
            self.radius = self.calculate_radius()
            self.dm = self.calculate_dm()
        else:
            self.center = None
            self.radius = 0.0
            self.dm = 0.0

    def calculate_radius(self) -> float:
        if self.size == 0:
            return 0.0
        distances = np.linalg.norm(self.points - self.center, axis=1)
        return float(np.max(distances))

    def calculate_dm(self) -> float:
        if self.size == 0:
            return 0.0
        distances = np.linalg.norm(self.points - self.center, axis=1)
        return float(np.sum(distances) / self.size)

    def find_farthest_points(self):
        if self.size < 2:
            return None, None

        idx1 = np.argmin(np.linalg.norm(self.points - self.center, axis=1))
        point1 = self.points[idx1]

        distances = np.linalg.norm(self.points - point1, axis=1)
        idx2 = np.argmax(distances)
        point2 = self.points[idx2]

        distances = np.linalg.norm(self.points - point2, axis=1)
        idx3 = np.argmax(distances)
        point3 = self.points[idx3]

        return point2, point3

    def split(self):
        if self.size < 6:
            return [self]

        center1, center2 = self.find_farthest_points()
        if center1 is None or center2 is None:
            return [self]

        dist_to_c1 = np.linalg.norm(self.points - center1, axis=1)
        dist_to_c2 = np.linalg.norm(self.points - center2, axis=1)
        labels = np.where(dist_to_c1 < dist_to_c2, 0, 1)

        mask1 = labels == 0
        mask2 = labels == 1

        if np.sum(mask1) < 3 or np.sum(mask2) < 3:
            return [self]

        return [
            GranularBall(self.points[mask1], self.indices[mask1]),
            GranularBall(self.points[mask2], self.indices[mask2]),
        ]


def algorithm1_generation_of_granular_balls(data: np.ndarray):
    GB_sets = [GranularBall(data, np.arange(len(data)))]

    # Stage 1: DM-based split
    while True:
        new_GB_sets = []
        changed = False

        for ball in GB_sets:
            if ball.size < 3:
                new_GB_sets.append(ball)
                continue

            DMA = ball.dm
            child_balls = ball.split()

            if len(child_balls) == 1:
                new_GB_sets.append(ball)
                continue

            ball1, ball2 = child_balls
            n1, n2 = ball1.size, ball2.size
            total_size = n1 + n2
            DMweight = (n1 / total_size) * ball1.dm + (n2 / total_size) * ball2.dm

            if DMweight < DMA:
                new_GB_sets.extend([ball1, ball2])
                changed = True
            else:
                new_GB_sets.append(ball)

        GB_sets = new_GB_sets
        if not changed:
            break

    # Stage 2: radius-based split
    MIN_RADIUS = 1e-5
    while True:
        new_GB_sets = []
        changed = False

        radii = [ball.radius for ball in GB_sets if ball.size > 0 and ball.radius > 0]
        if len(radii) == 0:
            break

        mean_r = np.mean(radii)
        median_r = np.median(radii)
        threshold = 2 * max(mean_r, median_r)

        for ball in GB_sets:
            if ball.size == 0:
                new_GB_sets.append(ball)
                continue

            if ball.radius >= threshold and ball.radius > MIN_RADIUS:
                child_balls = ball.split()
                if len(child_balls) > 1:
                    changed = True
                new_GB_sets.extend(child_balls)
            else:
                new_GB_sets.append(ball)

        GB_sets = new_GB_sets
        if not changed:
            break

    return GB_sets


# ============================================================
# DPC utilities - same idea as original code
# ============================================================
def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    return float(cm[row_ind, col_ind].sum() / y_true.size)


def ball_distance(centers: np.ndarray) -> np.ndarray:
    if len(centers) <= 1:
        return np.zeros((len(centers), len(centers)), dtype=float)
    return squareform(pdist(centers))


def ball_min_dist(ball_dist: np.ndarray, ball_dens: np.ndarray):
    n = ball_dist.shape[0]
    min_dist = np.zeros(n, dtype=float)
    nearest = np.zeros(n, dtype=int)

    sorted_idx = np.argsort(-ball_dens)
    for i in range(1, n):
        idx = sorted_idx[i]
        higher_dens_idx = sorted_idx[:i]
        min_dist[idx] = np.min(ball_dist[idx, higher_dens_idx])
        min_idx = np.argmin(ball_dist[idx, higher_dens_idx])
        nearest[idx] = higher_dens_idx[min_idx]

    if n > 0:
        min_dist[sorted_idx[0]] = np.max(min_dist)
    return min_dist, nearest


def auto_find_centers_by_top_gamma(ball_dens: np.ndarray, ball_min_dist_arr: np.ndarray, k: int) -> np.ndarray:
    """
    Batch replacement for manual rectangle selection.
    It selects top-k granular balls by gamma = density * delta.
    This is only to automate all-dataset execution.
    """
    n = len(ball_dens)
    if n == 0:
        return np.array([], dtype=int)
    k = int(max(1, min(k, n)))
    gamma = ball_dens * ball_min_dist_arr
    return np.argsort(-gamma)[:k]


def ball_cluster(ball_dens: np.ndarray, ball_centers: np.ndarray, ball_nearest: np.ndarray) -> np.ndarray:
    n = len(ball_dens)
    labels = -np.ones(n, dtype=int)

    if len(ball_centers) == 0:
        labels[:] = 0
        return labels

    for i, center in enumerate(ball_centers):
        labels[int(center)] = i

    sorted_idx = np.argsort(-ball_dens)
    for idx in sorted_idx:
        if labels[idx] == -1:
            labels[idx] = labels[ball_nearest[idx]]

    return labels


# ============================================================
# Dataset registry
# ============================================================
def get_default_dataset_registry(base_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    return {
        "iris": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "Iris.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "Iris_label.txt",
        ),
        "seed": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "Seed.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "Seed_label.txt",
        ),
        "segment_3": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "segment_3.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "segment_3_label.txt",
        ),
        "landsat_2": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "landsat_2.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "landsat_2_label.txt",
        ),
        "msplice_2": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "msplice_2.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "msplice_2_label.txt",
        ),
        "rice": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "rice.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "rice_label.txt",
        ),
        "banknote": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "banknote.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "banknote_label.txt",
        ),
        "htru2": (
            base_dir / "real_dataset_and_label" / "real_datasets" / "htru2.txt",
            base_dir / "real_dataset_and_label" / "real_datasets_label" / "htru2_label.txt",
        ),
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
# One dataset runner
# ============================================================
def run_gb_dpc_dataset(feature_file: Path, label_file: Path, dataset_name: str) -> Dict[str, object]:
    X = np.loadtxt(feature_file, dtype=float)
    y = np.loadtxt(label_file)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    le = LabelEncoder()
    y_true = le.fit_transform(y)
    k = len(np.unique(y_true))

    print("=" * 90)
    print(f"Running dataset: {dataset_name}")
    print(f"Feature file: {feature_file}")
    print(f"Label file  : {label_file}")
    print(f"Data shape  : n={X.shape[0]}, d={X.shape[1]}, classes={k}")
    print("=" * 90)

    # 1. Generate granular balls
    t_gen_start = time.time()
    gb_objects = algorithm1_generation_of_granular_balls(X)
    time_gen = time.time() - t_gen_start

    print(f"Generated granular balls: {len(gb_objects)}")

    # 2. Extract attributes
    t_attr_start = time.time()
    valid_balls = [ball for ball in gb_objects if ball.size > 0]
    centers_arr = np.array([ball.center for ball in valid_balls], dtype=float)
    sizes_arr = np.array([ball.size for ball in valid_balls], dtype=float)
    time_attr = time.time() - t_attr_start

    # 3. DPC clustering
    t_cluster_start = time.time()
    ball_dens = sizes_arr
    dist_mat = ball_distance(centers_arr)
    min_dist_arr, nearest_arr = ball_min_dist(dist_mat, ball_dens)
    ball_centers = auto_find_centers_by_top_gamma(ball_dens, min_dist_arr, k=k)
    ball_labels = ball_cluster(ball_dens, ball_centers, nearest_arr)
    time_cluster = time.time() - t_cluster_start

    # 4. Map granular-ball labels back to data points
    y_pred = np.full(len(X), -1, dtype=int)
    for i, ball in enumerate(valid_balls):
        y_pred[ball.indices] = ball_labels[i]

    if np.any(y_pred == -1):
        raise RuntimeError("Some data points were not assigned labels.")

    # 5. Evaluation
    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    total_time = time_gen + time_attr + time_cluster

    print("-" * 90)
    print("Evaluation metrics")
    print(f"ACC: {acc:.3f}")
    print(f"NMI: {nmi:.3f}")
    print(f"ARI: {ari:.3f}")
    print("-" * 90)
    print("Runtime statistics")
    print(f"1. Granular-ball generation time : {time_gen * 1000:.6f} ms")
    print(f"2. Attribute computation time    : {time_attr * 1000:.6f} ms")
    print(f"3. Clustering process time       : {time_cluster * 1000:.6f} ms")
    print(f"Total program time               : {total_time * 1000:.6f} ms")
    print("=" * 90)

    return {
        "dataset": dataset_name,
        "acc": acc,
        "nmi": nmi,
        "ari": ari,
        "n": X.shape[0],
        "d": X.shape[1],
        "k": k,
        "n_granular_balls": len(valid_balls),
        "generation_time": time_gen,
        "attribute_time": time_attr,
        "cluster_time": time_cluster,
        "total_time": total_time,
    }


def run_named_dataset(dataset_name: str, base_dir: Path) -> Dict[str, object]:
    registry = get_default_dataset_registry(base_dir)
    key = dataset_name.lower()

    if key not in registry:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {list(registry.keys())}")

    feature_file, label_file = registry[key]
    if not feature_file.exists() or not label_file.exists():
        raise FileNotFoundError(
            f"Dataset files not found for '{dataset_name}'.\n"
            f"Feature: {feature_file}\n"
            f"Label  : {label_file}"
        )

    return run_gb_dpc_dataset(feature_file, label_file, key)


def run_all_default_datasets(
    base_dir: Path,
    dataset_names: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, object]]:
    if dataset_names is None:
        dataset_names = [
            "iris", "seed", "segment_3", "landsat_2",
            "msplice_2", "rice", "banknote", "htru2",
            "breast_cancer", "hcv_data", "dry_bean", "rice_cammeo",
        ]

    results: Dict[str, Dict[str, object]] = {}

    for name in dataset_names:
        try:
            results[name] = run_named_dataset(name, base_dir)
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
                f"{res['n_granular_balls']:>8} "
                f"{res['generation_time'] * 1000:>12.6f} "
                f"{res['attribute_time'] * 1000:>12.6f} "
                f"{res['cluster_time'] * 1000:>12.6f} "
                f"{res['total_time'] * 1000:>12.6f}"
            )

    print("=" * 120)
    return results


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # Case 1: this file is directly inside GE-DPC-main
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Case 2: if this file is inside GE-DPC-main/scripts, use this instead:
    # BASE_DIR = Path(__file__).resolve().parent.parent

    dataset_names = [
        "iris", "seed", "segment_3", "landsat_2",
        "msplice_2", "rice", "banknote", "htru2",
        "breast_cancer", "hcv_data", "dry_bean", "rice_cammeo",
    ]

    run_all_default_datasets(
        base_dir=BASE_DIR,
        dataset_names=dataset_names,
    )

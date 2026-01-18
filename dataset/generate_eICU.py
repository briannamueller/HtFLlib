#!/usr/bin/env python
"""Generate per-hospital eICU clients with pre-fused feature tensors."""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sparse
from sklearn.model_selection import train_test_split

from utils.dataset_utils import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare hospital clients for eICU.")
    parser.add_argument("--task", default="mortality_48h")
    parser.add_argument("--data-dir", default="/Users/brimueller/eICU/data")
    parser.add_argument("--output-root", default="dataset")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--partition-id", default="")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument(
        "--min-size",
        "--min-samples",
        dest="min_size",
        type=int,
        default=10,
        help="Minimum number of stays required to keep a hospital client.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--prefer-positive", action="store_true")
    parser.add_argument(
        "--client-sort-mode",
        choices=("positives", "prevalence"),
        default="positives",
        help="How to rank hospitals when selecting top-k clients.",
    )
    return parser.parse_args()


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_partition_id(args):
    dataset_name = args.dataset_name or f"eICU_task=[{args.task}]"
    return (
        f"eicu[task={args.task},min_size={args.min_size}]"
        f"_nc[{args.num_clients}]_tr[{_fmt(args.train_ratio)}]_s[{args.seed}]"
    )


def load_population_frame(data_dir: Path, task: str) -> Tuple[pd.DataFrame, np.ndarray]:
    population_path = data_dir / "population" / f"{task}.csv"
    patient_path = data_dir / "patient.csv"

    df_pop = pd.read_csv(population_path).reset_index(drop=True)
    df_patient = pd.read_csv(patient_path)
    df_patient = df_patient.rename(
        columns={
            "patientunitstayid": "ID",
            "uniquepid": "SUBJECT_ID",
            "hospitalid": "hospital_id",
        }
    )

    df = df_pop.merge(df_patient[["ID", "SUBJECT_ID", "hospital_id"]], on="ID", how="left")
    mask = df["hospital_id"].notna()
    valid_idx = np.flatnonzero(mask.to_numpy())
    df = df.loc[mask].reset_index(drop=True)
    df["hospital_id"] = df["hospital_id"].astype(int)
    return df, valid_idx


def select_one_stay_per_patient(df: pd.DataFrame, label_col: str) -> np.ndarray:
    def pick_row(group: pd.DataFrame) -> int:
        positives = group[group[label_col] == 1]
        if not positives.empty:
            return int(positives.index[0])
        return int(group.index[0])

    selected_indices = (
        df.groupby("SUBJECT_ID", sort=False)
        .apply(pick_row)
        .to_numpy(dtype=np.int64)
    )
    selected_indices.sort()
    return selected_indices


def load_feature_matrices(data_dir: Path, task: str) -> Tuple[np.ndarray, np.ndarray]:
    feature_dir = data_dir / "features" / task
    s = sparse.load_npz(feature_dir / "s.npz").todense()
    X = sparse.load_npz(feature_dir / "X.npz").todense()
    return np.asarray(s, dtype=np.float32), np.asarray(X, dtype=np.float32)


def _fuse_sample(x_row: np.ndarray, s_row: np.ndarray) -> np.ndarray:
    ts = np.asarray(x_row, dtype=np.float32, order="C")
    static = np.asarray(s_row, dtype=np.float32, order="C")

    assert ts.ndim == 2, f"Expected time series shape (steps, features), got {ts.shape}"
    assert static.ndim == 1, f"Expected 1D static vector, got {static.shape}"

    static_expanded = np.tile(static, (ts.shape[0], 1))
    return np.hstack((ts, static_expanded)).astype(np.float32, copy=False)


def build_clients(
    df: pd.DataFrame,
    s_matrix: np.ndarray,
    X_matrix: np.ndarray,
    label_col: str,
    train_ratio: float,
    min_size: int,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], List[List[Tuple[int, int]]], List[int]]:
    train_entries: List[Dict[str, np.ndarray]] = []
    test_entries: List[Dict[str, np.ndarray]] = []
    stats: List[List[Tuple[int, int]]] = []
    client_ids: List[int] = []

    grouped = df.groupby("hospital_id", sort=False)
    for hospital_id, group in grouped:
        stay_indices = group.index.to_numpy(dtype=np.int64)
        labels = group[label_col].to_numpy(dtype=np.int64)
        if len(stay_indices) < max(min_size, 2):
            continue

        try:
            random_state = int(rng.integers(0, np.iinfo(np.int32).max))
            tr_idx, te_idx = train_test_split(
                stay_indices,
                train_size=train_ratio,
                stratify=labels,
                random_state=random_state,
            )
        except ValueError:
            continue

        x_train = [_fuse_sample(X_matrix[i], s_matrix[i]) for i in tr_idx]
        y_train = df.loc[tr_idx, label_col].to_numpy(dtype=np.int64)

        x_test = [_fuse_sample(X_matrix[i], s_matrix[i]) for i in te_idx]
        y_test = df.loc[te_idx, label_col].to_numpy(dtype=np.int64)

        train_entries.append({"x": x_train, "y": y_train})
        test_entries.append({"x": x_test, "y": y_test})

        label_counts = group[label_col].value_counts().sort_index()
        stats.append([(int(lbl), int(cnt)) for lbl, cnt in label_counts.items()])
        client_ids.append(int(hospital_id))

    if not train_entries:
        raise RuntimeError("No hospitals satisfied the splitting requirements.")

    return train_entries, test_entries, stats, client_ids


def choose_top_clients(
    train_entries: List[Dict[str, np.ndarray]],
    test_entries: List[Dict[str, np.ndarray]],
    stats: List[List[Tuple[int, int]]],
    client_ids: List[int],
    top_k: int,
    *,
    sort_mode: str = "positives",
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], List[List[Tuple[int, int]]], List[int]]:
    positives = [int(entry["y"].sum()) for entry in train_entries]
    if sort_mode == "prevalence":
        scores = [
            (pos / max(1, int(entry["y"].shape[0])))
            for pos, entry in zip(positives, train_entries)
        ]
    else:
        scores = positives
    order = np.argsort(scores)[::-1][:top_k]
    return (
        [train_entries[i] for i in order],
        [test_entries[i] for i in order],
        [stats[i] for i in order],
        [client_ids[i] for i in order],
    )


def clear_directory(dir_path: Path) -> None:
    if dir_path.exists():
        for npz in dir_path.glob("*.npz"):
            npz.unlink()
    else:
        dir_path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name or f"eICU_task=[{args.task}]"
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    dataset_root = output_root / dataset_name

    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data_dir)
    label_base = args.task.split("_")[0]
    label_col = f"{label_base}_LABEL"

    df, valid_idx = load_population_frame(data_dir, args.task)
    s_matrix, X_matrix = load_feature_matrices(data_dir, args.task)
    s_matrix = s_matrix[valid_idx]
    X_matrix = X_matrix[valid_idx]

    if args.prefer_positive:
        selected = select_one_stay_per_patient(df, label_col)
        df = df.iloc[selected].reset_index(drop=True)
        s_matrix = s_matrix[selected]
        X_matrix = X_matrix[selected]
        print(f"[info] Reduced to {len(df)} unique patients (positive-preference enabled).")

    train_entries, test_entries, stats, client_ids = build_clients(
        df=df,
        s_matrix=s_matrix,
        X_matrix=X_matrix,
        label_col=label_col,
        train_ratio=args.train_ratio,
        min_size=args.min_size,
        rng=rng,
    )

    if args.num_clients is not None and args.num_clients < len(train_entries):
        train_entries, test_entries, stats, client_ids = choose_top_clients(
            train_entries,
            test_entries,
            stats,
            client_ids,
            args.num_clients,
            sort_mode=args.client_sort_mode,
        )

    num_clients_actual = len(train_entries)
    num_classes = int(df[label_col].nunique())

    split_config = {
        "dataset_name": dataset_name,
        "task": args.task,
        "data_dir": str(args.data_dir),
        "train_ratio": args.train_ratio,
        "min_size": args.min_size,
        "min_samples": args.min_size,
        "seed": args.seed,
        "num_clients_requested": args.num_clients,
        "num_clients_actual": num_clients_actual,
        "prefer_positive": args.prefer_positive,
        "client_sort_mode": args.client_sort_mode,
        "client_ids": client_ids,
        "num_classes": num_classes,
    }

    partition_id = args.partition_id or build_partition_id(args)
    split_config["partition_id"] = partition_id
    dataset_dir = dataset_root / partition_id
    train_path = dataset_dir / "train"
    test_path = dataset_dir / "test"
    config_path = dataset_dir / "config.json"

    if config_path.exists():
        print(
            f"\nDataset already generated at {dataset_dir} (partition_id={partition_id})."
        )
        print(
            "Point your experiment at --dataset "
            f"{dataset_name} --data-partition {partition_id}."
        )
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    clear_directory(train_path)
    clear_directory(test_path)

    save_file(
        str(config_path),
        str(train_path),
        str(test_path),
        train_entries,
        test_entries,
        num_clients_actual,
        num_classes,
        stats,
        split_config=split_config,
    )

    try:
        relative_path = dataset_dir.relative_to(repo_root)
    except ValueError:
        relative_path = dataset_dir

    print(f"\nGenerated {dataset_name} split with partition_id {partition_id} at {dataset_dir}.")
    print(f"Relative path: {relative_path}")
    print(
        f"Point your experiment at --dataset {dataset_name} --data-partition {partition_id}."
    )


if __name__ == "__main__":
    main()

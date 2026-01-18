from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ROOT = REPO_ROOT / "system"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from system.des.helpers import build_dataset_partition_id, build_eicu_partition_id


def _proto_config_id(proto_min_samples: int, proto_max_k: int) -> str:
    payload = json.dumps(
        {"proto_min_samples": int(proto_min_samples), "proto_max_k": int(proto_max_k)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def _load_bundle(bundle_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    graph_data = torch.load(bundle_path, map_location="cpu")
    tr = graph_data["train"]
    val = graph_data["val"]
    return tr, val


def _load_prototypes(proto_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    protos = torch.load(proto_path, map_location="cpu")
    if not isinstance(protos, list):
        raise ValueError(f"Expected list of prototypes in {proto_path}, got {type(protos)}")
    feats = []
    labels = []
    for entry in protos:
        p_class = entry.get("class")
        if isinstance(p_class, torch.Tensor):
            p_class = int(p_class.item())
        p_feats = entry.get("feats")
        if p_feats is None:
            continue
        if isinstance(p_feats, torch.Tensor):
            p_feats = p_feats.detach().cpu().numpy()
        feats.append(p_feats)
        labels.append(np.full((p_feats.shape[0],), int(p_class), dtype=np.int64))
    if not feats:
        return np.zeros((0, 0)), np.zeros((0,), dtype=np.int64)
    return np.vstack(feats), np.concatenate(labels)


def _knn_acc(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, k: int) -> float:
    if train_x.size == 0:
        return float("nan")
    n_neighbors = int(min(max(k, 1), train_x.shape[0]))
    train_x_t = torch.from_numpy(train_x).float()
    train_y_t = torch.from_numpy(train_y).long()
    val_x_t = torch.from_numpy(val_x).float()
    val_y_t = torch.from_numpy(val_y).long()

    dists = torch.cdist(val_x_t, train_x_t)  # [N_val, N_train]
    knn_idx = torch.topk(dists, k=n_neighbors, largest=False).indices
    knn_labels = train_y_t[knn_idx]  # [N_val, k]
    preds = torch.mode(knn_labels, dim=1).values
    return float((preds == val_y_t).float().mean().item())


def _defaults_from_section(section: Dict[str, object]) -> Dict[str, object]:
    cfg = {k: v for k, v in section.items() if k not in {"sweep", "conditional_sweep"}}
    sweep = section.get("sweep") or {}
    for k, v in sweep.items():
        if k in cfg:
            continue
        if isinstance(v, list):
            cfg[k] = v[0] if v else None
        else:
            cfg[k] = v
    return cfg


def _fingerprint(cfg: Dict[str, object], prefix: str) -> str:
    sub = {k: v for k, v in cfg.items() if isinstance(k, str) and k.startswith(prefix)}
    if not sub:
        return "none"
    blob = json.dumps(sub, sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:8]


def _default_ckpt_root(exp_family: str | None = None, dataset_partition: str | None = None) -> str:
    env_ckpt = os.environ.get("ckpt_root")
    if env_ckpt:
        return env_ckpt
    if exp_family is None:
        exp_family = os.environ.get("EXP_FAMILY", "eicu")
    if dataset_partition is None:
        dataset_partition = os.environ.get("DATASET_PARTITION")
    ckpt_partition = f"/{dataset_partition}" if dataset_partition else ""
    return f"/Shared/lss_brimueller/{exp_family}{ckpt_partition}"


def main() -> None:
    parser = argparse.ArgumentParser(description="KNN sanity check for prototype quality.")
    parser.add_argument("--ckpt_root", type=str, default=None)
    parser.add_argument("--exp_family", type=str, default=None)
    parser.add_argument("--data_partition", type=str, default=None)
    parser.add_argument("--configs", type=str, default=None)
    parser.add_argument("--role", type=str, default=None)
    parser.add_argument("--base_id", type=str, default=None)
    parser.add_argument("--bundle_path", type=str, default=None)
    parser.add_argument("--proto_path", type=str, default=None)
    parser.add_argument("--proto_min_samples", type=int, default=None)
    parser.add_argument("--proto_max_k", type=int, default=None)
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    repo_root = REPO_ROOT
    configs_path = Path(args.configs) if args.configs else repo_root / "system" / "conf" / "configs.yaml"
    cfg = yaml.safe_load(configs_path.read_text())

    base_cfg = _defaults_from_section(cfg.get("base_configs", {}))
    graph_cfg = _defaults_from_section(cfg.get("graph_configs", {}))
    pipeline_cfg = cfg.get("pipeline", {})

    base_id = args.base_id or _fingerprint(base_cfg, "base")
    proto_min_samples = args.proto_min_samples
    if proto_min_samples is None:
        proto_min_samples = int(graph_cfg.get("proto_min_samples", 5))
    proto_max_k = args.proto_max_k
    if proto_max_k is None:
        proto_max_k = int(graph_cfg.get("proto_max_k", 5))

    exp_family = args.exp_family or os.environ.get("EXP_FAMILY", "eicu")
    dataset_partition = args.data_partition or os.environ.get("DATASET_PARTITION")
    if not dataset_partition:
        common_args = pipeline_cfg.get("common_args", {})
        family_args = (pipeline_cfg.get("family_args", {}) or {}).get(exp_family, {}) or {}
        data_pipeline = pipeline_cfg.get("data_partition_args", {}) or {}
        family_data = family_args.get("data_partition_args", {}) or {}
        num_clients = family_args.get("num_clients", common_args.get("num_clients", 0))
        builder = family_args.get("partition_builder")
        if builder == "eicu":
            min_size = (
                family_data.get("min_size")
                or family_data.get("min_samples")
                or data_pipeline.get("min_size")
                or data_pipeline.get("min_samples")
                or 0
            )
            dataset_partition = build_eicu_partition_id(
                task=family_data.get("task", exp_family),
                min_size=min_size,
                seed=family_data.get("seed", data_pipeline.get("seed", 0)),
                train_ratio=family_data.get("train_ratio", data_pipeline.get("train_ratio", 0.75)),
                num_clients=num_clients,
            )
        else:
            merged_data = {**data_pipeline, **family_data}
            dataset_partition = build_dataset_partition_id(
                merged_data.get("partition", "dir"),
                merged_data.get("alpha", 1.0),
                merged_data.get("C", 5),
                merged_data.get("min_size"),
                merged_data.get("train_ratio", 0.75),
                merged_data.get("seed", 0),
                num_clients,
            )

    ckpt_root = args.ckpt_root or _default_ckpt_root(exp_family, dataset_partition)
    base_root = Path(ckpt_root) / "base_clf"
    if args.base_id is None:
        candidates = sorted(base_root.glob("base[*]"))
        candidates = [p for p in candidates if p.is_dir()]
        if len(candidates) == 1:
            base_id = candidates[0].name.replace("base[", "").replace("]", "")
        elif len(candidates) > 1:
            ids = [p.name.replace("base[", "").replace("]", "") for p in candidates]
            raise FileNotFoundError(
                "Multiple base IDs found under {root}. Provide --base_id. "
                "Found: {ids}".format(root=base_root, ids=", ".join(ids))
            )

    base_dir = base_root / f"base[{base_id}]"
    proto_id = _proto_config_id(proto_min_samples, proto_max_k)

    if args.bundle_path:
        bundle_paths = [Path(args.bundle_path)]
    else:
        if args.role:
            bundle_paths = [base_dir / f"{args.role}_graph_bundle.pt"]
        else:
            bundle_paths = sorted(base_dir.glob("*_graph_bundle.pt"))

    if not bundle_paths:
        search_root = Path(ckpt_root)
        found = list(search_root.glob("**/base_clf/base[*]/*_graph_bundle.pt"))
        if found:
            base_ids = sorted(
                {
                    p.parent.name.replace("base[", "").replace("]", "")
                    for p in found
                }
            )
            if len(base_ids) == 1:
                base_id = base_ids[0]
                base_dir = search_root / "base_clf" / f"base[{base_id}]"
                bundle_paths = sorted(base_dir.glob("*_graph_bundle.pt"))
            else:
                raise FileNotFoundError(
                    "Multiple base IDs with graph bundles found under {root}. "
                    "Provide --base_id. Found: {ids}".format(
                        root=search_root, ids=", ".join(base_ids)
                    )
                )
        else:
            raise FileNotFoundError(f"No graph bundle files found under {base_dir}")

    total_clients = 0
    proto_available = 0
    proto_wins = 0
    proto_ties = 0

    for bundle_path in bundle_paths:
        if not bundle_path.exists():
            print(f"Missing graph bundle: {bundle_path}")
            continue
        role = bundle_path.name.replace("_graph_bundle.pt", "")
        total_clients += 1

        if args.proto_path:
            proto_path = Path(args.proto_path)
        else:
            proto_path = base_dir / f"{role}_prototypes[{proto_id}].pt"

        tr, val = _load_bundle(bundle_path)
        tr_x = tr["feats"].detach().cpu().numpy()
        tr_y = tr["y"].detach().cpu().numpy()
        val_x = val["feats"].detach().cpu().numpy()
        val_y = val["y"].detach().cpu().numpy()

        train_acc = _knn_acc(tr_x, tr_y, val_x, val_y, args.k)
        print(f"[{role}] KNN (train nodes -> val) acc={train_acc:.4f} (k={args.k})")

        if proto_path.exists():
            proto_x, proto_y = _load_prototypes(proto_path)
            proto_acc = _knn_acc(proto_x, proto_y, val_x, val_y, args.k)
            print(f"[{role}] KNN (prototype nodes -> val) acc={proto_acc:.4f} (k={args.k})")
            proto_available += 1
            if proto_acc > train_acc:
                proto_wins += 1
            elif proto_acc == train_acc:
                proto_ties += 1
        else:
            print(f"[{role}] Prototype file not found: {proto_path}")

    if proto_available > 0:
        win_pct = 100.0 * (proto_wins / proto_available)
        print(
            "Prototype wins: {wins}/{avail} ({pct:.2f}%), ties={ties}, total_clients={total}".format(
                wins=proto_wins,
                avail=proto_available,
                pct=win_pct,
                ties=proto_ties,
                total=total_clients,
            )
        )
    else:
        print("Prototype wins: 0/0 (0.00%), ties=0, total_clients={total}".format(total=total_clients))


if __name__ == "__main__":
    main()

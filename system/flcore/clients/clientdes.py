from __future__ import annotations
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import time
import wandb
import torch.nn as nn
import matplotlib
from torch_geometric.loader import NeighborLoader
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from entmax import entmax_bisect, sparsemax
from sklearn.metrics import precision_score, f1_score
from torch_geometric.utils import degree

# ---------- DES utilities ----------
from des.dataset_stats import load_client_label_counts
from des.graph_utils import build_train_eval_graph, compute_meta_labels, project_to_DS, generate_prototypes
from des.base_clf_utils import fit_clf
from des.helpers import derive_config_ids, init_base_meta_loaders, get_kfold_loaders, get_performance_baselines
from des.meta_learner_utils import (
    build_meta_learner,
    compute_sample_weights,
    enforce_bidirectionality,
    compute_gnn_diversity_loss,
    drop_cc_edges,
    _neighborloader_supports_weight_attr,
    iter_weighted_ss_minibatches,
)
from flcore.clients.clientbase import Client


def _plot_phase3_line_chart(
    epochs: List[int],
    values: List[float],
    title: str,
    ylabel: str,
    out_path: Path,
    secondary: Optional[tuple[List[int], List[float]]] = None,
    secondary_label: str = "fallback",
) -> None:
    if not epochs or not values:
        return
    plt.figure()
    plt.plot(epochs, values, marker="o", linewidth=2.0, label="main")
    if secondary is not None:
        sec_epochs, sec_values = secondary
        if sec_epochs and sec_values:
            plt.plot(sec_epochs, sec_values, marker="o", linewidth=2.0, label=secondary_label)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    if secondary is not None:
        plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_phase3_scatter(
    xs: List[float],
    ys: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    colors: Optional[List[str]] = None,
) -> None:
    if not xs or not ys or len(xs) != len(ys):
        return
    if colors is not None and len(colors) != len(xs):
        colors = None
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, s=30, alpha=0.8, c=colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

class clientDES(Client):
    """
    FedDES client.
    Responsibilities:
      1. Train/Load base classifiers & OOF data.
      2. Build graph data bundles and PyG graphs.
      3. Train the Meta-Learner (GNN).
    """

    def __init__(self, args, id: int, train_samples: int, test_samples: int, **kwargs: Any) -> None:
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.args = args
        self.device = args.device
        total_models = len(args.models)
        if total_models == 0:
            raise ValueError("FedDES expects at least one model in args.models.")
        self.base_single_model = getattr(args, "base_single_model", False)
        if self.base_single_model:
            model_ids = [self.id % total_models]
        else:
            model_ids = list(range(total_models))
        self.model_ids = model_ids
        self.model_strs = [f"model_{model_id}" for model_id in model_ids]
        self.num_models = len(model_ids)

        self.base_config_id, self.graph_config_id, self.gnn_config_id = derive_config_ids(self.args)
        self.base_dir = args.ckpt_root / "base_clf" /  f"base[{self.base_config_id}]"
        self.graph_dir = args.ckpt_root / "graphs" / f"base[{self.base_config_id}]_graph[{self.graph_config_id}]"
        self.gnn_dir = args.ckpt_root / "gnn" / f"base[{self.base_config_id}]_graph[{self.graph_config_id}]_gnn[{self.gnn_config_id}]"
        self.base_outputs_dir = args.outputs_root / "base_clf" /  f"base[{self.base_config_id}]"
        self.graph_outputs_dir = args.outputs_root / "graphs" / f"base[{self.base_config_id}]_graph[{self.graph_config_id}]"

        for dir in [self.base_dir, self.graph_dir, self.gnn_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        for dir in [self.base_outputs_dir, self.graph_outputs_dir]:
            dir.mkdir(parents=True, exist_ok=True)

        self._base_train_loader = None
        self._meta_train_loader = None
        self.meta_history: List[Dict[str, Any]] = []

    def base_classifiers_exist(self) -> bool:
        if self.base_single_model:
            self._sync_single_model_from_multi_base()
        expected = [
            self.base_dir / f"{self.role}_{model_str}.pt"
            for model_str in self.model_strs
        ]
        statuses = {str(p): p.exists() for p in expected}
        return all(statuses.values()) if expected else True

    def _sync_single_model_from_multi_base(self) -> None:
        args_dict = dict(vars(self.args))
        args_dict["base_single_model"] = False
        multi_base_id, _, _ = derive_config_ids(args_dict)
        multi_base_dir = self.args.ckpt_root / "base_clf" / f"base[{multi_base_id}]"
        if not multi_base_dir.exists():
            return

        for model_str in self.model_strs:
            src = multi_base_dir / f"{self.role}_{model_str}.pt"
            dst = self.base_dir / f"{self.role}_{model_str}.pt"
            if dst.exists() or not src.exists():
                continue
            try:
                shutil.copy2(src, dst)
                print(
                    f"[FedDES][Client {self.role}] Copied base model {model_str} "
                    f"from base[{multi_base_id}] to base[{self.base_config_id}]"
                )
            except Exception as exc:
                print(
                    f"[FedDES][Client {self.role}] Failed to copy base model {model_str} "
                    f"from base[{multi_base_id}]: {exc}"
                )
    
    def graph_bundle_exists(self) -> bool:
        bundle_path = Path(self.base_dir) / f"{self.role}_graph_bundle.pt"
        return bundle_path.exists()
    
    def graphs_exist(self) -> bool:
        train_val_path = Path(self.graph_dir) / f"{self.role}_graph_train_val.pt"
        train_test_path = Path(self.graph_dir) / f"{self.role}_graph_train_test.pt"
        return all(p.exists() for p in (train_val_path, train_test_path))
    
    def missing_prototypes(self) -> bool:
        if not bool(getattr(self.args, "proto_use", False)):
            return False
        proto_path = self._proto_path()
        return not proto_path.exists()

    def _proto_config(self) -> Dict[str, int]:
        return {
            "proto_min_samples": int(getattr(self.args, "proto_min_samples", 5)),
            "proto_max_k": int(getattr(self.args, "proto_max_k", 5)),
        }

    def _proto_config_id(self) -> str:
        payload = json.dumps(self._proto_config(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]

    def _proto_path(self) -> Path:
        proto_id = self._proto_config_id()
        return Path(self.base_dir) / f"{self.role}_prototypes[{proto_id}].pt"

    def train_base_classifiers(self, device=None):
        device = torch.device(device if device is not None else self.device)
        split_mode = str(getattr(self.args, "base_split_mode", "split_train")).lower()
        if split_mode == "oof_stacking":
            self._train_oof_stacking(device)
            return

        base_train_loader, _ = init_base_meta_loaders(self)
        val_loader = self.load_val_data()

        for model_id, model_str in zip(self.model_ids, self.model_strs):
            best_epoch, best_score, model = fit_clf(
                self,
                model_id, base_train_loader, val_loader, device,
                max_epochs=self.args.local_epochs,
                patience=self.args.base_es_patience,
                es_metric=self.args.base_es_metric,
                lr=self.args.base_clf_lr,
                min_delta=self.args.base_es_min_delta,
            )
            torch.save(model.cpu(), self.base_dir / f"{self.role}_{model_str}.pt")
            print(f"{self.role} {model_id} stopping training at epoch {best_epoch}, score = {best_score}")

    def _train_oof_stacking(self, device):
        n_folds = int(getattr(self.args, "base_oof_folds", 5))
        seed = int(getattr(self.args, "base_split_seed", 0))
        folds = get_kfold_loaders(self, n_splits=n_folds, seed=seed)

        full_train_loader = self.load_train_data(batch_size=self.batch_size)
        n_total = len(full_train_loader.dataset)
        num_classes = int(getattr(self.args, "num_classes", 1))

        local_keys = [(self.role, model_str) for model_str in self.model_strs]
        global_keys = list(self.global_clf_keys)
        local_indices = [global_keys.index(k) for k in local_keys if k in global_keys]
        if len(local_indices) != len(local_keys):
            print(
                f"[FedDES][Client {self.role}][warn] local classifier keys not found in global keys "
                f"({len(local_indices)}/{len(local_keys)})."
            )

        oof_probs = torch.zeros((n_total, len(local_keys), num_classes), dtype=torch.float32)
        oof_preds = torch.zeros((n_total, len(local_keys)), dtype=torch.long)
        oof_labels = torch.full((n_total,), -1, dtype=torch.long)

        for fold_idx, (train_loader, val_loader, val_idx) in enumerate(folds, start=1):
            print(f"[FedDES][Client {self.role}] OOF fold {fold_idx}/{n_folds}")

            fold_pool = {}
            for model_id, model_str in zip(self.model_ids, self.model_strs):
                _, _, model = fit_clf(
                    self,
                    model_id,
                    train_loader,
                    val_loader,
                    device,
                    max_epochs=self.args.local_epochs,
                    patience=self.args.base_es_patience,
                    es_metric=self.args.base_es_metric,
                    lr=self.args.base_clf_lr,
                    min_delta=self.args.base_es_min_delta,
                    log_wandb=False,
                )
                fold_pool[(self.role, model_str)] = model

            prev_keys = self.global_clf_keys
            self.global_clf_keys = local_keys
            try:
                ds, preds, y_true, meta_labels, feats = project_to_DS(
                    self, val_loader, fold_pool, calibrate_probs=True
                )
            finally:
                self.global_clf_keys = prev_keys

            probs = ds.view(ds.size(0), len(local_keys), num_classes)
            val_idx = np.asarray(val_idx, dtype=np.int64)
            if probs.size(0) != val_idx.shape[0]:
                raise RuntimeError(
                    f"OOF fold size mismatch: probs={probs.size(0)} val_idx={val_idx.shape[0]}"
                )

            oof_probs[val_idx] = probs.detach().cpu()
            oof_preds[val_idx] = preds.detach().cpu()
            oof_labels[val_idx] = y_true.detach().cpu()

        if bool((oof_labels < 0).any().item()):
            missing = int((oof_labels < 0).sum().item())
            raise RuntimeError(
                f"OOF stacking failed to populate {missing} labels. Check fold indices."
            )

        oof_meta = compute_meta_labels(
            oof_probs,
            oof_preds,
            oof_labels,
            min_positive=int(getattr(self.args, "graph_meta_min_pos", 3)),
        )

        oof_cache = {
            "probs": oof_probs.cpu(),
            "preds": oof_preds.cpu(),
            "meta": oof_meta.cpu(),
            "y": oof_labels.cpu(),
            "local_clf_keys": local_keys,
            "local_clf_indices": local_indices,
        }
        oof_path = self.base_dir / f"{self.role}_oof_cache.pt"
        torch.save(oof_cache, oof_path)
        print(f"[FedDES][Client {self.role}] OOF cache saved to {oof_path}")

        print(f"[FedDES][Client {self.role}] Retraining final models on 100% data...")
        real_val_loader = self.load_val_data()
        for model_id, model_str in zip(self.model_ids, self.model_strs):
            _, _, model = fit_clf(
                self,
                model_id,
                full_train_loader,
                real_val_loader,
                device,
                max_epochs=self.args.local_epochs,
                patience=self.args.base_es_patience,
                es_metric=self.args.base_es_metric,
                lr=self.args.base_clf_lr,
                min_delta=self.args.base_es_min_delta,
                log_wandb=True,
            )
            torch.save(model.cpu(), self.base_dir / f"{self.role}_{model_str}.pt")
        
    def prepare_graph_data(self, device=None, classifier_pool: Dict[Any, torch.nn.Module] = None) -> None:
            """
            Prepare decision-space and meta-data for graph building.
            """
            device = torch.device(device if device is not None else self.device)
            self.device = device

            _, meta_train_loader = init_base_meta_loaders(self)
            val_loader = self.load_val_data()
            test_loader = self.load_test_data()

            data_loaders = {"train": meta_train_loader, "val": val_loader, "test": test_loader}

            cache_path = (Path(self.base_dir)/ f"{self.role}_graph_bundle.pt")
            oof_cache_path = self.base_dir / f"{self.role}_oof_cache.pt"
            split_mode = str(getattr(self.args, "base_split_mode", "split_train")).lower()
            use_oof = split_mode == "oof_stacking" and oof_cache_path.exists()
            
            generate_protos = getattr(self.args, "proto_use", False)
            proto_path = self._proto_path()

            if cache_path.exists():
                print(f"[FedDES][Client {self.role}] Loading cached graph artifacts from {cache_path}")
                graph_data = torch.load(cache_path, map_location="cpu")
                
                if generate_protos and not proto_path.exists():
                    print(f"[FedDES][Client {self.role}] Prototypes requested but missing on disk. Generating...")
                    tr = graph_data["train"]
                    prototypes = generate_prototypes(
                        self,
                        ds=tr["ds"], 
                        feats=tr["feats"], 
                        meta=tr["meta"], 
                        y=tr["y"],
                    )
                    proto_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(prototypes, proto_path)

            else:
                graph_data = {}
                splits_to_run = ["train", "val", "test"]
                if use_oof:
                    splits_to_run = ["val", "test"]
                    print(f"[FedDES][Client {self.role}] Using OOF cache for train split.")

                for data_split in splits_to_run:
                    loader = data_loaders[data_split]
                    ds, preds, y_true, meta_labels, feats = project_to_DS(
                        self, loader, classifier_pool
                    )
                    graph_data[data_split] = {
                        "ds": ds, "preds": preds, "y": y_true, "meta": meta_labels.float(), "feats": feats
                    }

                if use_oof:
                    oof_cache = torch.load(oof_cache_path, map_location="cpu")
                    full_train_loader = self.load_train_data(batch_size=self.batch_size)
                    train_loader = torch.utils.data.DataLoader(
                        full_train_loader.dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=0,
                    )
                    ds, preds, y_true, meta_labels, feats = project_to_DS(
                        self, train_loader, classifier_pool
                    )
                    graph_data["train"] = {
                        "ds": ds, "preds": preds, "y": y_true, "meta": meta_labels.float(), "feats": feats
                    }
                    self._apply_oof_to_train_graph_data(graph_data["train"], oof_cache)
                
                if generate_protos:
                    print(f"[FedDES][Client {self.role}] Generating prototypes from training data...")
                    tr = graph_data["train"]
                    prototypes = generate_prototypes(
                        self,
                        ds=tr["ds"], 
                        feats=tr["feats"], 
                        meta=tr["meta"], 
                        y=tr["y"],
                    )
                    proto_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(prototypes, proto_path)

                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(graph_data, cache_path)
                
    def build_graph(self, device=None, classifier_pool: Dict[Any, torch.nn.Module] = None) -> None:
        """
        Build FedDES graphs for this client.
        """
        cache_path = (Path(self.base_dir)/ f"{self.role}_graph_bundle.pt")

        print(f"[FedDES][Client {self.role}] Loading cached graph artifacts from {cache_path}")
        graph_data = torch.load(cache_path, map_location="cpu")

        tr, val, test = (SimpleNamespace(**graph_data[data_split]) for data_split in ["train", "val", "test"])
        
        peer_prototypes = []
        use_prototypes = getattr(self.args, "proto_use", False)
        
        if use_prototypes:
            proto_id = self._proto_config_id()
            all_proto_files = list(Path(self.base_dir).glob(f"*_prototypes[{proto_id}].pt"))
            print(f"[FedDES][Client {self.role}] Loading prototypes from {len(all_proto_files)} peer files...")
            for proto_path in all_proto_files:
                try:
                    peer_data = torch.load(proto_path, map_location="cpu")
                    if isinstance(peer_data, list):
                        peer_prototypes.extend(peer_data)
                except Exception as e:
                    print(f"Warning: Failed to load prototypes {proto_path}: {e}")

        train_val_graph = build_train_eval_graph(self,
            tr.ds,   tr.preds,   tr.meta,   tr.y,   tr.feats,
            val.ds,  val.y,  val.feats, eval_type="val", prototypes=peer_prototypes
        )

        train_test_graph = build_train_eval_graph(self,
            tr.ds,   tr.preds,   tr.meta,   tr.y,   tr.feats,
            test.ds, test.y, test.feats, eval_type="test", prototypes=peer_prototypes
        )

        if bool(getattr(self.args, "graph_drop_disconnected_cls", False)):
            self._drop_disconnected_classifiers(train_val_graph, train_test_graph, (tr, val, test))
            graph_data["train"]["meta"] = tr.meta
            graph_data["train"]["preds"] = tr.preds
            graph_data["train"]["ds"] = tr.ds
            graph_data["val"]["meta"] = val.meta
            graph_data["val"]["preds"] = val.preds
            graph_data["val"]["ds"] = val.ds
            graph_data["test"]["meta"] = test.meta
            graph_data["test"]["preds"] = test.preds
            graph_data["test"]["ds"] = test.ds
            torch.save(graph_data, cache_path)

        get_performance_baselines(self, graph_data["test"])
        torch.save(train_val_graph,  self.graph_dir / f"{self.role}_graph_train_val.pt")
        torch.save(train_test_graph, self.graph_dir / f"{self.role}_graph_train_test.pt")

    def train_meta_learner(self, device=None) -> None:
        """
        Train a meta-learner GNN on the pre-built graphs.
        """
        device = torch.device(device if device is not None else self.device)
        self.meta_history = []

        ensemble_criterion = torch.nn.CrossEntropyLoss()

        graph_data = torch.load(
            self.base_dir / f"{self.role}_graph_bundle.pt",
            map_location="cpu",
        )

        for data_split in ["train", "val", "test"]:
            graph_data[data_split] = {
                k: v.to(device) for k, v in graph_data[data_split].items()
            }

        tr, val, test = (
            SimpleNamespace(**graph_data[data_split])
            for data_split in ["train", "val", "test"]
        )

        train_val_graph = torch.load(
            self.graph_dir / f"{self.role}_graph_train_val.pt",
            map_location=device,
        )
        train_test_graph = torch.load(
            self.graph_dir / f"{self.role}_graph_train_test.pt",
            map_location=device,
        )

        self._maybe_normalize_sample_feats(train_val_graph)
        self._maybe_normalize_sample_feats(train_test_graph)

        bidir = self.args.gnn_bidirectionality
        enforce_bidirectionality(train_val_graph, bidir)
        enforce_bidirectionality(train_test_graph, bidir)

        if self.args.gnn_drop_cc_edges:
            drop_cc_edges(train_val_graph)
            drop_cc_edges(train_test_graph)

        self._active_clf_keys = None
        if bool(getattr(self.args, "graph_drop_disconnected_cls", False)):
            self._drop_disconnected_classifiers(train_val_graph, train_test_graph, (tr, val, test))

        num_models = int(tr.meta.size(1))
        metadata = train_val_graph.metadata()
        input_dims = {
            ntype: train_val_graph[ntype].x.size(-1)
            for ntype in metadata[0]
        }
        gnn_model = build_meta_learner(
            self.args,
            metadata,
            input_dims,
            num_models,
        ).to(device)

        epochs = int(getattr(self.args, "gnn_epochs", 300))
        patience = int(getattr(self.args, "gnn_patience", 20))
        es_metric = getattr(self.args, "gnn_es_metric", "val_loss")
        lr, wd = self.args.gnn_lr, self.args.gnn_weight_decay
        optimizer = torch.optim.Adam(
            gnn_model.parameters(),
            lr=lr,
            weight_decay=wd,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if es_metric == "val_loss" else "max",
            factor=0.5,
            patience=max(1, int(patience / 2)),
            verbose=True,
        )

        sample_weights = compute_sample_weights(
            self,
            tr.y,
            tr.meta,
            self.args.gnn_sample_weight_mode,
        )
        if sample_weights is not None:
            with torch.no_grad():
                sw = sample_weights.detach().float()
                sw_mean = sw.mean().clamp(min=1e-8)
                sample_weights = (sw / sw_mean).to(device=device)

        gnn_loss_mode = str(getattr(self.args, "gnn_loss", "meta_labels_BCE")).lower()
        if gnn_loss_mode in {"meta_labels", "meta_labels_bce"}:
            gnn_loss_mode = "meta_labels_bce"
        elif gnn_loss_mode in {"meta_labels_regression"}:
            gnn_loss_mode = "meta_labels_regression"
        elif gnn_loss_mode in {"meta_labels_soft_bce"}:
            gnn_loss_mode = "meta_labels_soft_bce"

        use_pos_weight = bool(getattr(self.args, "gnn_use_pos_weight", False))
        pos_weight = None
        if gnn_loss_mode in {"meta_labels_bce", "meta_labels_soft_bce"} and use_pos_weight:
            with torch.no_grad():
                pos = tr.meta.float().sum(dim=0)
                neg = tr.meta.size(0) - pos
                pos_weight = (neg / pos.clamp(min=1.0)).to(device=device, dtype=torch.float)
                pw_max = float(getattr(self.args, "gnn_pos_weight_max", 10.0))
                if pw_max is not None and pw_max > 0:
                    pos_weight = pos_weight.clamp(max=pw_max)
                try:
                    pw_q = {
                        "min": float(pos_weight.min().item()),
                        "q25": float(torch.quantile(pos_weight, torch.tensor(0.25, device=pos_weight.device)).item()),
                        "med": float(torch.quantile(pos_weight, torch.tensor(0.50, device=pos_weight.device)).item()),
                        "q75": float(torch.quantile(pos_weight, torch.tensor(0.75, device=pos_weight.device)).item()),
                        "max": float(pos_weight.max().item()),
                    }
                    print(
                        f"[FedDES][Client {self.role}][diag] BCE pos_weight stats "
                        f"min={pw_q['min']:.3f} q25={pw_q['q25']:.3f} med={pw_q['med']:.3f} q75={pw_q['q75']:.3f} max={pw_q['max']:.3f}"
                    )
                except Exception:
                    pass

        if gnn_loss_mode == "meta_labels_regression":
            criterion_none = torch.nn.MSELoss(reduction="none")
        else:
            if pos_weight is None:
                criterion_none = torch.nn.BCEWithLogitsLoss(reduction="none")
            else:
                criterion_none = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        train_mask = train_val_graph["sample"].train_mask
        val_mask = train_val_graph["sample"].val_mask
        test_mask = train_test_graph["sample"].test_mask
        if bool(getattr(self.args, "gnn_debug_degrees", False)):
            self._print_sample_degrees(train_val_graph, "train_val")
            self._print_sample_degrees(train_test_graph, "train_test")
        try:
            train_mask_tv = train_mask.cpu()
            val_mask_tv = val_mask.cpu()
            train_mask_tt = train_test_graph["sample"].train_mask.cpu()
            test_mask_tt = test_mask.cpu()
            print(
                "[FedDES][Client {role}][diag] masks: "
                "train={tr} val={va} test={te} "
                "overlap(train,val)={tv} overlap(train,test)={tt}".format(
                    role=self.role,
                    tr=int(train_mask_tv.sum().item()),
                    va=int(val_mask_tv.sum().item()),
                    te=int(test_mask_tt.sum().item()),
                    tv=int((train_mask_tv & val_mask_tv).sum().item()),
                    tt=int((train_mask_tt & test_mask_tt).sum().item()),
                )
            )
        except Exception as e:
            print(f"[FedDES][Client {self.role}][diag] mask overlap check failed: {e}")
        
        # --- Handle Legacy Naming and Set Combination Mode ---
        combination_mode = str(
            getattr(self.args, "gnn_ens_combination_mode", "soft")
        ).lower()
        legacy_modes = {
            "voting": "hard_voting",
            "weighted_vote": "soft_voting",
            "weighted_voting": "soft_voting",
        }
        combination_mode = legacy_modes.get(combination_mode, combination_mode)
        
        alpha = 1.5
        arch_name = str(getattr(self.args, "gnn_arch", "hetero_gat")).lower()
        sampler_mode = str(getattr(self.args, "gnn_sampler", "")).lower()
        if sampler_mode not in {"none", "unweighted", "weighted"}:
            print(f"[FedDES][Client {self.role}][warn] Unknown gnn_sampler={sampler_mode}; defaulting to none.")
            sampler_mode = "none"
        use_sampler = sampler_mode != "none"

        homo_archs = {"gat", "gatv3", "homog_gat", "homogeneous_gat", "sage", "graphsage", "sage_attn", "sage+attn"}
        use_homo_graph = arch_name in homo_archs
        use_homo_sage_sampling = arch_name in {"sage", "graphsage", "sage_attn", "sage+attn"}
        use_hetero_sage_sampling = arch_name in {"hetero_sage_attn", "hetero_sage", "sage_attn_hetero", "sage_hetero"}
        if (use_homo_sage_sampling or use_hetero_sage_sampling) and not use_sampler:
            use_sampler = True
            sampler_mode = "unweighted"
            print(f"[FedDES][Client {self.role}][diag] gnn_sampler forced to unweighted for SAGE-based arch={arch_name}")

        use_neighbor_sampling = use_sampler
        if use_neighbor_sampling and (not use_homo_graph):
            pair_decoder = getattr(self.args, "gnn_pair_decoder", None)
            if pair_decoder is not None and str(pair_decoder).lower() not in {"", "none"}:
                raise ValueError("NeighborLoader is incompatible with pairwise decoders. Disable gnn_sampler or set gnn_pair_decoder='none'.")

        train_nodes = train_mask.nonzero(as_tuple=False).view(-1)
        num_total_samples = int(train_val_graph["sample"].num_nodes)
        train_global_to_local = torch.full((num_total_samples,), -1, device=device, dtype=torch.long)
        train_global_to_local[train_nodes] = torch.arange(train_nodes.numel(), device=device, dtype=torch.long)

        sage_train_loader = None
        sage_train_loader_fn = None

        if use_neighbor_sampling:
            num_layers = int(getattr(self.args, "gnn_layers", 2))
            fanout = int(getattr(self.args, "gnn_sage_fanout", 5))
            sage_num_neighbors = getattr(self.args, "gnn_sage_num_neighbors", None)
            if sage_num_neighbors is None:
                sage_num_neighbors = [fanout] * num_layers
            elif isinstance(sage_num_neighbors, int):
                sage_num_neighbors = [sage_num_neighbors] * num_layers

            sage_batch_size = int(getattr(self.args, "gnn_batch_size", 64))
            sage_shuffle = bool(getattr(self.args, "gnn_sage_shuffle", True))
            if sampler_mode == "weighted": sage_weighted = True
            elif sampler_mode == "unweighted": sage_weighted = False
            else: sage_weighted = bool(getattr(self.args, "gnn_sage_weighted_sampling", True))

            if use_homo_graph:
                from des.meta_learner_utils import drop_classifier_nodes
                sage_graph = drop_classifier_nodes(train_val_graph)
                edge = sage_graph[("sample", "ss", "sample")]
                row, col = edge.edge_index
                train_mask_cpu = train_mask.to(row.device)
                keep = train_mask_cpu[row] & train_mask_cpu[col]
                edge.edge_index = edge.edge_index[:, keep]
                if edge.edge_attr is not None: edge.edge_attr = edge.edge_attr[keep]
                sage_graph = sage_graph.cpu()

                if sage_weighted and _neighborloader_supports_weight_attr():
                    try:
                        sage_train_loader = NeighborLoader(
                            sage_graph,
                            num_neighbors={("sample", "ss", "sample"): [fanout] * num_layers},
                            input_nodes=("sample", train_nodes.cpu()),
                            batch_size=sage_batch_size,
                            shuffle=sage_shuffle,
                            weight_attr="edge_attr",
                            num_workers=0,
                            persistent_workers=False,
                        )
                        print("[diag] NeighborLoader(weight_attr='edge_attr') enabled for homo sampler.")
                    except Exception as e:
                        print(f"[WARN] NeighborLoader(weight_attr) failed for homo sampler: {e}. Falling back to manual sampler.")
                        sage_train_loader = None

                if sage_train_loader is None:
                    sage_train_loader_fn = lambda: iter_weighted_ss_minibatches(
                        sage_graph,
                        seed_nodes=train_nodes,
                        num_neighbors=sage_num_neighbors,
                        batch_size=sage_batch_size,
                        device=device,
                        shuffle=sage_shuffle,
                        allowed_nodes=train_nodes,
                    )
                    print("[diag] Manual weighted S-S sampling enabled for homo sampler.")
            else:
                try:
                    sage_graph = train_val_graph.to("cpu", copy=True)
                except TypeError:
                    sage_graph = train_val_graph.clone().cpu()

                def _sanitize_store(store, *, store_name: str):
                    for key in list(store.keys()):
                        if key in {"num_nodes"}: continue
                        val = store[key]
                        if torch.is_tensor(val): continue
                        if isinstance(val, list):
                            if len(val) == 0: del store[key]; continue
                            if all(isinstance(x, (bool, int)) for x in val): store[key] = torch.tensor(val, dtype=torch.long); continue
                            if all(isinstance(x, (bool, int, float)) for x in val): store[key] = torch.tensor(val, dtype=torch.float); continue
                            del store[key]; continue
                        del store[key]

                for ntype in sage_graph.node_types: _sanitize_store(sage_graph[ntype], store_name=f"{ntype}")
                for etype in sage_graph.edge_types: _sanitize_store(sage_graph[etype], store_name=f"{etype}")

                train_mask_cpu = train_mask.cpu()
                rel_ss = ("sample", "ss", "sample")
                if rel_ss in sage_graph.edge_index_dict:
                    edge = sage_graph[rel_ss]
                    row, col = edge.edge_index
                    keep = train_mask_cpu[row] & train_mask_cpu[col]
                    edge.edge_index = edge.edge_index[:, keep]
                    if getattr(edge, "edge_attr", None) is not None: edge.edge_attr = edge.edge_attr[keep]

                rel_cs = ("classifier", "cs", "sample")
                if rel_cs in sage_graph.edge_index_dict:
                    edge = sage_graph[rel_cs]
                    src, dst = edge.edge_index
                    keep = train_mask_cpu[dst]
                    edge.edge_index = edge.edge_index[:, keep]
                    if getattr(edge, "edge_attr", None) is not None: edge.edge_attr = edge.edge_attr[keep]

                rel_cs_rev = ("sample", "cs_rev", "classifier")
                if rel_cs_rev in sage_graph.edge_index_dict:
                    edge = sage_graph[rel_cs_rev]
                    src, dst = edge.edge_index
                    keep = train_mask_cpu[src]
                    edge.edge_index = edge.edge_index[:, keep]
                    if getattr(edge, "edge_attr", None) is not None: edge.edge_attr = edge.edge_attr[keep]

                update_clf = bool(getattr(self.args, "gnn_update_classifier_nodes", False))
                drop_cc = bool(getattr(self.args, "gnn_drop_cc_edges", False))
                rel_cc = ("classifier", "cc", "classifier")

                if drop_cc and (rel_cc in sage_graph.edge_index_dict):
                    try: del sage_graph[rel_cc]
                    except Exception as e: print(f"[WARN] could not delete {rel_cc}: {e}")

                num_layers = int(getattr(self.args, "gnn_layers", 2))
                fanout = int(getattr(self.args, "gnn_sage_fanout", -1))
                c_fanout = -1

                num_neighbors_map = {
                    ("sample", "ss", "sample"): [fanout] * num_layers,
                    ("classifier", "cs", "sample"): [c_fanout] * num_layers,
                }
                if (not drop_cc) and (rel_cc in sage_graph.edge_index_dict): num_neighbors_map[rel_cc] = [c_fanout] * num_layers
                if update_clf and (rel_cs_rev in sage_graph.edge_index_dict): num_neighbors_map[rel_cs_rev] = [c_fanout] * num_layers

                num_neighbors = {et: [0] * num_layers for et in sage_graph.edge_types}
                for et, fo in num_neighbors_map.items():
                    if et in num_neighbors: num_neighbors[et] = fo

                sage_graph = sage_graph.cpu()
                active_rels = [et for et, fo in num_neighbors.items() if any(v != 0 for v in fo)]
                
                use_weight_attr = False
                if sage_weighted and _neighborloader_supports_weight_attr():
                    ok = True
                    for et in active_rels:
                        if getattr(sage_graph[et], "edge_attr", None) is None: ok = False; break
                    if ok: use_weight_attr = True
                    else: print("[diag] Disabling weight_attr for hetero sampler: missing edge_attr on active relations.")

                try:
                    kwargs = dict(
                        data=sage_graph,
                        num_neighbors=num_neighbors,
                        input_nodes=("sample", train_nodes.cpu()),
                        batch_size=sage_batch_size,
                        shuffle=sage_shuffle,
                        num_workers=0,
                        persistent_workers=False,
                    )
                    if use_weight_attr: kwargs["weight_attr"] = "edge_attr"
                    sage_train_loader = NeighborLoader(**kwargs)
                    print(f"[diag] NeighborLoader enabled for hetero sampler with relations={active_rels}")
                except Exception as e:
                    raise RuntimeError(f"NeighborLoader failed for hetero sampler: {e}.")

        # --- Helper for Calculating Margins ---
        def _margin_targets(ds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            num_classes = int(self.args.num_classes)
            if ds.dim() != 2 or ds.size(1) % max(num_classes, 1) != 0:
                raise ValueError("margin target expects ds with shape [N, M*C].")
            M = int(ds.size(1) // max(num_classes, 1))
            probs = ds.view(ds.size(0), M, num_classes).float()
            y_idx = y.view(-1, 1, 1).expand(-1, M, 1)
            p_true = probs.gather(2, y_idx).squeeze(-1)
            probs_other = probs.clone()
            probs_other.scatter_(2, y_idx, -1.0)
            p_other = probs_other.max(dim=2).values
            return p_true - p_other

        # Helper: Ranking Loss
        def within_sample_pairwise_rank_loss(logits: torch.Tensor, meta: torch.Tensor, margin: float = 0.0, max_pairs: int | None = None, sample_weights: torch.Tensor | None = None) -> torch.Tensor:
            assert logits.ndim == 2 and meta.shape == logits.shape
            meta_bool = meta.bool()
            N, _ = logits.shape
            losses = []
            weights = []
            for i in range(N):
                pos_i = logits[i][meta_bool[i]]
                neg_i = logits[i][~meta_bool[i]]
                if pos_i.numel() == 0 or neg_i.numel() == 0: continue
                if (max_pairs is not None) and (max_pairs > 0):
                    P, Q = pos_i.numel(), neg_i.numel()
                    pos_idx = torch.randint(P, (max_pairs,), device=logits.device)
                    neg_idx = torch.randint(Q, (max_pairs,), device=logits.device)
                    diffs = pos_i[pos_idx] - neg_i[neg_idx]
                    loss_i = F.softplus(margin - diffs).mean()
                else:
                    diffs = pos_i[:, None] - neg_i[None, :]
                    loss_i = F.softplus(margin - diffs).mean()
                losses.append(loss_i)
                if sample_weights is not None: weights.append(sample_weights[i])
            if len(losses) == 0: return logits.new_tensor(0.0)
            losses_t = torch.stack(losses)
            if sample_weights is None: return losses_t.mean()
            weights_t = torch.stack(weights).to(losses_t)
            denom = weights_t.sum().clamp(min=1e-8)
            return (losses_t * weights_t).sum() / denom

        def _fmt5(x: torch.Tensor) -> str:
            if x is None or x.numel() == 0: return "[]"
            x = x.detach().flatten()
            k = min(5, x.numel())
            vals = ", ".join([f"{v:.4f}" for v in x[:k].tolist()])
            return f"[{vals}{', ...' if x.numel() > k else ''}]"

        def _quantiles(x: torch.Tensor, qs=(0.0, 0.25, 0.5, 0.75, 1.0)):
            if x is None or x.numel() == 0: return {q: float('nan') for q in qs}
            x = x.detach().float().flatten()
            out = {}
            for q in qs: out[q] = float(torch.quantile(x, torch.tensor(q, device=x.device)).item())
            return out

        def _ess_stats(weights: torch.Tensor) -> Dict[str, float]:
            if weights is None or weights.numel() == 0: return {"mean": 0.0, "q25": 0.0, "med": 0.0, "q75": 0.0}
            w = weights.detach().float()
            row_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
            w = w / row_sum
            ess = 1.0 / w.pow(2).sum(dim=1).clamp(min=1e-8)
            q = _quantiles(ess, qs=(0.25, 0.5, 0.75))
            mean_val = float(ess.mean().item()) if ess.numel() else 0.0
            return {
                "mean": mean_val,
                "q25": q[0.25] if not (q[0.25] != q[0.25]) else 0.0,
                "med": q[0.5] if not (q[0.5] != q[0.5]) else 0.0,
                "q75": q[0.75] if not (q[0.75] != q[0.75]) else 0.0,
            }

        def _ess_stats_or_none(weights: torch.Tensor | None) -> Optional[Dict[str, float]]:
            if weights is None or weights.numel() == 0: return None
            return _ess_stats(weights)

        def _entmax_stats(logits_: torch.Tensor):
            with torch.no_grad():
                w = entmax_bisect(logits_, alpha=alpha, dim=-1)
                nnz = (w > 0).sum(dim=1).float()
                maxw = w.max(dim=1).values
                stdw = w.std(dim=1)
                top1 = w.argmax(dim=1)
                top2 = torch.topk(logits_, k=min(2, logits_.size(1)), dim=1).values
                margin = (top2[:, 0] - top2[:, 1]) if top2.size(1) > 1 else top2[:, 0]
                return {
                    "w": w,
                    "nnz_mean": float(nnz.mean().item()) if nnz.numel() else 0.0,
                    "nnz_q": _quantiles(nnz),
                    "maxw_mean": float(maxw.mean().item()) if maxw.numel() else 0.0,
                    "maxw_q": _quantiles(maxw),
                    "stdw_mean": float(stdw.mean().item()) if stdw.numel() else 0.0,
                    "stdw_q": _quantiles(stdw),
                    "top1": top1,
                    "logit_margin_mean": float(margin.mean().item()) if margin.numel() else 0.0,
                    "logit_margin_q": _quantiles(margin),
                }

        def _spotcheck_split_alignment(graph, split_name: str, mask: torch.Tensor, bundle: SimpleNamespace, *, n_checks: int = 3) -> None:
            try:
                nodes = mask.nonzero(as_tuple=False).view(-1)
                n_mask = int(nodes.numel())
            except Exception: return
            ds = getattr(bundle, "ds", None)
            n_ds = int(ds.size(0)) if (ds is not None and hasattr(ds, "size")) else -1
            if (n_ds >= 0) and (n_mask != n_ds):
                print(f"[FedDES][Client {self.role}][diag][align][{split_name}][WARN] mask_count ({n_mask}) != ds_rows ({n_ds}).")

        def _print_static_data_diagnostics_once():
            try:
                n_train_graph = int(train_mask.sum().item())
                n_val_graph = int(val_mask.sum().item())
                n_test_graph = int(test_mask.sum().item())
            except Exception: n_train_graph = n_val_graph = n_test_graph = -1
            print(f"[FedDES][Client {self.role}][diag] sizes: N_train={n_train_graph} N_val={n_val_graph} N_test={n_test_graph}")
            _spotcheck_split_alignment(train_val_graph, "train", train_mask, tr, n_checks=3)
            _spotcheck_split_alignment(train_val_graph, "val", val_mask, val, n_checks=3)
            _spotcheck_split_alignment(train_test_graph, "test", test_mask, test, n_checks=3)

        train_margin = None
        val_margin = None
        if gnn_loss_mode in {"meta_labels_regression", "meta_labels_soft_bce"}:
            train_margin = _margin_targets(tr.ds.to(device), tr.y.to(device))
            val_margin = _margin_targets(val.ds.to(device), val.y.to(device))

        # --- UPDATED: evaluate_ensemble with new modes and ReLU support ---
        def evaluate_ensemble(
            logits: torch.Tensor,
            ds: torch.Tensor,
            hard_preds: torch.Tensor | None = None,
        ):
            """
            Aggregates base classifier outputs based on GNN logits.
            
            Modes (self.args.gnn_ens_combination_mode):
              - Dense (Sum=1 via Entmax):
                  'soft': Weighted average of probabilities (ds).
                  'hard': Weighted average of one-hot predictions.
              - Gated (Subset selection via Threshold > 0):
                  'soft_voting':          Average probas of selected experts (Binary weights).
                  'hard_voting':          Vote counts of selected experts (Binary weights).
                  'soft_weighted_voting': Weighted sum of probas (ReLU magnitude weights).
                  'hard_weighted_voting': Weighted votes (ReLU magnitude weights).
            """
            M = logits.size(1)
            C = self.args.num_classes
            ds_local = ds.view(-1, M, C)

            # --- Group A: Dense Aggregation (Entmax) ---
            if combination_mode in {"soft", "hard"}:
                weights = entmax_bisect(logits, alpha=alpha, dim=-1)
                
                if combination_mode == "soft":
                    soft_probs = (weights.unsqueeze(-1) * ds_local).sum(dim=1)
                else: # hard
                    if hard_preds is None: raise ValueError("hard_preds required for 'hard' combination.")
                    one_hot = F.one_hot(hard_preds, num_classes=C).float()
                    soft_probs = (weights.unsqueeze(-1) * one_hot).sum(dim=1)

            # --- Group B: Gated / Voting Aggregation ---
            elif "voting" in combination_mode:
                if hard_preds is None: raise ValueError(f"hard_preds required for '{combination_mode}' mode.")

                # 1. Calculate Raw Weights (Selection Logic)
                if "weighted" in combination_mode:
                    # Meritocratic: Weight by confidence magnitude (ReLU)
                    raw_weights = F.relu(logits)
                else:
                    # Democratic: Binary selection (Logits > 0)
                    raw_weights = (logits > 0).float()

                # 2. Normalize & Handle Fallback
                sum_w = raw_weights.sum(dim=1, keepdim=True)
                fallback_mask = (sum_w == 0)
                # If no one selected, fallback to all (or uniform)
                final_weights = torch.where(fallback_mask, torch.ones_like(raw_weights), raw_weights)
                
                sum_w_final = final_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
                norm_weights = final_weights / sum_w_final

                # 3. Aggregate
                if "soft" in combination_mode: # soft_voting / soft_weighted_voting
                    target = ds_local
                else: # hard_voting / hard_weighted_voting
                    target = F.one_hot(hard_preds, num_classes=C).float()

                soft_probs = (norm_weights.unsqueeze(-1) * target).sum(dim=1)

            else:
                raise ValueError(f"Unknown gnn_ens_combination_mode: {combination_mode}")

            hard_out = soft_probs.argmax(dim=1)
            return soft_probs, hard_out

        # ---------------------------------------------------
        # 3.5) (Optional) Precompute train-local S-S edges
        # ---------------------------------------------------
        use_div = bool(getattr(self.args, "gnn_diversity_regularization", False))
        div_lambda = float(getattr(self.args, "gnn_diversity_lambda", 0.1))
        top_k = getattr(self.args, "gnn_top_k", None)
        top_k = int(top_k) if top_k is not None else None

        ss_edge_index_train = None
        if use_div and int(self.args.num_classes) == 2:
            rel_ss = ("sample", "ss", "sample")
            if rel_ss in train_val_graph.edge_index_dict:
                ei = train_val_graph[rel_ss].edge_index
                train_nodes = train_mask.nonzero(as_tuple=False).view(-1)
                num_total = train_val_graph["sample"].num_nodes
                mapping = torch.full((num_total,), -1, device=ei.device, dtype=torch.long)
                mapping[train_nodes] = torch.arange(train_nodes.numel(), device=ei.device, dtype=torch.long)
                src, dst = ei[0], ei[1]
                src_l = mapping[src]
                dst_l = mapping[dst]
                keep = (src_l >= 0) & (dst_l >= 0)
                ss_edge_index_train = torch.stack([src_l[keep], dst_l[keep]], dim=0)

        # -----------------------------------
        # 4) Training loop + early stopping
        # -----------------------------------
        best_metric = -float("inf")
        best_state = None
        patience_counter = 0

        def is_better(curr: float, best: float) -> bool:
            return curr > best + 1e-6

        # Diagnostics state
        prev_val_top1 = None
        _print_static_data_diagnostics_once()

        for epoch in range(1, epochs + 1):
            # 4.1 Train step
            with torch.set_grad_enabled(True):
                gnn_model.train()

                if use_neighbor_sampling:
                    meta_loss_sum = 0.0
                    rank_loss_sum = 0.0
                    n_batches = 0
                    epoch_loader = sage_train_loader_fn() if sage_train_loader_fn is not None else sage_train_loader
                    drop_edge_rate = float(getattr(self.args, "gnn_drop_edge_rate", 0.0))

                    for batch in epoch_loader:
                        batch = batch.to(device)
                        if drop_edge_rate > 0:
                            edge = batch[("sample", "ss", "sample")]
                            edge_index = edge.edge_index
                            if edge_index is not None and edge_index.numel() > 0:
                                mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_edge_rate
                                edge.edge_index = edge_index[:, mask]
                                if getattr(edge, "edge_attr", None) is not None: edge.edge_attr = edge.edge_attr[mask]

                        out = gnn_model(batch)
                        bs = int(batch["sample"].batch_size)
                        if bs <= 0: continue
                        seed_nid = batch["sample"].n_id[:bs]
                        train_local = train_global_to_local[seed_nid]
                        keep = train_local >= 0
                        if not bool(keep.any()): continue

                        seed_local = train_local[keep]
                        logits = out[:bs][keep]
                        train_meta = tr.meta[seed_local]

                        # --- LOSS LOGIC FIX ---
                        if gnn_loss_mode == "meta_labels_regression":
                            pred_for_loss = torch.tanh(logits)
                            target = train_margin[seed_local]
                            pred_for_ranking = pred_for_loss
                        elif gnn_loss_mode == "meta_labels_soft_bce":
                            raw_margin = train_margin[seed_local]
                            soft_target = (raw_margin + 1.0) / 2.0
                            pred_for_loss = logits
                            target = soft_target
                            pred_for_ranking = logits
                        else:
                            pred_for_loss = logits
                            target = train_meta
                            pred_for_ranking = logits

                        # Main Loss
                        per_elem = criterion_none(pred_for_loss, target)
                        per_sample = per_elem.mean(dim=1)
                        if sample_weights is not None:
                            sw = sample_weights[seed_local].to(per_sample.device)
                            per_sample = per_sample * sw
                        meta_loss = per_sample.mean()

                        # Ranking Loss
                        rank_loss = None
                        rank_lambda = float(getattr(self.args, "gnn_rank_lambda", 0.2))
                        if rank_lambda > 0:
                            rank_margin = float(getattr(self.args, "gnn_rank_margin", 0.5))
                            rank_weights = None
                            if bool(getattr(self.args, "gnn_weight_rank_loss", False)) and (sample_weights is not None):
                                rank_weights = sample_weights[seed_local].to(device)
                            
                            rank_loss = within_sample_pairwise_rank_loss(
                                pred_for_ranking,  # Use correctly scaled input
                                train_meta,        # Use binary truth for pairs
                                margin=rank_margin,
                                max_pairs=getattr(self.args, "gnn_rank_max_pairs", 256),
                                sample_weights=rank_weights,
                            )

                        optimizer.zero_grad()
                        total_loss = meta_loss
                        if rank_loss is not None:
                            total_loss = total_loss * (1-rank_lambda) + rank_lambda * rank_loss
                        total_loss.backward()
                        optimizer.step()

                        meta_loss_sum += float(meta_loss.detach().item())
                        if rank_loss is not None: rank_loss_sum += float(rank_loss.detach().item())
                        n_batches += 1

                    denom = max(n_batches, 1)
                    meta_loss = torch.tensor(meta_loss_sum / denom, device=device)
                    rank_loss = torch.tensor(rank_loss_sum / denom, device=device) if rank_lambda > 0 else None

                # -----------------------------
                # Full-batch training
                # -----------------------------
                else:
                    drop_edge_rate = float(getattr(self.args, "gnn_drop_edge_rate", 0.0))
                    train_graph = train_val_graph
                    if drop_edge_rate > 0:
                        train_graph = train_val_graph.clone()
                        edge = train_graph[("sample", "ss", "sample")]
                        ei = edge.edge_index
                        if ei is not None and ei.numel() > 0:
                            mask = torch.rand(ei.size(1), device=ei.device) > drop_edge_rate
                            edge.edge_index = ei[:, mask]
                            if getattr(edge, "edge_attr", None) is not None: edge.edge_attr = edge.edge_attr[mask]

                    logits = gnn_model(train_graph)[train_mask]
                    train_meta = tr.meta

                    # --- LOSS LOGIC FIX ---
                    if gnn_loss_mode == "meta_labels_regression":
                        pred_for_loss = torch.tanh(logits)
                        target = train_margin
                        pred_for_ranking = pred_for_loss
                    elif gnn_loss_mode == "meta_labels_soft_bce":
                        raw_margin = train_margin
                        soft_target = (raw_margin + 1.0) / 2.0
                        pred_for_loss = logits
                        target = soft_target
                        pred_for_ranking = logits
                    else:
                        pred_for_loss = logits
                        target = train_meta
                        pred_for_ranking = logits

                    per_elem = criterion_none(pred_for_loss, target)
                    per_sample = per_elem.mean(dim=1)
                    if sample_weights is not None:
                        sw = sample_weights.to(per_sample.device)
                        per_sample = per_sample * sw
                    meta_loss = per_sample.mean()

                    # Ranking Loss
                    rank_lambda = float(getattr(self.args, "gnn_rank_lambda", 0.2))
                    rank_loss = None
                    if rank_lambda > 0:
                        rank_margin = float(getattr(self.args, "gnn_rank_margin", 0.5))
                        rank_weights = sample_weights if bool(getattr(self.args, "gnn_weight_rank_loss", False)) else None
                        
                        rank_loss = within_sample_pairwise_rank_loss(
                            pred_for_ranking,
                            train_meta,
                            margin=rank_margin,
                            max_pairs=getattr(self.args, "gnn_rank_max_pairs", 256),
                            sample_weights=rank_weights,
                        )

                    # Diversity
                    div_loss = None
                    if use_div and div_lambda > 0:
                        div_loss = compute_gnn_diversity_loss(
                            logits=logits,
                            ds=tr.ds,
                            y=tr.y,
                            num_classes=int(self.args.num_classes),
                            entmax_alpha=alpha,
                            top_k=top_k,
                            eps=float(getattr(self.args, "gnn_diversity_eps", 1e-6)),
                            ss_edge_index_train=ss_edge_index_train,
                            binary_neighbor_k_cap=self.args.gnn_diversity_binary_neighbor_k_cap,
                        )

                    optimizer.zero_grad()
                    total_loss = meta_loss
                    if rank_loss is not None:
                        total_loss = total_loss * (1 - rank_lambda) + rank_lambda * rank_loss
                    if div_loss is not None:
                        total_loss = total_loss + div_lambda * div_loss
                    total_loss.backward()
                    optimizer.step()

            # -----------------------------
            # 4.2 Validation step
            # -----------------------------
            gnn_model.eval()
            with torch.no_grad():
                val_logits = gnn_model(train_val_graph)[val_mask]

                # Validation Loss (using Regression/BCE logic)
                if gnn_loss_mode == "meta_labels_regression":
                    val_pred = torch.tanh(val_logits)
                    val_meta_loss = criterion_none(val_pred, val_margin).mean()
                elif gnn_loss_mode == "meta_labels_soft_bce":
                    soft_target_val = (val_margin + 1.0) / 2.0
                    val_meta_loss = criterion_none(val_logits, soft_target_val).mean()
                else:
                    val_meta_loss = criterion_none(val_logits, val.meta).mean()

                # Ensemble metrics on val
                val_soft_probs, val_hard_preds = evaluate_ensemble(val_logits, val.ds, val.preds)
                
                # ESS / Stats logic updated for new modes
                if "voting" in combination_mode:
                    if "weighted" in combination_mode:
                        val_selection_matrix = F.relu(val_logits)
                    else:
                        val_selection_matrix = (val_logits > 0).float()
                    
                    val_fallback_mask = val_selection_matrix.sum(dim=1, keepdim=True) == 0
                    if val_fallback_mask.any():
                        val_selection_matrix = torch.where(val_fallback_mask, torch.ones_like(val_selection_matrix), val_selection_matrix)
                    val_fallback_rows = val_fallback_mask.squeeze(1)
                else:
                    val_selection_matrix = entmax_bisect(val_logits, alpha=alpha, dim=-1)
                    val_fallback_rows = None

                if val_fallback_rows is None:
                    val_ess_stats = _ess_stats(val_selection_matrix)
                    val_ess_fallback_stats = None
                    val_fallback_count = 0
                else:
                    val_ess_stats = _ess_stats_or_none(val_selection_matrix[~val_fallback_rows]) or {"mean": 0.0, "q25": 0.0, "med": 0.0, "q75": 0.0}
                    val_ess_fallback_stats = _ess_stats_or_none(val_selection_matrix[val_fallback_rows])
                    val_fallback_count = int(val_fallback_rows.sum().item())

                val_ensemble_size = float((val_selection_matrix > 0).sum(dim=1).float().mean().item()) if val_selection_matrix.numel() else 0.0
                val_acc = (val_hard_preds == val.y).float().mean().item()
                val_bacc = self.balanced_accuracy(val_hard_preds, val.y)
                val_ens_loss = ensemble_criterion(val_soft_probs, val.y)

                # Scheduler & Logging
                perf_metrics = {
                    "val_acc": val_acc,
                    "val_bacc": val_bacc,
                    "val_loss": float(val_meta_loss.item()),
                    "val_ens_loss": float(val_ens_loss.item()),
                    "val_ensemble_size": val_ensemble_size,
                }

                if epoch == 1 or epoch % 10 == 0:
                    print(
                        f"[FedDES][Client {self.role}][diag][val][ep={epoch}] "
                        f"val_loss={perf_metrics['val_loss']:.4f} val_acc={perf_metrics['val_acc']:.4f} val_bacc={perf_metrics['val_bacc']:.4f} "
                        f"val_ens_size={perf_metrics['val_ensemble_size']:.2f}"
                    )

            if es_metric == "val_loss":
                scheduler_metric = perf_metrics["val_loss"]
                curr_metric = -perf_metrics["val_loss"]
            else:
                scheduler_metric = perf_metrics.get(es_metric, perf_metrics["val_loss"])
                curr_metric = perf_metrics.get(es_metric, perf_metrics["val_loss"])

            scheduler.step(scheduler_metric)

            self.meta_history.append({
                "epoch": epoch,
                "train_meta_loss": float(meta_loss.item()),
                "val_loss": float(val_meta_loss.item()),
                "val_ens_loss": float(val_ens_loss.item()),
                "val_acc": float(val_acc),
                "val_bacc": float(val_bacc),
                "val_ensemble_size": float(val_ensemble_size),
                "ess_mean_val": float(val_ess_stats["mean"]),
                "ess_fallback_count_val": int(val_fallback_count),
            })

            if is_better(curr_metric, best_metric):
                best_metric = curr_metric
                best_state = gnn_model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[EarlyStop] epoch={epoch} best_{es_metric}={best_metric:.6f}")
                    break

        if best_state is not None:
            gnn_model.load_state_dict(best_state)

        # ------------------------------------
        # 5) Test-time ensemble evaluation
        # ------------------------------------
        gnn_model.eval()
        with torch.no_grad():
            test_logits = gnn_model(train_test_graph)[test_mask]
            soft_preds, hard_preds = evaluate_ensemble(test_logits, test.ds, test.preds)

            # Updated Logging for new modes
            if "voting" in combination_mode:
                if "weighted" in combination_mode:
                    selection_matrix = F.relu(test_logits)
                else:
                    selection_matrix = (test_logits > 0).float()
            else:
                selection_matrix = entmax_bisect(test_logits, alpha=alpha, dim=-1)

            self._save_meta_selection_summary(selection_matrix, test.y, combination_mode)

            FedDES_acc = (hard_preds == test.y).float().mean().item()
            FedDES_bacc = self.balanced_accuracy(hard_preds, test.y)

        # ------------------------------------
        # 6) Baseline metrics & Performance Summary
        # ------------------------------------
        baseline_path = self.base_dir / f"{self.role}_performance_baselines.json"
        with open(baseline_path, "r") as f:
            loaded = json.load(f)
        
        baselines = loaded.get(combination_mode)
        if baselines is None:
            print(f"[FedDES][Client {self.role}] Baseline missing for {combination_mode}, recomputing.")
            stored = get_performance_baselines(self, graph_data["test"])
            # Fallback to 'hard' if specific mode baseline is missing
            baselines = stored.get(combination_mode, stored.get("hard"))
        
        # Extract all baseline metrics
        local_acc = float(baselines["local_acc"])
        global_acc = float(baselines["global_acc"])
        local_bacc = float(baselines["local_bacc"])
        global_bacc = float(baselines["global_bacc"])
        
        print(
            f"[FedDES][Client {self.role}] Final Test: "
            f"acc={FedDES_acc:.4f} (loc={local_acc:.4f}, glob={global_acc:.4f}) | "
            f"bacc={FedDES_bacc:.4f} (loc={local_bacc:.4f}, glob={global_bacc:.4f})"
        )

        # Populate the full dictionary expected by serverdes.py
        self.perf_summary = {
            "local_acc": local_acc,
            "local_bacc": local_bacc,
            "global_acc": global_acc,
            "global_bacc": global_bacc,
            "FedDES_acc": FedDES_acc,
            "FedDES_bacc": FedDES_bacc,
            "acc_beats_local": int(FedDES_acc > local_acc),
            "bacc_beats_local": int(FedDES_bacc > local_bacc),
            "acc_beats_global": int(FedDES_acc > global_acc),
            "bacc_beats_global": int(FedDES_bacc > global_bacc),
            # Sum of beats (0, 1, or 2)
            "acc_beats_baselines": int((FedDES_acc > local_acc) + (FedDES_acc > global_acc)),
            "bacc_beats_baselines": int((FedDES_bacc > local_bacc) + (FedDES_bacc > global_bacc)),
        }
        
        self._save_phase3_line_plots()

    def _maybe_normalize_sample_feats(self, graph) -> None:
        mode = str(getattr(self.args, "gnn_sample_feat_norm", "none")).lower()
        if mode in {"none", "", "off"}: return
        if "sample" not in graph.node_types: return
        x = graph["sample"].x
        if x is None or x.numel() == 0: return

        if bool(getattr(self.args, "gnn_debug_feat_stats", False)):
            with torch.no_grad():
                mean = float(x.mean().item())
                std = float(x.std().item())
                print(f"[FedDES][Client {self.role}][diag] sample.x stats before norm mean={mean:.4f} std={std:.4f}")

        if mode == "layernorm":
            graph["sample"].x = torch.nn.functional.layer_norm(x, x.shape[1:])
        elif mode == "l2":
            graph["sample"].x = torch.nn.functional.normalize(x, p=2, dim=1)

    def _print_sample_degrees(self, graph, label: str) -> None:
        rel = ("classifier", "cs", "sample")
        if rel not in graph.edge_index_dict:
            print(f"[FedDES][Client {self.role}][diag] {label} graph missing {rel} edges")
            return
        edge_index = graph[rel].edge_index
        sample_count = int(graph["sample"].x.size(0))
        deg = degree(edge_index[1], num_nodes=sample_count)
        zero = int((deg == 0).sum().item())
        mean = float(deg.float().mean().item())
        print(f"[FedDES][Client {self.role}][diag] {label} cs sample degree mean={mean:.2f} zero={zero}/{sample_count}")

    def _drop_disconnected_classifiers(self, train_val_graph, train_test_graph, bundles):
        rel_cs = ("classifier", "cs", "sample")
        if ("classifier" not in train_val_graph.node_types) or (rel_cs not in train_val_graph.edge_index_dict):
            return None
        num_classifiers = int(getattr(train_val_graph["classifier"], "num_nodes", train_val_graph["classifier"].x.size(0)))
        if num_classifiers <= 0: return None

        deg = torch.zeros(num_classifiers, dtype=torch.long)
        for graph in (train_val_graph, train_test_graph):
            if rel_cs in graph.edge_index_dict:
                ei = graph[rel_cs].edge_index
                if ei is None or ei.numel() == 0: continue
                src = ei[0].detach().to("cpu", non_blocking=True).long()
                deg.index_add_(0, src, torch.ones_like(src))

        keep_mask = deg > 0
        keep_count = int(keep_mask.sum().item())
        if keep_count == num_classifiers: return keep_mask
        if keep_count == 0:
            print(f"[FedDES][Client {self.role}][warn] all classifiers have 0 sample connections; skipping dropping.")
            return None

        keep_idx = keep_mask.nonzero(as_tuple=False).view(-1)

        def _filter_graph(graph):
            clf_map = torch.full((num_classifiers,), -1, dtype=torch.long, device=graph["classifier"].x.device)
            clf_map[keep_idx.to(clf_map.device)] = torch.arange(keep_idx.numel(), device=clf_map.device)
            
            clf_store = graph["classifier"]
            for key in list(clf_store.keys()):
                val = clf_store[key]
                if torch.is_tensor(val) and val.size(0) == num_classifiers:
                    clf_store[key] = val[keep_idx.to(val.device)]
                elif isinstance(val, list) and len(val) == num_classifiers:
                    clf_store[key] = [val[i] for i in keep_idx.tolist()]
            clf_store.num_nodes = keep_idx.numel()

            for et in graph.edge_types:
                src_t, _, dst_t = et
                if (src_t != "classifier") and (dst_t != "classifier"): continue
                edge = graph[et]
                ei = edge.edge_index
                if ei is None or ei.numel() == 0: continue
                src, dst = ei[0], ei[1]
                keep = torch.ones(src.size(0), dtype=torch.bool, device=src.device)
                if src_t == "classifier": keep = keep & keep_mask.to(src.device)[src]
                if dst_t == "classifier": keep = keep & keep_mask.to(dst.device)[dst]
                
                if not bool(keep.any()):
                    edge.edge_index = torch.empty((2, 0), dtype=ei.dtype, device=ei.device)
                    if getattr(edge, "edge_attr", None) is not None: edge.edge_attr = edge.edge_attr[:0]
                    continue
                
                new_src = src[keep]
                new_dst = dst[keep]
                if src_t == "classifier": new_src = clf_map[new_src]
                if dst_t == "classifier": new_dst = clf_map[new_dst]
                edge.edge_index = torch.stack([new_src, new_dst], dim=0)
                if getattr(edge, "edge_attr", None) is not None: edge.edge_attr = edge.edge_attr[keep]
                for key, val in list(edge.items()):
                    if key in {"edge_index", "edge_attr"}: continue
                    if torch.is_tensor(val) and val.size(0) == ei.size(1): edge[key] = val[keep]

        def _slice_bundle(bundle):
            if getattr(bundle, "meta", None) is not None: bundle.meta = bundle.meta[:, keep_idx.to(bundle.meta.device)]
            if getattr(bundle, "preds", None) is not None: bundle.preds = bundle.preds[:, keep_idx.to(bundle.preds.device)]
            if getattr(bundle, "ds", None) is not None:
                ds = bundle.ds
                num_classes = int(getattr(self.args, "num_classes", 1))
                if ds.dim() == 2 and ds.size(1) % max(num_classes, 1) == 0:
                    M = int(ds.size(1) // max(num_classes, 1))
                    if M == num_classifiers:
                        ds_view = ds.view(ds.size(0), M, num_classes)
                        ds_view = ds_view[:, keep_idx.to(ds.device), :]
                        bundle.ds = ds_view.reshape(ds.size(0), -1)

        _filter_graph(train_val_graph)
        _filter_graph(train_test_graph)
        for bundle in bundles: _slice_bundle(bundle)

        kept_keys = None
        if hasattr(train_val_graph["classifier"], "clf_keys"):
            kept_keys = train_val_graph["classifier"].clf_keys
        else:
            if len(self.global_clf_keys) == num_classifiers:
                kept_keys = [self.global_clf_keys[i] for i in keep_idx.tolist()]
        if kept_keys is not None: self._active_clf_keys = kept_keys

        print(f"[FedDES][Client {self.role}] Dropped classifiers with 0 sample connections: {num_classifiers} -> {keep_count}")
        return keep_mask

    def _apply_oof_to_train_graph_data(self, train_dict: Dict[str, torch.Tensor], oof_cache: Dict[str, Any]) -> bool:
        if not oof_cache: return False
        local_indices = oof_cache.get("local_clf_indices", [])
        if not local_indices: return False
        ds = train_dict.get("ds")
        y = train_dict.get("y")
        if ds is None or y is None: return False

        num_classes = int(getattr(self.args, "num_classes", 1))
        M = int(ds.size(1) // max(num_classes, 1))
        probs = ds.view(ds.size(0), M, num_classes).clone()
        oof_probs = oof_cache.get("probs", None)
        if oof_probs is None or oof_probs.size(0) != probs.size(0): return False

        probs[:, local_indices, :] = oof_probs.to(probs.device)
        preds = probs.argmax(dim=2)
        meta = compute_meta_labels(probs, preds, y, min_positive=int(getattr(self.args, "graph_meta_min_pos", 3)))

        train_dict["ds"] = probs.reshape(probs.size(0), -1)
        train_dict["preds"] = preds
        train_dict["meta"] = meta.float()
        if str(getattr(self.args, "graph_sample_node_feats", "ds")).lower() == "ds":
            train_dict["feats"] = train_dict["ds"]

        print(f"[FedDES][Client {self.role}] Applied OOF stacking to train split (local classifiers={len(local_indices)}).")
        return True

    def _save_meta_selection_summary(self, selection_matrix: torch.Tensor, target_labels: torch.Tensor, combination_mode: str) -> None:
        dataset_name = getattr(self.args, "dataset", "")
        label_counts_map = load_client_label_counts(dataset_name)
        selection = selection_matrix.detach().cpu().numpy()
        labels_np = target_labels.detach().cpu().numpy()
        total_samples = labels_np.size

        summary = {"client": self.role, "combination_mode": combination_mode, "total_samples": int(total_samples), "rows": []}
        unique_labels = np.unique(labels_np)
        if unique_labels.size == 0 or selection.size == 0: return

        clf_keys = getattr(self, "_active_clf_keys", None) or self.global_clf_keys
        for cls in unique_labels:
            mask = labels_np == cls
            if not mask.any(): continue
            target_class_count = int(mask.sum())
            entries = []
            avg_selection = selection[mask].mean(axis=0)
            for clf_idx, (home_role, model_str) in enumerate(clf_keys):
                home_counts = label_counts_map.get(home_role, {})
                home_total = sum(home_counts.values()) or 1
                support_count = int(home_counts.get(int(cls), 0))
                entries.append({
                    "classifier_idx": clf_idx,
                    "home_client": home_role,
                    "model": model_str,
                    "selection_score": float(avg_selection[clf_idx]),
                    "home_support_count": support_count,
                    "home_support_ratio": float(support_count) / home_total,
                    "home_total": int(home_total),
                })
            summary["rows"].append({
                "target_class": int(cls),
                "target_class_count": target_class_count,
                "target_class_fraction": float(target_class_count) / total_samples if total_samples else 0.0,
                "entries": entries,
            })

        plots_dir = self.graph_outputs_dir / self.role / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        summary_path = plots_dir / "meta_selection.json"
        with open(summary_path, "w") as f: json.dump(summary, f, indent=2)

        support_root = self.graph_outputs_dir / self.role / "phase3_plots" / "support"
        support_root.mkdir(parents=True, exist_ok=True)

        for row in summary["rows"]:
            target_class = row.get("target_class")
            if target_class is None: continue
            class_dir = support_root / f"class_{target_class}"
            class_dir.mkdir(parents=True, exist_ok=True)
            support_x = []
            support_y = []
            support_colors = []
            for entry in row.get("entries", []):
                x = entry.get("home_support_ratio")
                y = entry.get("selection_score")
                if x is None or y is None: continue
                support_x.append(float(x))
                support_y.append(float(y))
                if entry.get("home_client") == self.role: support_colors.append("#d95f02")
                else: support_colors.append("#1b9e77")
            _plot_phase3_scatter(
                support_x,
                support_y,
                title=f"meta selection vs home prevalence (class {target_class})",
                xlabel="home support ratio",
                ylabel="selection score",
                out_path=class_dir / f"{self.role}.png",
                colors=support_colors,
            )

    def _save_phase3_line_plots(self) -> None:
        if not self.meta_history: return
        history_dir = self.graph_outputs_dir / self.role
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / "meta_history.json"
        with open(history_path, "w") as f: json.dump(self.meta_history, f, indent=2)

        plots_root = self.graph_outputs_dir / self.role / "phase3_plots"
        metrics = [
            ("train_loss", "train_meta_loss", "Train meta loss", "train meta loss", None),
            ("val_loss", "val_loss", "Validation loss", "val loss", None),
            ("val_acc", "val_acc", "Validation accuracy", "val acc", None),
            ("val_bacc", "val_bacc", "Validation balanced accuracy", "val balanced acc", None),
            ("ensemble_size", "val_ensemble_size", "Validation ensemble size", "avg selected classifiers", None),
            ("ess/mean_val", "ess_mean_val", "ESS mean (val)", "ESS mean", "ess_fallback_count_val"),
        ]

        for alias, key, title, ylabel, fallback_key in metrics:
            values = [(row["epoch"], row.get(key)) for row in self.meta_history if row.get(key) is not None]
            if not values: continue
            epochs, series = zip(*values)
            secondary = None
            if fallback_key is not None:
                fallback_values = [(row["epoch"], row.get(fallback_key)) for row in self.meta_history if row.get(fallback_key) is not None]
                if fallback_values:
                    fallback_epochs, fallback_series = zip(*fallback_values)
                    secondary = (list(fallback_epochs), list(fallback_series))
            metric_dir = plots_root / alias
            metric_dir.mkdir(parents=True, exist_ok=True)
            out_path = metric_dir / f"{self.role}.png"
            _plot_phase3_line_chart(list(epochs), list(series), title=f"{self.role} {title}", ylabel=ylabel, out_path=out_path, secondary=secondary, secondary_label="fallback")

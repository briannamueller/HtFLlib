from __future__ import annotations
# system/flcore/clients/clientdes.py
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
import torch
import torch.nn as nn
import torch.nn.functional as F  # local import to support one-hot voting
import matplotlib
from torch_geometric.loader import NeighborLoader
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from entmax import entmax_bisect, sparsemax
from sklearn.metrics import precision_score, f1_score
# Shared training helpers

# ---------- DES utilities ----------
from des.dataset_stats import load_client_label_counts
from des.graph_utils_prototypes import build_train_eval_graph, project_to_DS, generate_prototypes
from des.base_clf_utils import fit_clf
from des.helpers import derive_config_ids, init_base_meta_loaders, get_performance_baselines
from des.meta_learner_utils_prototypes import (
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

class clientDESPrototypes(Client):
    """
    Skeleton FedDES client.

    The production version has three responsibilities:
      1. Train or load the per-client base classifiers and their OOF data.
      2. Build decision-space artifacts and the heterograph used by the meta learner.
      3. Train a graph-based meta learner (and optional ensemble head).

    This skeleton exposes the same surface area with stripped-down logic so
    each stage can be implemented and tested incrementally.
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
        # self.gnn_outputs_dir = args.outputs_root / "gnn" / f"base[{self.base_config_id}]_graph[{self.graph_config_id}]_gnn[{self.gnn_config_id}]"

        for dir in [self.base_dir, self.graph_dir, self.gnn_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        for dir in [self.base_outputs_dir, self.graph_outputs_dir]:
            dir.mkdir(parents=True, exist_ok=True)

        # self.base_dir.mkdir(parents=True, exist_ok=True)
        # self.graph_dir.mkdir(parents=True, exist_ok=True)
        # self.gnn_dir.mkdir(parents=True, exist_ok=True)
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
        ok = all(statuses.values()) if expected else True
        return ok

    def _refresh_config_ids_and_dirs(self) -> None:
        self.base_config_id, self.graph_config_id, self.gnn_config_id = derive_config_ids(self.args)
        self.base_dir = self.args.ckpt_root / "base_clf" /  f"base[{self.base_config_id}]"
        self.graph_dir = self.args.ckpt_root / "graphs" / f"base[{self.base_config_id}]_graph[{self.graph_config_id}]"
        self.gnn_dir = self.args.ckpt_root / "gnn" / f"base[{self.base_config_id}]_graph[{self.graph_config_id}]_gnn[{self.gnn_config_id}]"
        self.base_outputs_dir = self.args.outputs_root / "base_clf" /  f"base[{self.base_config_id}]"
        self.graph_outputs_dir = self.args.outputs_root / "graphs" / f"base[{self.base_config_id}]_graph[{self.graph_config_id}]"
        self.gnn_outputs_dir = self.args.outputs_root / "gnn" / f"base[{self.base_config_id}]_graph[{self.graph_config_id}]_gnn[{self.gnn_config_id}]"

        for dir in [self.base_dir, self.graph_dir, self.gnn_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        for dir in [self.base_outputs_dir, self.graph_outputs_dir, self.gnn_outputs_dir]:
            dir.mkdir(parents=True, exist_ok=True)

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
        base_train_loader, _ = init_base_meta_loaders(self)
        val_loader = self.load_val_data()

        for model_id, model_str in zip(self.model_ids, self.model_strs):
            best_epoch, best_score, model= fit_clf(
                self,
                model_id, base_train_loader, val_loader, device,
                max_epochs=self.args.local_epochs,
                patience=self.args.base_es_patience,
                es_metric=self.args.base_es_metric,   # "val_loss" or "val_temp_scaled_loss"
                lr=self.args.base_clf_lr,
                min_delta=self.args.base_es_min_delta,
            )

            torch.save(model.cpu(), self.base_dir /f"{self.role}_{model_str}.pt")
            print(f"{self.role} {model_id} stopping training at epoch {best_epoch}, score = {best_score}")
        
    def prepare_graph_data(self, device=None, classifier_pool: Dict[Any, torch.nn.Module] = None) -> None:
            """
            Prepare decision-space and meta-data for graph building.

            Steps:
            1) Evaluate classifier pool → decision-space (DS), preds, labels,
                meta-labels, and sample features for train/val/test.
            2) Optionally load/save these tensors from/to a flat cache.
            3) (New) Optionally generate and save prototypes from the training data.
            """

            device = torch.device(device if device is not None else self.device)
            self.device = device

            _, meta_train_loader = init_base_meta_loaders(self)
            val_loader = self.load_val_data()
            test_loader = self.load_test_data()

            data_loaders = {"train": meta_train_loader, "val": val_loader, "test": test_loader}

            # Flat cache for graph inputs
            cache_path = (Path(self.base_dir)/ f"{self.role}_graph_bundle.pt")
            
            # Check flag for prototype generation
            generate_protos = getattr(self.args, "proto_use", False)
            proto_path = self._proto_path()

            if cache_path.exists():
                print(f"[FedDES][Client {self.role}] Loading cached graph artifacts from {cache_path}")
                graph_data = torch.load(cache_path, map_location="cpu")
                
                # Logic to retroactively generate prototypes if enabled but missing on disk
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
                for data_split in ["train", "val", "test"]:
                    loader = data_loaders[data_split]
                    ds, preds, y_true, meta_labels, feats = project_to_DS(
                        self, loader, classifier_pool
                    )
                    graph_data[data_split] = {
                        "ds": ds, "preds": preds, "y": y_true, "meta": meta_labels.float(), "feats": feats
                    }
                
                # Generate prototypes if requested (using Training data)
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

        Steps:
        1) Evaluate classifier pool → decision-space (DS), preds, labels,
            meta-labels, and sample features for train/val/test.
        2) Optionally load/save these tensors from/to a flat cache.
        3) Build train+val and train+test hetero-graphs and save them.
        """

        # Flat cache for graph inputs
        cache_path = (Path(self.base_dir)/ f"{self.role}_graph_bundle.pt")

        print(f"[FedDES][Client {self.role}] Loading cached graph artifacts from {cache_path}")
        graph_data = torch.load(cache_path, map_location="cpu")

        tr, val, test = (SimpleNamespace(**graph_data[data_split]) for data_split in ["train", "val", "test"])
        
        peer_prototypes = []
        use_prototypes = getattr(self.args, "proto_use", False)
        
        if use_prototypes:
            # Glob all bundles in the shared folder (e.g., client0_.., client1_..)
            # This simulates the client having access to the "Global Prototype Registry"
            proto_id = self._proto_config_id()
            suffix = f"_prototypes[{proto_id}].pt"
            all_proto_files = [
                p for p in Path(self.base_dir).iterdir()
                if p.is_file() and p.name.endswith(suffix)
            ]
            
            print(f"[FedDES][Client {self.role}] proto_id={proto_id}")
            print(f"[FedDES][Client {self.role}] proto_glob_dir={self.base_dir}")
            print(f"[FedDES][Client {self.role}] proto_glob_suffix={suffix}")
            if not all_proto_files:
                try:
                    present = sorted(p.name for p in Path(self.base_dir).iterdir() if p.is_file())
                except Exception:
                    present = []
                print(f"[FedDES][Client {self.role}] proto_glob_dir_files={present}")

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

        get_performance_baselines(self, graph_data["test"])
        torch.save(train_val_graph,  self.graph_dir / f"{self.role}_graph_train_val.pt")
        torch.save(train_test_graph, self.graph_dir / f"{self.role}_graph_train_test.pt")
        # Save baseline metrics (both soft and hard) while classifier keys are set



    # def train_meta_learner(self, device=None) -> None:
    #     """
    #     Train a simple HeteroGAT meta-learner on the pre-built graphs.

    #     - BCEWithLogitsLoss on meta labels
    #     - Early-stopping on validation loss
    #     - Softmax-weighted ensemble for val (for monitoring) and test (final)
    #     """

    #     device = torch.device(device if device is not None else self.device)

    #     # -----------------------------
    #     # 2) Load decision-space bundle and graphs
    #     # -----------------------------
    #     graph_data = torch.load(self.base_dir / f"{self.role}_graph_bundle.pt", map_location="cpu",)

    #     for data_split in ["train", "val", "test"]:
    #         graph_data[data_split] = {k: v.to(device) for k, v in graph_data[data_split].items()}

    #     tr, val, test = (SimpleNamespace(**graph_data[data_split]) for data_split in ["train", "val", "test"])

    #     train_val_graph = torch.load(self.graph_dir / f"{self.role}_graph_train_val.pt", map_location=device)
    #     train_test_graph = torch.load(self.graph_dir / f"{self.role}_graph_train_test.pt", map_location=device)

    #     bidir=self.args.gnn_bidirectionality
    #     enforce_bidirectionality(train_val_graph, bidir)
    #     enforce_bidirectionality(train_test_graph, bidir)

    #     if self.args.gnn_drop_cc_edges:
    #         drop_cc_edges(train_val_graph)
    #         drop_cc_edges(train_test_graph)

    #     # ---------------------------------------------------------
    #     # 3) Build model + optimizer + simple loss (no extra tricks)
    #     # ---------------------------------------------------------

    #     num_models = self.num_models * self.args.num_clients
    #     metadata = train_val_graph.metadata()
    #     input_dims = {ntype: train_val_graph[ntype].x.size(-1) for ntype in metadata[0]}
    #     gnn_model = build_meta_learner(
    #         self.args,
    #         metadata,
    #         input_dims,
    #         num_models,
    #     ).to(device)

    #     lr, wd = self.args.gnn_lr, self.args.gnn_weight_decay
    #     optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr, weight_decay=wd)
    #     criterion_none = torch.nn.BCEWithLogitsLoss(reduction="none")
    #     sample_weights = compute_sample_weights(self, tr.y, tr.meta, self.args.gnn_sample_weight_mode)

    #     # ---------------------------------------------------
    #     # 4) Masks for train/val/test sample nodes in graphs
    #     # ---------------------------------------------------
    #     train_mask = train_val_graph["sample"].train_mask
    #     val_mask   = train_val_graph["sample"].val_mask
    #     test_mask  = train_test_graph["sample"].test_mask
    #     combination_mode = str(getattr(self.args, "gnn_ens_combination_mode", "soft")).lower()

    #     # Small helper: entmax ensemble over decision space
    #     def evaluate_ensemble(logits: torch.Tensor, ds: torch.Tensor, hard_preds: torch.Tensor | None = None):
    #         """
    #         Return (soft_probs, hard_preds) given logits and decision-space tensor.

    #         logits: [N, M]
    #         ds:     [N, M*C] (flattened probs)
    #         """
    #         M = logits.size(1)   
    #         C = self.args.num_classes

    #         alpha=1.5
    #         weights = entmax_bisect(logits, alpha=alpha, dim=-1)     # [N, M]
    #         ds = ds.view(-1, M, C)                                   # [N, M*C] → [N, M, C]

    #         if combination_mode == "hard":
    #             if hard_preds is None:
    #                 raise ValueError("hard_preds required for hard ensemble combination.")
    #             one_hot = F.one_hot(hard_preds, num_classes=C).float()  # [N, M, C]
    #             soft_probs = (weights.unsqueeze(-1) * one_hot).sum(dim=1)  # [N, C]
    #         elif combination_mode == "voting":
    #             if hard_preds is None:
    #                 raise ValueError("hard_preds required for voting ensemble combination.")
    #             gate = torch.sigmoid(logits)
    #             mask = (gate > 0.5).float()
    #             mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    #             mask_weights = mask / mask_sum
    #             one_hot = F.one_hot(hard_preds, num_classes=C).float()
    #             soft_probs = (mask_weights.unsqueeze(-1) * one_hot).sum(dim=1)
    #         else:
    #             # Weighted averaging: [N, M] x [N, M, C] → [N, C]
    #             soft_probs = (weights.unsqueeze(-1) * ds).sum(dim=1)     # [N, C]
    #         hard_preds = soft_probs.argmax(dim=1)                    # [N]

    #         return soft_probs, hard_preds

    #     # ---------------------------------------------------
    #     # 4.5) (Optional) Precompute train-local sample-sample edges for binary diversity
    #     # ---------------------------------------------------
    #     use_div = bool(getattr(self.args, "gnn_diversity_regularization", False))
    #     div_lambda = float(getattr(self.args, "gnn_diversity_lambda", 0.1))
    #     top_k = getattr(self.args, "gnn_top_k", None)
    #     top_k = int(top_k) if top_k is not None else None

    #     ss_edge_index_train = None
    #     if use_div and int(self.args.num_classes) == 2:
    #         rel_ss = ("sample", "ss", "sample")
    #         if rel_ss not in train_val_graph.edge_index_dict:
    #             raise KeyError(
    #                 f"Binary diversity regularization requested, but {rel_ss} edges not found in train_val_graph."
    #             )
    #         ei = train_val_graph[rel_ss].edge_index  # [2, E] in global sample-node indexing
    #         train_nodes = train_mask.nonzero(as_tuple=False).view(-1)  # [N_train]
    #         num_total = train_val_graph["sample"].num_nodes
    #         mapping = torch.full((num_total,), -1, device=ei.device, dtype=torch.long)
    #         mapping[train_nodes] = torch.arange(train_nodes.numel(), device=ei.device, dtype=torch.long)
    #         src, dst = ei[0], ei[1]
    #         src_l = mapping[src]
    #         dst_l = mapping[dst]
    #         keep = (src_l >= 0) & (dst_l >= 0)
    #         ss_edge_index_train = torch.stack([src_l[keep], dst_l[keep]], dim=0)

    #     # -----------------------------------
    #     # 5) Simple training loop + earlystop
    #     # -----------------------------------
    #     epochs   = int(getattr(self.args, "gnn_epochs", 300))
    #     patience = int(getattr(self.args, "gnn_patience", 20))

    #     es_metric = getattr(self.args, "gnn_es_metric", "val_loss")
    #     best_metric = -float("inf")
    #     best_state = None
    #     patience_counter = 0

    #     def is_better(curr_metric: float, best_metric: float) -> bool:
    #         return curr_metric > best_metric + 1e-6

    #     for epoch in range(1, epochs + 1):
    #         # ---- train step ----
    #         gnn_model.train()
    #         logits = gnn_model(train_val_graph)[train_mask]           # e.g., [N_samples_total, M]
    #         train_meta = tr.meta                      # [N_train, M]

    #         per_elem = criterion_none(logits, train_meta)  # [N_train, M]
    #         per_sample = per_elem.mean(dim=1)              # [N_train]
    #         if sample_weights is not None:
    #             sw = sample_weights.to(per_sample.device)
    #             per_sample = per_sample * sw
    #         loss = per_sample.mean()

    #         alpha=1.5
    #         # ---- optional diversity regularization ----
    #         if use_div and div_lambda > 0:
    #             div_loss = compute_gnn_diversity_loss(
    #                 logits=logits,
    #                 ds=tr.ds,
    #                 y=tr.y,
    #                 num_classes=int(self.args.num_classes),
    #                 entmax_alpha=alpha,
    #                 top_k=top_k,
    #                 eps=float(getattr(self.args, "gnn_diversity_eps", 1e-6)),
    #                 ss_edge_index_train=ss_edge_index_train,
    #                 binary_neighbor_k_cap=self.args.gnn_diversity_binary_neighbor_k_cap,
    #             )
    #             loss = loss + div_lambda * div_loss


    #         optimizer.zero_grad(); loss.backward(); optimizer.step()

    #         # ---- validation step ----
    #         gnn_model.eval()
    #         with torch.no_grad():
    #             val_logits = gnn_model(train_val_graph)[val_mask]
    #             val_loss = criterion_none(val_logits, val.meta).mean()

    #             # Optional: ensemble metrics on val for monitoring (same logic as test)
    #             soft_preds, hard_preds = evaluate_ensemble(val_logits, val.ds, val.preds)
    #             val_acc = (hard_preds == val.y).float().mean().item()
    #             val_bacc = self.balanced_accuracy(hard_preds, val.y)

    #             meta_preds = (val_logits > 0)  
    #             y_true = val.meta.detach().cpu().numpy()
    #             y_pred = meta_preds.detach().cpu().numpy()

    #             val_micro_prec = precision_score(y_true, y_pred, average="micro", zero_division=0)
    #             val_micro_f1 = f1_score(y_true ,y_pred, average="micro", zero_division=0)
    #             perf_metrics = {"val_acc": val_acc, "val_bacc": val_bacc, "val_micro_prec": val_micro_prec, "val_micro_f1": val_micro_f1, "val_loss": float(val_loss.item())}

    #             if es_metric == "val_loss":
    #                 curr_metric = -perf_metrics[es_metric]  # invert for early stopping
    #             else: 
    #                 curr_metric = perf_metrics[es_metric]

    #         if epoch % 10 == 0:
    #             print(
    #                 f"[FedDES][Client {self.role}] "
    #                 f"epoch={epoch} train_loss={loss.item():.4f} "
    #                 f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_bacc={val_bacc:.4f}"
    #             )

    #         # ---- early stopping bookkeeping ----
    #         if is_better(curr_metric, best_metric):
    #             best_metric = curr_metric
    #             best_state = gnn_model.state_dict()
    #             patience_counter = 0
    #         else:
    #             patience_counter += 1
    #             if patience_counter >= patience:
    #                 print(f"[EarlyStop] epoch={epoch} best_{es_metric}={best_metric:.6f}")
    #                 break

    #     if best_state is not None:
    #         gnn_model.load_state_dict(best_state)

    #     # ------------------------------------
    #     # 6) Test-time softmax-gated ensemble
    #     # ------------------------------------
    #     gnn_model.eval()
    #     with torch.no_grad():
    #         test_logits = gnn_model(train_test_graph)[test_mask]
    #         soft_preds, hard_preds = evaluate_ensemble(test_logits, test.ds, test.preds)
    #         # Recompute entmax weights for logging (matches evaluate_ensemble logic).
    #         alpha=1.5
    #         weights = entmax_bisect(test_logits, alpha=alpha, dim=-1)  # [N_test, M]
    #         avg_weights = weights.mean(dim=0)
    #         avg_nonzero_count = (weights.gt(0)).sum(dim=1).float().mean().item()
    #         print(
    #             f"[FedDES][Client {self.role}] Test meta-weight stats: "
    #             f"avg classifier weights={avg_weights.tolist()} | "
    #             f"avg non-zero weights per sample={avg_nonzero_count:.2f}"
    #         )
    #         FedDES_acc = (hard_preds == test.y).float().mean().item()
    #         FedDES_bacc = self.balanced_accuracy(hard_preds, test.y)

    #     # Baseline metrics for summary; uses DS+preds

    #     baseline_path = self.base_dir / f"{self.role}_performance_baselines.json"
    #     with open(baseline_path, "r") as f:
    #         loaded = json.load(f)
    #     baselines = loaded.get(combination_mode)
    #     if baselines is None:
    #         print(f"[FedDES][Client {self.role}] Baseline for combination_mode={combination_mode} missing; recomputing baselines.")
    #         stored = get_performance_baselines(self, graph_data["test"])
    #         baselines = stored.get(combination_mode)
    #         if baselines is None:
    #             baselines = stored.get("hard")
    #         loaded = stored

    #     local_acc, global_acc, local_bacc, global_bacc = (
    #         float(baselines["local_acc"]), float(baselines["global_acc"]), float(baselines["local_bacc"]), float(baselines["global_bacc"])
    #     )

    #     print(
    #         f"[FedDES][Client {self.role}] "
    #         f"des acc={FedDES_acc:.4f} | bacc={FedDES_bacc:.4f}"
    #         f" vs local acc={local_acc:.4f} | bacc={local_bacc:.4f}"
    #         f" vs global acc={global_acc:.4f} | bacc={global_bacc:.4f}"
    #     )

    #     summary_payload = {
    #         "local_acc": local_acc, "local_bacc": local_bacc,
    #         "global_acc": global_acc, "global_bacc": global_bacc,
    #         "FedDES_acc": FedDES_acc, "FedDES_bacc": FedDES_bacc,
    #         "acc_beats_local": int(FedDES_acc > local_acc), "bacc_beats_local": int(FedDES_bacc > local_bacc),
    #         "acc_beats_global": int(FedDES_acc > global_acc), "bacc_beats_global": int(FedDES_bacc > global_bacc),
    #         "acc_beats_baselines": int((FedDES_acc > local_acc) + (FedDES_acc > global_acc)),
    #         "bacc_beats_baselines": int((FedDES_bacc > local_bacc) + (FedDES_bacc > global_bacc)),
    #     }

    #     self.perf_summary = summary_payload


    def train_meta_learner(self, device=None) -> None:
        """
        Train a meta-learner GNN on the pre-built graphs.

        gnn_loss options (self.args.gnn_loss):
          - "meta_labels":   optimize only BCE on meta labels.
          - "ensemble":      optimize only ensemble CrossEntropy on true labels.
        """

        device = torch.device(device if device is not None else self.device)
        self.meta_history = []

        # -----------------------------
        # 0) Loss mode / JD configuration
        # -----------------------------
        ensemble_criterion = torch.nn.CrossEntropyLoss()

        # -----------------------------
        # 1) Load decision-space bundle and graphs
        # -----------------------------
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

        bidir = self.args.gnn_bidirectionality
        enforce_bidirectionality(train_val_graph, bidir)
        enforce_bidirectionality(train_test_graph, bidir)

        if self.args.gnn_drop_cc_edges:
            drop_cc_edges(train_val_graph)
            drop_cc_edges(train_test_graph)

        # ---------------------------------------------------------
        # 2) Build model + optimizer + base meta-label loss
        # ---------------------------------------------------------
        num_models = self.num_models * self.args.num_clients
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

        lr, wd = self.args.gnn_lr, self.args.gnn_weight_decay
        optimizer = torch.optim.Adam(
            gnn_model.parameters(),
            lr=lr,
            weight_decay=wd,
        )

        # Per-sample weights (e.g., inverse class frequency) – preserves your current behavior.
        sample_weights = compute_sample_weights(
            self,
            tr.y,
            tr.meta,
            self.args.gnn_sample_weight_mode,
        )
        # Normalize sample weights so combining with other weighting (e.g., pos_weight) does not
        # unintentionally change the overall gradient scale.
        if sample_weights is not None:
            with torch.no_grad():
                sw = sample_weights.detach().float()
                sw_mean = sw.mean().clamp(min=1e-8)
                sample_weights = (sw / sw_mean).to(device=device)

        # Per-classifier positive weighting to avoid trivial "mostly-zero" meta solutions.
        # pos_weight[m] = (#neg_m / #pos_m). This upweights positive meta-labels for rare-correct classifiers.
        use_pos_weight = bool(getattr(self.args, "gnn_use_pos_weight", False))
        pos_weight = None
        if use_pos_weight:
            with torch.no_grad():
                pos = tr.meta.float().sum(dim=0)  # [M]
                neg = tr.meta.size(0) - pos       # [M]
                pos_weight = (neg / pos.clamp(min=1.0)).to(device=device, dtype=torch.float)

                # Clamp to avoid exploding gradients when a classifier is almost never correct.
                pw_max = float(getattr(self.args, "gnn_pos_weight_max", 10.0))
                if pw_max is not None and pw_max > 0:
                    pos_weight = pos_weight.clamp(max=pw_max)

                # Light diagnostic so we can see if pos_weight is extreme.
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
                        f"min={pw_q['min']:.3f} q25={pw_q['q25']:.3f} med={pw_q['med']:.3f} q75={pw_q['q75']:.3f} max={pw_q['max']:.3f} "
                        f"(clamp_max={float(getattr(self.args, 'gnn_pos_weight_max', 10.0)):.1f})"
                    )
                except Exception:
                    pass

        if pos_weight is None:
            criterion_none = torch.nn.BCEWithLogitsLoss(reduction="none")
        else:
            criterion_none = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        # ---------------------------------------------------
        # 3) Masks for train/val/test sample nodes in graphs
        # ---------------------------------------------------
        train_mask = train_val_graph["sample"].train_mask
        val_mask = train_val_graph["sample"].val_mask
        test_mask = train_test_graph["sample"].test_mask
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
        combination_mode = str(
            getattr(self.args, "gnn_ens_combination_mode", "soft")
        ).lower()
        alpha = 1.5

        arch_name = str(getattr(self.args, "gnn_arch", "hetero_gat")).lower()
        # New: general training sampler toggle
        sampler_mode = str(getattr(self.args, "gnn_sampler", "")).lower()
        if sampler_mode not in {"none", "unweighted", "weighted"}:
            print(
                f"[FedDES][Client {self.role}][warn] Unknown gnn_sampler={sampler_mode}; defaulting to none."
            )
            sampler_mode = "none"
        use_sampler = sampler_mode != "none"

        # These architectures expect a *sample-only* graph.
        # Everything else (hetero_gat / hetero_hgt / hetero_han / hetero_sage_attn)
        # can keep classifier nodes.
        homo_archs = {
            "gat",
            "homog_gat",
            "homogeneous_gat",
            "sage",
            "graphsage",
            "sage_attn",
            "sage+attn",
        }
        use_homo_graph = arch_name in homo_archs

        # SAGE implementations always use neighbor sampling.
        use_homo_sage_sampling = arch_name in {"sage", "graphsage", "sage_attn", "sage+attn"}
        use_hetero_sage_sampling = arch_name in {
            "hetero_sage_attn",
            "hetero_sage",
            "sage_attn_hetero",
            "sage_hetero",
        }
        if (use_homo_sage_sampling or use_hetero_sage_sampling) and not use_sampler:
            use_sampler = True
            sampler_mode = "unweighted"
            print(
                "[FedDES][Client {role}][diag] gnn_sampler forced to unweighted for SAGE-based arch={arch}".format(
                    role=self.role, arch=arch_name
                )
            )

        # Neighbor sampling can be used for any GNN when gnn_sampler != "none".
        use_neighbor_sampling = use_sampler

        # Pairwise decoders require all classifier nodes; NeighborLoader batches do not guarantee that.
        if use_neighbor_sampling and (not use_homo_graph):
            pair_decoder = getattr(self.args, "gnn_pair_decoder", None)
            if pair_decoder is not None and str(pair_decoder).lower() not in {"", "none"}:
                raise ValueError(
                    "NeighborLoader is incompatible with pairwise decoders because batches may "
                    "omit classifier nodes. Disable gnn_sampler or set gnn_pair_decoder='none'."
                )

        # Document potential transductive exposure via ss edges in eval graphs.
        try:
            ss_rel = ("sample", "ss", "sample")
            if ss_rel in train_val_graph.edge_index_dict:
                ei = train_val_graph[ss_rel].edge_index.cpu()
                src, dst = ei[0], ei[1]
                train_mask_cpu = train_mask.cpu()
                val_mask_cpu = val_mask.cpu()
                tv_edges = int((train_mask_cpu[src] & val_mask_cpu[dst]).sum().item())
                vt_edges = int((val_mask_cpu[src] & train_mask_cpu[dst]).sum().item())
                print(
                    "[FedDES][Client {role}][diag] train_val ss edges: train->val={tv} val->train={vt}".format(
                        role=self.role, tv=tv_edges, vt=vt_edges
                    )
                )
            if ss_rel in train_test_graph.edge_index_dict:
                ei = train_test_graph[ss_rel].edge_index.cpu()
                src, dst = ei[0], ei[1]
                train_mask_cpu = train_test_graph["sample"].train_mask.cpu()
                test_mask_cpu = test_mask.cpu()
                tt_edges = int((train_mask_cpu[src] & test_mask_cpu[dst]).sum().item())
                tt_rev_edges = int((test_mask_cpu[src] & train_mask_cpu[dst]).sum().item())
                print(
                    "[FedDES][Client {role}][diag] train_test ss edges: train->test={tt} test->train={tt_rev}".format(
                        role=self.role, tt=tt_edges, tt_rev=tt_rev_edges
                    )
                )
        except Exception as e:
            print(f"[FedDES][Client {self.role}][diag] ss edge exposure check failed: {e}")

        # Map global sample node ids -> train-local row index (same ordering as boolean indexing).
        train_nodes = train_mask.nonzero(as_tuple=False).view(-1)
        num_total_samples = int(train_val_graph["sample"].num_nodes)
        train_global_to_local = torch.full(
            (num_total_samples,),
            -1,
            device=device,
            dtype=torch.long,
        )
        train_global_to_local[train_nodes] = torch.arange(train_nodes.numel(), device=device, dtype=torch.long)

        # Neighbor-sampling training loaders.
        # - Homogeneous GNNs: use a sample-only graph (drop classifier nodes)
        # - Heterogeneous GNNs: keep classifier nodes and sample neighbors across SS + CS (and optional CC/CS_REV)
        sage_train_loader = None
        sage_train_loader_fn = None

        if use_neighbor_sampling:
            # fanouts per layer
            num_layers = int(getattr(self.args, "gnn_layers", 2))
            fanout = int(getattr(self.args, "gnn_sage_fanout", 5))
            sage_num_neighbors = getattr(self.args, "gnn_sage_num_neighbors", None)
            if sage_num_neighbors is None:
                sage_num_neighbors = [fanout] * num_layers
            elif isinstance(sage_num_neighbors, int):
                sage_num_neighbors = [sage_num_neighbors] * num_layers

            sage_batch_size = int(getattr(self.args, "gnn_sage_batch_size", 32))
            sage_shuffle = bool(getattr(self.args, "gnn_sage_shuffle", True))
            if sampler_mode == "weighted":
                sage_weighted = True
            elif sampler_mode == "unweighted":
                sage_weighted = False
            else:
                sage_weighted = bool(getattr(self.args, "gnn_sage_weighted_sampling", True))

            # ----------------------------
            # Path A: homogeneous GNNs (sample-only)
            # ----------------------------
            if use_homo_graph:
                from des.meta_learner_utils import drop_classifier_nodes

                sage_graph = drop_classifier_nodes(train_val_graph)

                # Keep only train->train edges so NeighborLoader cannot traverse to val/test.
                edge = sage_graph[("sample", "ss", "sample")]
                row, col = edge.edge_index
                train_mask_cpu = train_mask.to(row.device)
                keep = train_mask_cpu[row] & train_mask_cpu[col]
                edge.edge_index = edge.edge_index[:, keep]
                if edge.edge_attr is not None:
                    edge.edge_attr = edge.edge_attr[keep]

                try:
                    row_f, col_f = edge.edge_index
                    only_train = bool((train_mask_cpu[row_f] & train_mask_cpu[col_f]).all().item())
                    print(
                        "[FedDES][Client {role}][diag] sage(homo) ss edges train-only={ok} edges={n}".format(
                            role=self.role, ok=only_train, n=int(edge.edge_index.size(1))
                        )
                    )
                except Exception as e:
                    print(f"[FedDES][Client {self.role}][diag] homo sampler edge check failed: {e}")

                # NeighborLoader typically expects CPU graphs.
                sage_graph = sage_graph.cpu()

                # Try to enable weighted sampling if supported (requires edge_attr on the SS relation).
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

                # Guaranteed fallback for homo SAGEAttn: manual weighted SS minibatches
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

            # ----------------------------
            # Path B: heterogeneous GNNs (keep classifier nodes)
            # ----------------------------
            else:
                # Start from the full hetero train_val_graph and prune edges so training batches
                # cannot traverse to val/test sample nodes.
                # IMPORTANT: `Data.to()/cpu()` in PyG is in-place by default.
                # We must not move `train_val_graph` off-device because we still use it for
                # full-batch validation (`gnn_model(train_val_graph)`), so take a copy.
                try:
                    sage_graph = train_val_graph.to("cpu", copy=True)
                except TypeError:
                    # Older PyG versions may not support `copy=`
                    sage_graph = train_val_graph.clone().cpu()

                # NeighborLoader cannot handle non-tensor attributes stored on node/edge stores.
                # Some of our graph-building code may attach Python lists (e.g., names/keys) for debugging.
                # Convert numeric lists to tensors; drop non-numeric lists/objects.
                def _sanitize_store(store, *, store_name: str):
                    for key in list(store.keys()):
                        if key in {"num_nodes"}:
                            continue
                        val = store[key]
                        # Tensors are OK
                        if torch.is_tensor(val):
                            continue
                        # Convert simple numeric lists to tensors; drop everything else
                        if isinstance(val, list):
                            if len(val) == 0:
                                del store[key]
                                continue
                            if all(isinstance(x, (bool, int)) for x in val):
                                store[key] = torch.tensor(val, dtype=torch.long)
                                continue
                            if all(isinstance(x, (bool, int, float)) for x in val):
                                store[key] = torch.tensor(val, dtype=torch.float)
                                continue
                            print(f"[diag] dropping non-tensor attr {store_name}.{key} (list)")
                            del store[key]
                            continue
                        # Drop common non-tensor metadata types
                        if isinstance(val, (str, dict, tuple, set, int, float, bool)):
                            print(f"[diag] dropping non-tensor attr {store_name}.{key} ({type(val).__name__})")
                            del store[key]
                            continue
                        # Fallback: drop anything else we can't index_select
                        print(f"[diag] dropping non-tensor attr {store_name}.{key} ({type(val).__name__})")
                        del store[key]

                # Sanitize node stores
                for ntype in sage_graph.node_types:
                    _sanitize_store(sage_graph[ntype], store_name=f"{ntype}")
                # Sanitize edge stores (e.g., if any edge_attr was accidentally saved as a Python list)
                for etype in sage_graph.edge_types:
                    _sanitize_store(sage_graph[etype], store_name=f"{etype}")

                train_mask_cpu = train_mask.cpu()

                # 1) Prune SS edges to train<->train only.
                rel_ss = ("sample", "ss", "sample")
                if rel_ss in sage_graph.edge_index_dict:
                    edge = sage_graph[rel_ss]
                    row, col = edge.edge_index
                    keep = train_mask_cpu[row] & train_mask_cpu[col]
                    edge.edge_index = edge.edge_index[:, keep]
                    if getattr(edge, "edge_attr", None) is not None:
                        edge.edge_attr = edge.edge_attr[keep]

                # 2) Prune CS edges so only TRAIN samples receive classifier neighbors.
                rel_cs = ("classifier", "cs", "sample")
                if rel_cs in sage_graph.edge_index_dict:
                    edge = sage_graph[rel_cs]
                    src, dst = edge.edge_index
                    keep = train_mask_cpu[dst]
                    edge.edge_index = edge.edge_index[:, keep]
                    if getattr(edge, "edge_attr", None) is not None:
                        edge.edge_attr = edge.edge_attr[keep]

                # 3) If present, prune CS_REV edges so only TRAIN samples send messages to classifiers.
                rel_cs_rev = ("sample", "cs_rev", "classifier")
                if rel_cs_rev in sage_graph.edge_index_dict:
                    edge = sage_graph[rel_cs_rev]
                    src, dst = edge.edge_index
                    keep = train_mask_cpu[src]
                    edge.edge_index = edge.edge_index[:, keep]
                    if getattr(edge, "edge_attr", None) is not None:
                        edge.edge_attr = edge.edge_attr[keep]

                # (CC edges are classifier-only; no train/val/test notion needed)

                # Build NeighborLoader fanouts.
                # IMPORTANT: For HeteroData, NeighborLoader expects a num_neighbors entry for *every* edge type
                # present in the graph (sage_graph.edge_types). We set unused relations to 0.
                update_clf = bool(getattr(self.args, "gnn_update_classifier_nodes", False))
                drop_cc = bool(getattr(self.args, "gnn_drop_cc_edges", False))

                rel_ss = ("sample", "ss", "sample")
                rel_cs = ("classifier", "cs", "sample")
                rel_cs_rev = ("sample", "cs_rev", "classifier")
                rel_cc = ("classifier", "cc", "classifier")

                # Extra safety: if we intend to drop CC edges, try to delete the relation entirely.
                if drop_cc and (rel_cc in sage_graph.edge_index_dict):
                    try:
                        del sage_graph[rel_cc]
                    except Exception as e:
                        print(f"[WARN] could not delete {rel_cc}: {e}")

                # Build NeighborLoader fanouts.
                # NOTE: We keep a full `num_neighbors` dict (all edge types -> 0 by default)
                # for compatibility with older PyG versions, but we explicitly enable the
                # relations we want to sample.

                num_layers = int(getattr(self.args, "gnn_layers", 2))
                fanout = int(getattr(self.args, "gnn_sage_fanout", 5))
                # ss_fanout = fanout
                c_fanout = 3

                # Relations we want to actively sample
                num_neighbors_map = {
                    ("sample", "ss", "sample"): [fanout] * num_layers,
                    ("classifier", "cs", "sample"): [c_fanout] * num_layers,
                }

                # If cc edges were not dropped, also sample them (so hetero SAGE can use CC when present).
                if (not drop_cc) and (rel_cc in sage_graph.edge_index_dict):
                    num_neighbors_map[rel_cc] = [c_fanout] * num_layers

                # Only sample cs_rev if we plan to update classifier nodes and the relation exists.
                if update_clf and (rel_cs_rev in sage_graph.edge_index_dict):
                    num_neighbors_map[rel_cs_rev] = [c_fanout] * num_layers

                # Start with all relations disabled (0 fanout), then enable the requested ones.
                num_neighbors = {et: [0] * num_layers for et in sage_graph.edge_types}
                for et, fo in num_neighbors_map.items():
                    if et in num_neighbors:
                        num_neighbors[et] = fo

                # Light diagnostic
                active_rels = [et for et, fo in num_neighbors.items() if any(v > 0 for v in fo)]
                print(
                    f"[diag] hetero sampler update_classifier_nodes={update_clf} | gnn_drop_cc_edges={drop_cc} | "
                    f"num_neighbors_rels={active_rels}"
                )

                # NeighborLoader expects CPU graphs.
                sage_graph = sage_graph.cpu()

                # Hetero graphs often lack edge_attr on CS/CC relations; only enable weight_attr if all
                # ACTIVELY-SAMPLED relations have edge_attr.
                use_weight_attr = False
                if sage_weighted and _neighborloader_supports_weight_attr():
                    ok = True
                    for et in active_rels:
                        if getattr(sage_graph[et], "edge_attr", None) is None:
                            ok = False
                            break
                    if ok:
                        use_weight_attr = True
                    else:
                        print("[diag] Disabling weight_attr for hetero sampler: missing edge_attr on one or more active relations.")

                # Prefer NeighborLoader for hetero SAGE; manual fallback currently only supports SS-only.
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
                    if use_weight_attr:
                        kwargs["weight_attr"] = "edge_attr"
                    sage_train_loader = NeighborLoader(**kwargs)
                    print(f"[diag] NeighborLoader enabled for hetero sampler with relations={active_rels}")
                except Exception as e:
                    raise RuntimeError(
                        f"NeighborLoader failed for hetero sampler: {e}. "
                        "If you want to proceed, set --gnn_sage_weighted_sampling false and ensure your graph has the needed edge types."
                    )
                
            # If your PyG supports it, this is ideal:
            # if sage_weighted and _neighborloader_supports_weight_attr():
            #     try:
            #         sage_train_loader = NeighborLoader(
            #             train_val_graph,
            #             num_neighbors={('sample', 'ss', 'sample'): [fanout] * num_layers},
            #             input_nodes=('sample', train_nodes),
            #             batch_size=sage_batch_size,
            #             shuffle=sage_shuffle,
            #             weight_attr="edge_attr",
            #         )
            #         print(f"[diag] NeighborLoader(weight_attr='edge_attr') enabled for SAGE.")
            #     except Exception as e:
            #         print(f"[WARN] NeighborLoader(weight_attr) failed: {e}. Falling back to manual sampler.")
            #         sage_train_loader = None

            # # Guaranteed fallback:
            # if sage_train_loader is None:
            #     sage_train_loader = lambda: iter_weighted_ss_minibatches(
            #         train_val_graph,
            #         seed_nodes=train_nodes,
            #         num_neighbors=sage_num_neighbors,
            #         batch_size=sage_batch_size,
            #         device=device,
            #         shuffle=sage_shuffle,
            #         allowed_nodes=train_nodes,
            #     )
            #     print(f"[diag] Manual weighted S-S sampling enabled for SAGE.")

        def within_sample_pairwise_rank_loss(
            logits: torch.Tensor,   # [N, M] (pre-sigmoid)
            meta: torch.Tensor,     # [N, M] in {0,1} or bool
            margin: float = 0.0,
            max_pairs: int | None = None,
            sample_weights: torch.Tensor | None = None,
            ) -> torch.Tensor:
            """Within-sample pairwise ranking loss.

            For each sample i: encourage logits[i, correct] > logits[i, incorrect].
            softplus logistic pairwise loss: softplus(margin - (pos - neg)).

            Returns a scalar tensor on the same device/dtype as logits.
            """
            assert logits.ndim == 2 and meta.shape == logits.shape
            meta_bool = meta.bool()

            N, _ = logits.shape
            losses = []
            weights = []

            for i in range(N):
                pos_i = logits[i][meta_bool[i]]
                neg_i = logits[i][~meta_bool[i]]

                # Skip samples that don't have both sides
                if pos_i.numel() == 0 or neg_i.numel() == 0:
                    continue

                # Optional: sample pairs to avoid O(P*Q) blow-up
                if (max_pairs is not None) and (max_pairs > 0):
                    P = pos_i.numel()
                    Q = neg_i.numel()
                    pos_idx = torch.randint(P, (max_pairs,), device=logits.device)
                    neg_idx = torch.randint(Q, (max_pairs,), device=logits.device)
                    diffs = pos_i[pos_idx] - neg_i[neg_idx]
                    loss_i = F.softplus(margin - diffs).mean()
                else:
                    diffs = pos_i[:, None] - neg_i[None, :]
                    loss_i = F.softplus(margin - diffs).mean()

                losses.append(loss_i)
                if sample_weights is not None:
                    weights.append(sample_weights[i])

            if len(losses) == 0:
                return logits.new_tensor(0.0)

            losses_t = torch.stack(losses)
            if sample_weights is None:
                return losses_t.mean()

            weights_t = torch.stack(weights).to(losses_t)
            denom = weights_t.sum().clamp(min=1e-8)
            return (losses_t * weights_t).sum() / denom
        # -----------------------------
        # Diagnostics helpers
        # -----------------------------
        def _fmt5(x: torch.Tensor) -> str:
            # format first 5 values safely
            if x is None or x.numel() == 0:
                return "[]"
            x = x.detach().flatten()
            k = min(5, x.numel())
            vals = ", ".join([f"{v:.4f}" for v in x[:k].tolist()])
            return f"[{vals}{', ...' if x.numel() > k else ''}]"

        def _quantiles(x: torch.Tensor, qs=(0.0, 0.25, 0.5, 0.75, 1.0)):
            if x is None or x.numel() == 0:
                return {q: float('nan') for q in qs}
            x = x.detach().float().flatten()
            out = {}
            for q in qs:
                out[q] = float(torch.quantile(x, torch.tensor(q, device=x.device)).item())
            return out

        def _ess_stats(weights: torch.Tensor) -> Dict[str, float]:
            if weights is None or weights.numel() == 0:
                return {"mean": 0.0, "q25": 0.0, "med": 0.0, "q75": 0.0}
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
            if weights is None or weights.numel() == 0:
                return None
            return _ess_stats(weights)

        def _entmax_stats(logits_: torch.Tensor):
            with torch.no_grad():
                w = entmax_bisect(logits_, alpha=alpha, dim=-1)
                nnz = (w > 0).sum(dim=1).float()
                maxw = w.max(dim=1).values
                stdw = w.std(dim=1)
                top1 = w.argmax(dim=1)
                # separation in logits
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

        def _sigmoid_norm_weights(logits_: torch.Tensor):
            with torch.no_grad():
                p = torch.sigmoid(logits_)
                denom = p.sum(dim=1, keepdim=True).clamp(min=1e-8)
                return p / denom

        def _evaluate_ensemble_sigmoid_norm(
            logits_: torch.Tensor,
            ds_: torch.Tensor,
            hard_preds_: torch.Tensor | None = None,
        ):
            # Uses sigmoid(logits) as correctness probability then normalizes across classifiers.
            M_ = logits_.size(1)
            C_ = self.args.num_classes
            w_ = _sigmoid_norm_weights(logits_)
            ds_local_ = ds_.view(-1, M_, C_)
            if combination_mode == "hard":
                if hard_preds_ is None:
                    raise ValueError("hard_preds required for hard ensemble combination.")
                one_hot_ = F.one_hot(hard_preds_, num_classes=C_).float()
                soft_probs_ = (w_.unsqueeze(-1) * one_hot_).sum(dim=1)
            elif combination_mode in {"voting", "weighted_vote"}:
                if hard_preds_ is None:
                    raise ValueError("hard_preds required for voting ensemble combination.")
                # voting mode still gates by >0.5; keep current behavior
                gate_ = torch.sigmoid(logits_)
                mask_ = (gate_ > 0.5).float()
                if combination_mode == "weighted_vote":
                    soft_probs_ = (mask_.unsqueeze(-1) * ds_local_).sum(dim=1)
                else:
                    mask_sum_ = mask_.sum(dim=1, keepdim=True).clamp(min=1.0)
                    mask_w_ = mask_ / mask_sum_
                    one_hot_ = F.one_hot(hard_preds_, num_classes=C_).float()
                    soft_probs_ = (mask_w_.unsqueeze(-1) * one_hot_).sum(dim=1)
            else:
                soft_probs_ = (w_.unsqueeze(-1) * ds_local_).sum(dim=1)
            hard_out_ = soft_probs_.argmax(dim=1)
            return soft_probs_, hard_out_

        def _spotcheck_split_alignment(
            graph,
            split_name: str,
            mask: torch.Tensor,
            bundle: SimpleNamespace,
            *,
            n_checks: int = 3,
            tol: float = 1e-5,
        ) -> None:
            """Spot-check that the graph sample-node ordering under `mask` aligns with bundle rows.

            Assumptions this checks:
              - rows of `bundle.ds` / `bundle.meta` correspond to the SAME ordering as
                `graph['sample'].x[mask]` and `gnn_model(graph)[mask]`.

            What it does:
              1) Verifies counts match (mask.sum vs bundle.ds rows).
              2) If `bundle.preds` exists, recomputes meta labels via (preds == y) and compares to bundle.meta.
              3) Prints a few random local rows (local idx, global node idx) with small summaries.
            """

            # Basic sizes
            try:
                nodes = mask.nonzero(as_tuple=False).view(-1)
                n_mask = int(nodes.numel())
            except Exception:
                print(f"[FedDES][Client {self.role}][diag][align][{split_name}] could not read mask nodes")
                return

            ds = getattr(bundle, "ds", None)
            meta = getattr(bundle, "meta", None)
            y = getattr(bundle, "y", None)
            preds = getattr(bundle, "preds", None)

            n_ds = int(ds.size(0)) if (ds is not None and hasattr(ds, "size")) else -1
            n_meta = int(meta.size(0)) if (meta is not None and hasattr(meta, "size")) else -1

            print(
                f"[FedDES][Client {self.role}][diag][align][{split_name}] "
                f"mask_count={n_mask} | ds_rows={n_ds} | meta_rows={n_meta}"
            )
            # Print graph sample feature tensor shape (after the above print)
            sx = getattr(graph["sample"], "x", None) if (graph is not None and ("sample" in graph.node_types)) else None
            if sx is not None:
                print(
                    f"[FedDES][Client {self.role}][diag][align][{split_name}] "
                    f"graph_sample_x_shape={tuple(sx.shape)}"
                )
            else:
                print(
                    f"[FedDES][Client {self.role}][diag][align][{split_name}] "
                    "graph['sample'].x missing"
                )

            if (n_ds >= 0) and (n_mask != n_ds):
                print(
                    f"[FedDES][Client {self.role}][diag][align][{split_name}][WARN] "
                    f"mask_count ({n_mask}) != ds_rows ({n_ds}). "
                    "This suggests ordering/misalignment between graph sample indexing and bundle tensors."
                )

            # 1) Meta-label consistency check (independent of graph)
            if (preds is not None) and (y is not None) and (meta is not None):
                try:
                    # preds expected [N, M], y expected [N]
                    meta_re = (preds == y.view(-1, 1)).float()

                    # If meta-label generation enforces a minimum number of positives per sample
                    # (e.g., pad to at least 5), mirror that logic here so this recompute check
                    # matches the stored bundle.
                    try:
                        min_pos = int(
                            getattr(
                                self.args,
                                "graph_meta_min_pos",
                                getattr(self.args, "graph_meta_min_pos", 3),
                            )
                        )
                    except Exception:
                        min_pos = 3
                    if min_pos > 0:
                        ds_local = getattr(bundle, "ds", None)
                        if ds_local is not None and hasattr(ds_local, "shape"):
                            # ds_local expected [N, M*C]; infer C and score by prob(true class)
                            M_ = int(meta_re.size(1))
                            if ds_local.dim() == 2 and ds_local.size(0) == meta_re.size(0) and ds_local.size(1) % max(M_, 1) == 0:
                                C_ = int(ds_local.size(1) // max(M_, 1))
                                if C_ > 0:
                                    probs = ds_local.view(-1, M_, C_).detach().float()
                                    y_long = y.view(-1).long()
                                    # score each classifier by probability assigned to the true class
                                    p_true = probs[torch.arange(probs.size(0), device=probs.device), :, y_long]

                                    k_now = meta_re.sum(dim=1)
                                    need = (min_pos - k_now).clamp(min=0).to(dtype=torch.long)

                                    if int(need.max().item()) > 0:
                                        # mask out already-correct classifiers so we only add new positives
                                        scores = p_true.clone()
                                        scores[meta_re.bool()] = -1e9

                                        # precompute a fixed top-k per row; then take first `need[i]`
                                        topk_cap = int(min(min_pos, scores.size(1)))
                                        topk_idx = torch.topk(scores, k=topk_cap, dim=1).indices

                                        rows_to_pad = (need > 0).nonzero(as_tuple=False).view(-1)
                                        for i in rows_to_pad.tolist():
                                            n_add = int(need[i].item())
                                            if n_add <= 0:
                                                continue
                                            idx = topk_idx[i, :n_add]
                                            meta_re[i, idx] = 1.0
                            else:
                                print(
                                    f"[FedDES][Client {self.role}][diag][align][{split_name}][WARN] "
                                    "ds shape does not match expected [N, M*C]; skipping pad-to-min_pos meta recompute."
                                )

                    meta_f = meta.float()
                    if meta_re.shape == meta_f.shape:
                        absdiff = (meta_re - meta_f).abs()
                        max_abs = float(absdiff.max().item()) if absdiff.numel() else 0.0
                        mean_abs = float(absdiff.mean().item()) if absdiff.numel() else 0.0
                        frac_mismatch = float((absdiff > 0.5).float().mean().item()) if absdiff.numel() else 0.0
                        print(
                            f"[FedDES][Client {self.role}][diag][align][{split_name}] "
                            f"meta_recompute_check: max_abs={max_abs:.3f} mean_abs={mean_abs:.6f} frac_mismatch={frac_mismatch:.6f}"
                        )
                    else:
                        print(
                            f"[FedDES][Client {self.role}][diag][align][{split_name}][WARN] "
                            f"cannot recompute meta: shape mismatch meta_re={tuple(meta_re.shape)} vs meta={tuple(meta.shape)}"
                        )
                except Exception as e:
                    print(f"[FedDES][Client {self.role}][diag][align][{split_name}][WARN] meta recompute failed: {e}")

            # 3) A few random row spot-checks
            if (ds is None) or (meta is None) or (y is None) or (n_ds <= 0):
                return

            g = torch.Generator(device="cpu")

            # Robust, deterministic seed per client.
            # `self.role` may be a string like "Client_0"; prefer integer `self.id` when available.
            seed_id = getattr(self, "id", None)
            if seed_id is None:
                try:
                    seed_id = int(self.role)
                except Exception:
                    import re
                    m = re.search(r"(\d+)", str(self.role))
                    seed_id = int(m.group(1)) if m is not None else 0

            g.manual_seed(1234 + int(seed_id) * 17)

            k = int(min(max(1, n_checks), n_ds))
            # random local indices into the split-ordered tensors
            local_idx = torch.randperm(n_ds, generator=g)[:k].tolist()

            for j in local_idx:
                global_idx = int(nodes[j].item()) if j < n_mask else -1
                # DS summary
                ds_row = ds[j].detach().float()
                ds_row_absmax = float(ds_row.abs().max().item()) if ds_row.numel() else float("nan")
                ds_row_mean = float(ds_row.mean().item()) if ds_row.numel() else float("nan")

                # Meta label summary
                meta_row = meta[j].detach().float()
                correct_idx = (meta_row > 0.5).nonzero(as_tuple=False).view(-1)
                show_k = int(min(10, correct_idx.numel()))
                correct_preview = correct_idx[:show_k].tolist() if show_k > 0 else []

                print(
                    f"[FedDES][Client {self.role}][diag][align][{split_name}] "
                    f"local_row={j} -> global_node={global_idx} | "
                    f"y={int(y[j].item())} | #correct={int(correct_idx.numel())} correct_idx_head={correct_preview} | "
                    f"ds(mean={ds_row_mean:.4f}, absmax={ds_row_absmax:.4f})"
                )

        def _print_static_data_diagnostics_once():
            # Shapes + meta-label imbalance + easy mismatch checks
            try:
                n_train_graph = int(train_mask.sum().item())
                n_val_graph = int(val_mask.sum().item())
                n_test_graph = int(test_mask.sum().item())
            except Exception:
                n_train_graph = n_val_graph = n_test_graph = -1

            print(
                f"[FedDES][Client {self.role}][diag] sizes: "
                f"N_train_graph={n_train_graph} N_val_graph={n_val_graph} N_test_graph={n_test_graph} | "
                f"tr.meta={tuple(tr.meta.shape)} val.meta={tuple(val.meta.shape)} test.meta={tuple(test.meta.shape)} | "
                f"tr.ds={tuple(tr.ds.shape)} val.ds={tuple(val.ds.shape)} test.ds={tuple(test.ds.shape)}"
            )

            if n_train_graph != int(tr.meta.size(0)):
                print(
                    f"[FedDES][Client {self.role}][diag][WARN] train_mask count ({n_train_graph}) != tr.meta rows ({int(tr.meta.size(0))}). "
                    "This can indicate ordering/misalignment between graph sample indexing and tr.meta/tr.ds."
                )
            if int(tr.meta.size(1)) != int(num_models):
                print(
                    f"[FedDES][Client {self.role}][diag][WARN] tr.meta columns (M={int(tr.meta.size(1))}) != num_models ({int(num_models)})."
                )

            # Meta-label prevalence per classifier (imbalance)
            with torch.no_grad():
                p_m = tr.meta.float().mean(dim=0)  # [M]
                k_i = tr.meta.float().sum(dim=1)   # [N_train]
                pm_q = _quantiles(p_m)
                ki_q = _quantiles(k_i)
                print(
                    f"[FedDES][Client {self.role}][diag] meta-label prevalence p_m (mean over train) "
                    f"min={float(p_m.min().item()):.4f} med={pm_q[0.5]:.4f} max={float(p_m.max().item()):.4f} | "
                    f"head={_fmt5(p_m)}"
                )
                print(
                    f"[FedDES][Client {self.role}][diag] per-sample #correct k_i (sum over classifiers) "
                    f"min={float(k_i.min().item()):.1f} med={ki_q[0.5]:.1f} max={float(k_i.max().item()):.1f} | "
                    f"q25={ki_q[0.25]:.1f} q75={ki_q[0.75]:.1f}"
                )
                # Top/bottom classifiers by correctness prevalence (helps spot extreme imbalance)
                k_show = int(min(5, p_m.numel()))
                if k_show > 0:
                    topv, topi = torch.topk(p_m, k=k_show, largest=True)
                    botv, boti = torch.topk(p_m, k=k_show, largest=False)
                    top_pairs = ", ".join([f"{int(i.item())}:{float(v.item()):.3f}" for i, v in zip(topi, topv)])
                    bot_pairs = ", ".join([f"{int(i.item())}:{float(v.item()):.3f}" for i, v in zip(boti, botv)])
                    print(f"[FedDES][Client {self.role}][diag] p_m top-{k_show} (idx:val) = {top_pairs}")
                    print(f"[FedDES][Client {self.role}][diag] p_m bot-{k_show} (idx:val) = {bot_pairs}")

            # Graph sample feature sanity (not guaranteed to match ds; still useful)
            try:
                sx = train_val_graph["sample"].x
                print(f"[FedDES][Client {self.role}][diag] graph['sample'].x shape={tuple(sx.shape)}")
            except Exception as e:
                print(f"[FedDES][Client {self.role}][diag] could not read graph['sample'].x: {e}")

            # Alignment spot-checks: does (graph[mask]) ordering match bundle row ordering?
            _spotcheck_split_alignment(train_val_graph, "train", train_mask, tr, n_checks=3)
            _spotcheck_split_alignment(train_val_graph, "val", val_mask, val, n_checks=3)
            _spotcheck_split_alignment(train_test_graph, "test", test_mask, test, n_checks=3)

        # Small helper: entmax / voting / hard ensemble over decision space
        def evaluate_ensemble(
            logits: torch.Tensor,
            ds: torch.Tensor,
            hard_preds: torch.Tensor | None = None,
        ):
            """
            Return (soft_probs, hard_preds) given logits and decision-space tensor.

            logits: [N, M]
            ds:     [N, M*C] (flattened probs)
            """
            M = logits.size(1)
            C = self.args.num_classes
            
            weights = entmax_bisect(logits, alpha=alpha, dim=-1)  # [N, M]
            ds_local = ds.view(-1, M, C)                          # [N, M*C] → [N, M, C]

            if combination_mode == "hard":
                # Use non-zero entmax weights on hard base predictions.
                if hard_preds is None:
                    raise ValueError(
                        "hard_preds required for hard ensemble combination."
                    )
                one_hot = F.one_hot(hard_preds, num_classes=C).float()    # [N, M, C]
                soft_probs = (weights.unsqueeze(-1) * one_hot).sum(dim=1) # [N, C]

            elif combination_mode in {"voting", "weighted_vote"}:
                # Gate classifiers via sigmoid(logits) > 0.5, then normalized voting.
                if hard_preds is None:
                    raise ValueError(
                        "hard_preds required for voting ensemble combination."
                    )
                gate = torch.sigmoid(logits)  # [N, M] in [0,1]
                mask = (gate > 0.5).float()   # 0/1 votes
                mask_sum = mask.sum(dim=1, keepdim=True)
                if (mask_sum == 0).any():
                    # Fall back to using all classifiers when none are selected.
                    mask = torch.where(mask_sum == 0, torch.ones_like(mask), mask)
                if combination_mode == "weighted_vote":
                    soft_probs = (mask.unsqueeze(-1) * ds_local).sum(dim=1)
                else:
                    mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                    mask_weights = mask / mask_sum
                    one_hot = F.one_hot(hard_preds, num_classes=C).float()
                    soft_probs = (mask_weights.unsqueeze(-1) * one_hot).sum(dim=1)

            else:
                # "soft": entmax weights over per-class probs [N, M, C] → [N, C]
                soft_probs = (weights.unsqueeze(-1) * ds_local).sum(dim=1)

            hard_out = soft_probs.argmax(dim=1)  # [N]
            return soft_probs, hard_out

        # ---------------------------------------------------
        # 3.5) (Optional) Precompute train-local S-S edges for binary diversity
        # ---------------------------------------------------
        use_div = bool(getattr(self.args, "gnn_diversity_regularization", False))
        div_lambda = float(getattr(self.args, "gnn_diversity_lambda", 0.1))
        top_k = getattr(self.args, "gnn_top_k", None)
        top_k = int(top_k) if top_k is not None else None

        ss_edge_index_train = None
        if use_div and int(self.args.num_classes) == 2:
            rel_ss = ("sample", "ss", "sample")
            if rel_ss not in train_val_graph.edge_index_dict:
                raise KeyError(
                    f"Binary diversity regularization requested, but {rel_ss} "
                    "edges not found in train_val_graph."
                )
            ei = train_val_graph[rel_ss].edge_index  # [2, E] global indexing
            train_nodes = train_mask.nonzero(as_tuple=False).view(-1)  # [N_train]
            num_total = train_val_graph["sample"].num_nodes
            mapping = torch.full(
                (num_total,),
                -1,
                device=ei.device,
                dtype=torch.long,
            )
            mapping[train_nodes] = torch.arange(
                train_nodes.numel(),
                device=ei.device,
                dtype=torch.long,
            )
            src, dst = ei[0], ei[1]
            src_l = mapping[src]
            dst_l = mapping[dst]
            keep = (src_l >= 0) & (dst_l >= 0)
            ss_edge_index_train = torch.stack(
                [src_l[keep], dst_l[keep]],
                dim=0,
            )

        # -----------------------------------
        # 4) Training loop + early stopping
        # -----------------------------------
        epochs = int(getattr(self.args, "gnn_epochs", 300))
        patience = int(getattr(self.args, "gnn_patience", 20))

        es_metric = getattr(self.args, "gnn_es_metric", "val_loss")
        best_metric = -float("inf")
        best_state = None
        patience_counter = 0

        def is_better(curr: float, best: float) -> bool:
            return curr > best + 1e-6

        # -----------------------------
        # Diagnostics state
        # -----------------------------
        prev_val_top1 = None
        prev_train_top1 = None
        _print_static_data_diagnostics_once()

        for epoch in range(1, epochs + 1):
            # -----------------------------
            # 4.1 Train step (force autograd ON)
            # -----------------------------
            with torch.set_grad_enabled(True):
                gnn_model.train()

                # -----------------------------
                # Neighbor-sampled training
                # -----------------------------
                if use_neighbor_sampling:
                    meta_loss_sum = 0.0
                    ens_loss_sum = 0.0
                    rank_loss_sum = 0.0
                    n_batches = 0

                    # epoch_loader = sage_train_loader() if callable(sage_train_loader) else sage_train_loader
                    epoch_loader = sage_train_loader_fn() if sage_train_loader_fn is not None else sage_train_loader

                    for batch in epoch_loader:
                        batch = batch.to(device)

                        out = gnn_model(batch)  # [n_subgraph, M]
                        bs = int(batch["sample"].batch_size)
                        if bs <= 0:
                            continue

                        # First `bs` nodes are the seed nodes
                        seed_nid = batch["sample"].n_id[:bs]
                        train_local = train_global_to_local[seed_nid]
                        keep = train_local >= 0
                        if not bool(keep.any()):
                            continue

                        seed_local = train_local[keep]
                        logits = out[:bs][keep]                 # [B, M]
                        train_meta = tr.meta[seed_local]        # [B, M]

                        # --- meta-label loss (BCE on classifier correctness) ---
                        per_elem = criterion_none(logits, train_meta)  # [B, M]
                        per_sample = per_elem.mean(dim=1)              # [B]
                        if sample_weights is not None:
                            sw = sample_weights[seed_local].to(per_sample.device)
                            per_sample = per_sample * sw
                        meta_loss = per_sample.mean()

                        # --- within-sample ranking loss (optional) ---
                        rank_lambda = float(getattr(self.args, "gnn_rank_lambda", 0.2))
                        rank_margin = float(getattr(self.args, "gnn_rank_margin", 0.5))
                        rank_max_pairs = getattr(self.args, "gnn_rank_max_pairs", 256)
                        rank_max_pairs = int(rank_max_pairs) if rank_max_pairs is not None else None
                        rank_loss = None
                        if rank_lambda > 0:
                            weight_rank_loss = bool(getattr(self.args, "gnn_weight_rank_loss", False))
                            rank_weights = None
                            if weight_rank_loss and (sample_weights is not None):
                                rank_weights = sample_weights[seed_local].to(device)
                            rank_loss = within_sample_pairwise_rank_loss(
                                logits,
                                train_meta,
                                margin=rank_margin,
                                max_pairs=rank_max_pairs,
                                sample_weights=rank_weights,
                            )

                        # NOTE: Diversity regularization depends on global S-S structure; skip for neighbor-sampled updates.
                        div_loss = None

                        optimizer.zero_grad()

                        total_loss = meta_loss
                        if rank_loss is not None:
                            total_loss = total_loss * (1-rank_lambda) + rank_lambda * rank_loss
                        total_loss.backward()


                        optimizer.step()

                        meta_loss_sum += float(meta_loss.detach().item())
                        if rank_loss is not None:
                            rank_loss_sum += float(rank_loss.detach().item())
                        n_batches += 1

                    # Epoch-level logging tensors (not used for backprop)
                    denom = max(n_batches, 1)
                    meta_loss = torch.tensor(meta_loss_sum / denom, device=device)
                    rank_loss = torch.tensor(rank_loss_sum / denom, device=device) if rank_lambda > 0 else None

                # -----------------------------
                # Full-batch training (original behavior for non-SAGE)
                # -----------------------------
                else:
                    logits = gnn_model(train_val_graph)[train_mask]  # [N_train, M]
                    train_meta = tr.meta                              # [N_train, M]

                    # --- meta-label loss (BCE on classifier correctness) ---
                    per_elem = criterion_none(logits, train_meta)  # [N_train, M]
                    per_sample = per_elem.mean(dim=1)              # [N_train]
                    if sample_weights is not None:
                        sw = sample_weights.to(per_sample.device)
                        per_sample = per_sample * sw
                    meta_loss = per_sample.mean()                  # scalar, requires_grad=True

                    # --- within-sample ranking loss (encourage correct classifiers to outrank incorrect ones) ---
                    rank_lambda = float(getattr(self.args, "gnn_rank_lambda", 0.2))
                    rank_margin = float(getattr(self.args, "gnn_rank_margin", 0.5))
                    rank_max_pairs = getattr(self.args, "gnn_rank_max_pairs", 256)
                    rank_max_pairs = int(rank_max_pairs) if rank_max_pairs is not None else None

                    # optional: subsample samples to keep this cheap (recommended if N_train is large)
                    rank_sample_n = getattr(self.args, "gnn_rank_sample_n", 256)
                    rank_sample_n = int(rank_sample_n) if rank_sample_n is not None else None

                    rank_loss = None

                    if rank_lambda > 0:
                        # if rank_sample_n is not None and rank_sample_n > 0 and rank_sample_n < logits.size(0):
                        #     idx = torch.randperm(logits.size(0), device=logits.device)[:rank_sample_n]
                        #     rank_loss = within_sample_pairwise_rank_loss(
                        #         logits[idx], train_meta[idx], margin=rank_margin, max_pairs=rank_max_pairs
                        #     )
                        # else:
                        weight_rank_loss = bool(getattr(self.args, "gnn_weight_rank_loss", False))
                        rank_weights = sample_weights if (weight_rank_loss and sample_weights is not None) else None
                        rank_loss = within_sample_pairwise_rank_loss(
                            logits,
                            train_meta,
                            margin=rank_margin,
                            max_pairs=rank_max_pairs,
                            sample_weights=rank_weights,
                        )

                    # --- diversity regularization ---
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

                # ---- diagnostics on VAL: selection behavior + alternative weighting sanity ----
                if epoch == 1 or epoch % 10 == 0:
                    lt = val_logits.detach()
                    esv = _entmax_stats(lt)
                    top1v = esv["top1"].detach()
                    frac_change_v = float('nan')
                    if prev_val_top1 is not None and prev_val_top1.numel() == top1v.numel():
                        frac_change_v = float((top1v != prev_val_top1).float().mean().item())
                    prev_val_top1 = top1v.clone()

                    std_m_v = lt.std(dim=1)
                    std_m_v_q = _quantiles(std_m_v)

                    print(
                        f"[FedDES][Client {self.role}][diag][val][ep={epoch}] "
                        f"logits std_across_M mean={float(std_m_v.mean().item()):.4f} "
                        f"(q25={std_m_v_q[0.25]:.4f}, med={std_m_v_q[0.5]:.4f}, q75={std_m_v_q[0.75]:.4f}) | "
                        f"entmax nnz_mean={esv['nnz_mean']:.2f} maxw_mean={esv['maxw_mean']:.3f} "
                        f"logit_margin_mean={esv['logit_margin_mean']:.3f} top1_change={frac_change_v if frac_change_v==frac_change_v else float('nan'):.3f}"
                    )

                    # Compare entmax-weighted vs sigmoid-normalized-weighted ensemble accuracy on VAL
                    try:
                        # entmax ensemble (compute locally to avoid using variables defined later)
                        val_soft_probs_ent, val_hard_ent = evaluate_ensemble(lt, val.ds, val.preds)
                        val_acc_ent = float((val_hard_ent == val.y).float().mean().item())
                        val_bacc_ent = float(self.balanced_accuracy(val_hard_ent, val.y))

                        # sigmoid-normalized ensemble
                        val_soft_probs_sig, val_hard_sig = _evaluate_ensemble_sigmoid_norm(lt, val.ds, val.preds)
                        val_acc_sig = float((val_hard_sig == val.y).float().mean().item())
                        val_bacc_sig = float(self.balanced_accuracy(val_hard_sig, val.y))

                        print(
                            f"[FedDES][Client {self.role}][diag][val][ep={epoch}] "
                            f"acc(entmax)={val_acc_ent:.4f} bacc(entmax)={val_bacc_ent:.4f} | "
                            f"acc(sig_norm)={val_acc_sig:.4f} bacc(sig_norm)={val_bacc_sig:.4f}"
                        )
                    except Exception as e:
                        print(f"[FedDES][Client {self.role}][diag][val][ep={epoch}] sigmoid-norm eval failed: {e}")

                # Meta-label validation loss (kept for backwards compatibility)
                val_meta_loss = criterion_none(val_logits, val.meta).mean()

                # Ensemble metrics on val
                val_soft_probs, val_hard_preds = evaluate_ensemble(val_logits, val.ds, val.preds)
                if combination_mode in {"voting", "weighted_vote"}:
                    val_gate = torch.sigmoid(val_logits)
                    val_selection_matrix = (val_gate > 0.5).float()
                    val_fallback_mask = val_selection_matrix.sum(dim=1, keepdim=True) == 0
                    if val_fallback_mask.any():
                        val_selection_matrix = torch.where(
                            val_fallback_mask,
                            torch.ones_like(val_selection_matrix),
                            val_selection_matrix,
                        )
                    val_fallback_rows = val_fallback_mask.squeeze(1)
                else:
                    val_selection_matrix = entmax_bisect(val_logits, alpha=alpha, dim=-1)
                    val_fallback_rows = None
                if val_fallback_rows is None:
                    val_ess_stats = _ess_stats(val_selection_matrix)
                    val_ess_fallback_stats = None
                    val_fallback_count = 0
                else:
                    val_ess_stats = _ess_stats_or_none(val_selection_matrix[~val_fallback_rows]) or {
                        "mean": 0.0,
                        "q25": 0.0,
                        "med": 0.0,
                        "q75": 0.0,
                    }
                    val_ess_fallback_stats = _ess_stats_or_none(val_selection_matrix[val_fallback_rows])
                    val_fallback_count = int(val_fallback_rows.sum().item())
                val_ensemble_size = float((val_selection_matrix > 0).sum(dim=1).float().mean().item()) if val_selection_matrix.numel() else 0.0
                val_acc = (val_hard_preds == val.y).float().mean().item()
                val_bacc = self.balanced_accuracy(val_hard_preds, val.y)

                # ESS stats on train (same selection logic as val)
                train_logits = gnn_model(train_val_graph)[train_mask]
                if combination_mode in {"voting", "weighted_vote"}:
                    train_gate = torch.sigmoid(train_logits)
                    train_selection_matrix = (train_gate > 0.5).float()
                    train_fallback_mask = train_selection_matrix.sum(dim=1, keepdim=True) == 0
                    if train_fallback_mask.any():
                        train_selection_matrix = torch.where(
                            train_fallback_mask,
                            torch.ones_like(train_selection_matrix),
                            train_selection_matrix,
                        )
                    train_fallback_rows = train_fallback_mask.squeeze(1)
                else:
                    train_selection_matrix = entmax_bisect(train_logits, alpha=alpha, dim=-1)
                    train_fallback_rows = None
                if train_fallback_rows is None:
                    train_ess_stats = _ess_stats(train_selection_matrix)
                    train_ess_fallback_stats = None
                    train_fallback_count = 0
                else:
                    train_ess_stats = _ess_stats_or_none(train_selection_matrix[~train_fallback_rows]) or {
                        "mean": 0.0,
                        "q25": 0.0,
                        "med": 0.0,
                        "q75": 0.0,
                    }
                    train_ess_fallback_stats = _ess_stats_or_none(train_selection_matrix[train_fallback_rows])
                    train_fallback_count = int(train_fallback_rows.sum().item())

                val_ens_loss = ensemble_criterion(
                    val_soft_probs,
                    val.y,
                )

                meta_preds = (val_logits > 0)
                y_true = val.meta.detach().cpu().numpy()
                y_pred = meta_preds.detach().cpu().numpy()

                val_micro_prec = precision_score(
                    y_true,
                    y_pred,
                    average="micro",
                    zero_division=0,
                )
                val_micro_f1 = f1_score(
                    y_true,
                    y_pred,
                    average="micro",
                    zero_division=0,
                )

                # Default ES uses "val_loss" = meta-label loss
                perf_metrics = {
                    "val_acc": val_acc,
                    "val_bacc": val_bacc,
                    "val_micro_prec": val_micro_prec,
                    "val_micro_f1": val_micro_f1,
                    "val_loss": float(val_meta_loss.item()),
                    "val_ens_loss": float(val_ens_loss.item()),
                    "val_ensemble_size": val_ensemble_size,
                }

                if epoch == 1 or epoch % 10 == 0:
                    print(
                        f"[FedDES][Client {self.role}][diag][val][ep={epoch}] "
                        f"earlystop_metric={es_metric} "
                        f"val_loss(meta)={perf_metrics['val_loss']:.4f} val_ens_loss={perf_metrics['val_ens_loss']:.4f} "
                        f"val_acc={perf_metrics['val_acc']:.4f} val_bacc={perf_metrics['val_bacc']:.4f} "
                        f"val_ensemble_size={perf_metrics['val_ensemble_size']:.2f}"
                    )

            if es_metric == "val_loss":
                curr_metric = -perf_metrics[es_metric]  # minimize loss
            else:
                curr_metric = perf_metrics[es_metric]   # maximize metric


            self.meta_history.append({
                "epoch": epoch,
                "train_meta_loss": float(meta_loss.item()),
                "val_loss": float(val_meta_loss.item()),
                "val_ens_loss": float(val_ens_loss.item()),
                "val_acc": float(val_acc),
                "val_bacc": float(val_bacc),
                "val_micro_prec": float(val_micro_prec),
                "val_micro_f1": float(val_micro_f1),
                "val_ensemble_size": float(val_ensemble_size),
                "ess_mean_train": float(train_ess_stats["mean"]),
                "ess_q25_train": float(train_ess_stats["q25"]),
                "ess_med_train": float(train_ess_stats["med"]),
                "ess_q75_train": float(train_ess_stats["q75"]),
                "ess_mean_train_fallback": float(train_ess_fallback_stats["mean"]) if train_ess_fallback_stats else None,
                "ess_q25_train_fallback": float(train_ess_fallback_stats["q25"]) if train_ess_fallback_stats else None,
                "ess_med_train_fallback": float(train_ess_fallback_stats["med"]) if train_ess_fallback_stats else None,
                "ess_q75_train_fallback": float(train_ess_fallback_stats["q75"]) if train_ess_fallback_stats else None,
                "ess_fallback_count_train": int(train_fallback_count),
                "ess_mean_val": float(val_ess_stats["mean"]),
                "ess_q25_val": float(val_ess_stats["q25"]),
                "ess_med_val": float(val_ess_stats["med"]),
                "ess_q75_val": float(val_ess_stats["q75"]),
                "ess_mean_val_fallback": float(val_ess_fallback_stats["mean"]) if val_ess_fallback_stats else None,
                "ess_q25_val_fallback": float(val_ess_fallback_stats["q25"]) if val_ess_fallback_stats else None,
                "ess_med_val_fallback": float(val_ess_fallback_stats["med"]) if val_ess_fallback_stats else None,
                "ess_q75_val_fallback": float(val_ess_fallback_stats["q75"]) if val_ess_fallback_stats else None,
                "ess_fallback_count_val": int(val_fallback_count),
                "es_metric": float(perf_metrics[es_metric]),
                "es_metric_name": es_metric,
            })

            if epoch % 10 == 0:
                msg = (
                    f"[FedDES][Client {self.role}] epoch={epoch} "
                    f"train_meta_loss={meta_loss.item():.4f} "
                )
                msg += (
                    f"val_loss(meta)={val_meta_loss:.4f} "
                    f"val_ens_loss={val_ens_loss:.4f} "
                    f"val_acc={val_acc:.4f} val_bacc={val_bacc:.4f}"
                )
                print(msg)

            # -----------------------------
            # 4.3 Early stopping bookkeeping
            # -----------------------------
            if is_better(curr_metric, best_metric):
                best_metric = curr_metric
                best_state = gnn_model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"[EarlyStop] epoch={epoch} best_{es_metric}={best_metric:.6f}"
                    )
                    break

        if best_state is not None:
            gnn_model.load_state_dict(best_state)

        # ------------------------------------
        # 5) Test-time ensemble evaluation
        # ------------------------------------
        gnn_model.eval()
        with torch.no_grad():
            test_logits = gnn_model(train_test_graph)[test_mask]
            soft_preds, hard_preds = evaluate_ensemble(
                test_logits,
                test.ds,
                test.preds,
            )

            alpha = 1.5
            weights = entmax_bisect(test_logits, alpha=alpha, dim=-1)  # [N_test, M]
            avg_weights = weights.mean(dim=0)
            avg_nonzero_count = (weights.gt(0)).sum(dim=1).float().mean().item()
            print(
                f"[FedDES][Client {self.role}] Test meta-weight stats: "
                f"avg classifier weights={avg_weights.tolist()} | "
                f"avg non-zero weights per sample={avg_nonzero_count:.2f}"
            )

            if combination_mode in {"voting", "weighted_vote"}:
                gate = torch.sigmoid(test_logits)
                if combination_mode == "weighted_vote":
                    selection_matrix = gate * (gate > 0.5).float()
                else:
                    selection_matrix = (gate > 0.5).float()
            else:
                selection_matrix = weights
            self._save_meta_selection_summary(selection_matrix, test.y, combination_mode)

            FedDES_acc = (hard_preds == test.y).float().mean().item()
            FedDES_bacc = self.balanced_accuracy(hard_preds, test.y)

        # ------------------------------------
        # 6) Baseline metrics for summary
        # ------------------------------------
        baseline_path = self.base_dir / f"{self.role}_performance_baselines.json"
        with open(baseline_path, "r") as f:
            loaded = json.load(f)
        baselines = loaded.get(combination_mode)
        if baselines is None:
            print(
                f"[FedDES][Client {self.role}] Baseline for combination_mode={combination_mode} "
                f"missing; recomputing baselines."
            )
            stored = get_performance_baselines(self, graph_data["test"])
            baselines = stored.get(combination_mode)
            if baselines is None:
                baselines = stored.get("hard")
            loaded = stored

        local_acc, global_acc, local_bacc, global_bacc = (
            float(baselines["local_acc"]),
            float(baselines["global_acc"]),
            float(baselines["local_bacc"]),
            float(baselines["global_bacc"]),
        )

        print(
            f"[FedDES][Client {self.role}] "
            f"des acc={FedDES_acc:.4f} | bacc={FedDES_bacc:.4f}"
            f" vs local acc={local_acc:.4f} | bacc={local_bacc:.4f}"
            f" vs global acc={global_acc:.4f} | bacc={global_bacc:.4f}"
        )

        summary_payload = {
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
            "acc_beats_baselines": int(
                (FedDES_acc > local_acc) + (FedDES_acc > global_acc)
            ),
            "bacc_beats_baselines": int(
                (FedDES_bacc > local_bacc) + (FedDES_bacc > global_bacc)
            ),
        }

        self.perf_summary = summary_payload

        self._save_phase3_line_plots()

    def _save_meta_selection_summary(
        self,
        selection_matrix: torch.Tensor,
        target_labels: torch.Tensor,
        combination_mode: str,
    ) -> None:
        dataset_name = getattr(self.args, "dataset", "")
        label_counts_map = load_client_label_counts(dataset_name)
        selection = selection_matrix.detach().cpu().numpy()
        labels_np = target_labels.detach().cpu().numpy()
        total_samples = labels_np.size

        summary = {
            "client": self.role,
            "combination_mode": combination_mode,
            "total_samples": int(total_samples),
            "rows": [],
        }

        unique_labels = np.unique(labels_np)
        if unique_labels.size == 0 or selection.size == 0:
            return

        for cls in unique_labels:
            mask = labels_np == cls
            if not mask.any():
                continue
            target_class_count = int(mask.sum())
            entries = []
            avg_selection = selection[mask].mean(axis=0)
            for clf_idx, (home_role, model_str) in enumerate(self.global_clf_keys):
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
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        support_root = self.graph_outputs_dir / self.role / "phase3_plots" / "support"
        support_root.mkdir(parents=True, exist_ok=True)

        for row in summary["rows"]:
            target_class = row.get("target_class")
            if target_class is None:
                continue
            class_dir = support_root / f"class_{target_class}"
            class_dir.mkdir(parents=True, exist_ok=True)
            support_x = []
            support_y = []
            support_colors = []
            for entry in row.get("entries", []):
                x = entry.get("home_support_ratio")
                y = entry.get("selection_score")
                if x is None or y is None:
                    continue
                support_x.append(float(x))
                support_y.append(float(y))
                if entry.get("home_client") == self.role:
                    support_colors.append("#d95f02")
                else:
                    support_colors.append("#1b9e77")
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
        if not self.meta_history:
            return

        history_dir = self.graph_outputs_dir / self.role
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / "meta_history.json"
        with open(history_path, "w") as f:
            json.dump(self.meta_history, f, indent=2)

        plots_root = self.graph_outputs_dir / self.role / "phase3_plots"
        metrics = [
            ("train_loss", "train_meta_loss", "Train meta loss", "train meta loss", None),
            ("val_loss", "val_loss", "Validation loss", "val loss", None),
            ("val_acc", "val_acc", "Validation accuracy", "val acc", None),
            ("val_bacc", "val_bacc", "Validation balanced accuracy", "val balanced acc", None),
            ("ensemble_size", "val_ensemble_size", "Validation ensemble size", "avg selected classifiers", None),
            ("ess/mean_val", "ess_mean_val", "ESS mean (val)", "ESS mean", "ess_mean_val_fallback"),
            ("ess/q25_val", "ess_q25_val", "ESS Q25 (val)", "ESS Q25", "ess_q25_val_fallback"),
            ("ess/med_val", "ess_med_val", "ESS median (val)", "ESS median", "ess_med_val_fallback"),
            ("ess/q75_val", "ess_q75_val", "ESS Q75 (val)", "ESS Q75", "ess_q75_val_fallback"),
        ]

        for alias, key, title, ylabel, fallback_key in metrics:
            values = [(row["epoch"], row.get(key)) for row in self.meta_history if row.get(key) is not None]
            if not values:
                continue
            epochs, series = zip(*values)
            secondary = None
            if fallback_key is not None:
                fallback_values = [
                    (row["epoch"], row.get(fallback_key))
                    for row in self.meta_history
                    if row.get(fallback_key) is not None
                ]
                if fallback_values:
                    fallback_epochs, fallback_series = zip(*fallback_values)
                    secondary = (list(fallback_epochs), list(fallback_series))
            metric_dir = plots_root / alias
            metric_dir.mkdir(parents=True, exist_ok=True)
            out_path = metric_dir / f"{self.role}.png"
            _plot_phase3_line_chart(
                list(epochs),
                list(series),
                title=f"{self.role} {title}",
                ylabel=ylabel,
                out_path=out_path,
                secondary=secondary,
                secondary_label="fallback",
            )

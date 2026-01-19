from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from des.edge_builders_prototypes import build_ps_edges_cmdw
from des.edge_builders import build_ss_edges_cmdw
from des.graph_utils import (
    calibrate_pool,
    compute_meta_des_sample_feats,
    compute_meta_labels,
    encode_eicu_features,
    encode_with_graph_encoder,
    generate_prototypes,
    init_graph_encoder,
    preprocess_for_encoder,
    project_to_DS,
)
from des.viz import save_graph_summaries


def build_train_eval_graph(
    self,
    tr_ds: torch.Tensor,
    tr_preds: torch.Tensor,
    tr_meta_labels: torch.Tensor,
    y_tr: torch.Tensor,
    tr_feats: torch.Tensor,
    eval_ds: torch.Tensor,
    y_eval: torch.Tensor,
    eval_feats: torch.Tensor,
    eval_type: str = "val",
    prototypes: Optional[List[Dict[str, Any]]] = None,
) -> HeteroData:
    """
    Constructs a heterograph with sample and prototype node types.
    """

    def to_np(t):
        if isinstance(t, torch.Tensor):
            return t.cpu().numpy()
        return np.asarray(t)

    def to_cpu(t):
        if isinstance(t, torch.Tensor):
            return t.cpu()
        return torch.from_numpy(t)

    local_classes = torch.unique(y_tr.cpu())
    valid_protos: List[Dict[str, Any]] = []
    if prototypes is not None:
        for proto in prototypes:
            proto_cls = proto["class"]
            if isinstance(proto_cls, torch.Tensor):
                proto_cls = proto_cls.item()
            if proto_cls in local_classes:
                valid_protos.append(proto)

    tr_ds_np = to_np(tr_ds)
    eval_ds_np = to_np(eval_ds)
    y_tr_np = to_np(y_tr)
    y_eval_np = to_np(y_eval)

    n_train_real = int(tr_ds.shape[0])
    n_eval_real = int(eval_ds.shape[0])
    n_sample_total = n_train_real + n_eval_real

    sample_ds = np.vstack([tr_ds_np, eval_ds_np])
    sample_y = np.concatenate([y_tr_np, y_eval_np])

    ss_edge_index, ss_weights = build_ss_edges_cmdw(
        decision_matrix=sample_ds,
        label_vector=sample_y,
        source_indices=np.arange(n_train_real, dtype=np.int64),
        destination_indices=np.arange(n_sample_total, dtype=np.int64),
        k_per_class=int(getattr(self.args, "graph_k_per_class", 5)),
    )

    feat_mode = str(getattr(self.args, "graph_sample_node_feats", "ds")).lower()
    if feat_mode == "meta_feats":
        sample_feats = compute_meta_des_sample_feats(
            combined_ds=sample_ds,
            ss_edge_index=ss_edge_index,
            y_train=y_tr_np,
            tr_meta_labels=to_np(tr_meta_labels),
            n_train=n_train_real,
            n_total=n_sample_total,
            num_classifiers=len(self.global_clf_keys),
            num_classes=self.args.num_classes,
            K=int(getattr(self.args, "meta_des_K", 7)),
            Kp=int(getattr(self.args, "meta_des_Kp", 5)),
        )
        sample_feats = torch.from_numpy(sample_feats)
    else:
        sample_feats = torch.cat([to_cpu(tr_feats), to_cpu(eval_feats)], dim=0)

    use_protos_globally = bool(getattr(self.args, "proto_use", False))
    if len(valid_protos) > 0:
        p_feats = torch.cat([to_cpu(p["feats"]) for p in valid_protos], dim=0)
        p_ds = torch.cat([to_cpu(p["ds"]) for p in valid_protos], dim=0)
        p_y_list = []
        for proto in valid_protos:
            k = proto["feats"].shape[0]
            p_y_list.append(torch.full((k,), proto["class"], dtype=y_tr.dtype))
        p_y = torch.cat(p_y_list, dim=0)
    else:
        feat_dim = int(sample_feats.shape[1]) if sample_feats.numel() else 0
        ds_dim = int(tr_ds.shape[1]) if n_train_real > 0 else int(eval_ds.shape[1])
        p_feats = torch.empty((0, feat_dim), dtype=sample_feats.dtype)
        p_ds = torch.empty((0, ds_dim), dtype=tr_ds.dtype)
        p_y = torch.empty((0,), dtype=y_tr.dtype)
        if use_protos_globally:
            print(f"[FedDES][{self.role}] No valid prototypes found; building graph without prototype nodes.")

    n_proto = int(p_feats.shape[0])
    if n_proto > 0:
        ps_edge_index, ps_weights = build_ps_edges_cmdw(
            proto_decision_matrix=to_np(p_ds),
            proto_labels=to_np(p_y),
            sample_decision_matrix=sample_ds,
            sample_labels=sample_y,
            k_per_class=int(getattr(self.args, "graph_k_per_class", 5)),
        )
    else:
        ps_edge_index = np.zeros((2, 0), dtype=np.int64)
        ps_weights = np.zeros((0,), dtype=np.float32)

    data = HeteroData()
    data["sample"].x = sample_feats.float()
    data["sample"].y = torch.from_numpy(sample_y).long()

    idx = torch.arange(n_sample_total)
    data["sample"].train_mask = idx < n_train_real
    data["sample"][f"{eval_type}_mask"] = idx >= n_train_real

    data["prototype"].x = p_feats.float()
    data["prototype"].y = p_y.long()

    def to_tensor(x, dtype_fn):
        if isinstance(x, torch.Tensor):
            return dtype_fn(x)
        return dtype_fn(torch.from_numpy(x))

    data["sample", "ss", "sample"].edge_index = to_tensor(ss_edge_index, lambda t: t.long())
    data["sample", "ss", "sample"].edge_attr = to_tensor(ss_weights, lambda t: t.float())

    data["prototype", "ps", "sample"].edge_index = to_tensor(ps_edge_index, lambda t: t.long())
    data["prototype", "ps", "sample"].edge_attr = to_tensor(ps_weights, lambda t: t.float())

    save_graph_summaries(
        self,
        client_role=self.role,
        args=getattr(self, "args", None),
        n_train=n_train_real,
        n_eval=n_eval_real,
        n_classifiers=0,
        classifier_names=[],
        sample_labels=sample_y,
        ss_edge_index=ss_edge_index,
        ss_edge_attr=ss_weights,
        cs_edge_index=None,
        cs_edge_attr=None,
        cc_edge_index=None,
        cc_edge_attr=None,
        bins=50,
        eval_kind=eval_type,
    )

    return data


__all__ = [
    "build_train_eval_graph",
    "calibrate_pool",
    "compute_meta_labels",
    "project_to_DS",
    "generate_prototypes",
    "compute_meta_des_sample_feats",
    "encode_with_graph_encoder",
    "encode_eicu_features",
    "preprocess_for_encoder",
    "init_graph_encoder",
]

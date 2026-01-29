
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torchvision import models
# Import the new hybrid builder alongside existing ones
from des.edge_builders import build_cs_edges, build_cc_edges, build_cs_edges_v2, build_ss_edges_cmdw, build_cs_edges_hybrid
from des.viz import save_graph_summaries
from des.base_clf_utils import fit_calibrator, process_batch
from des.dataset_stats import load_client_label_counts
from torch_geometric.data import HeteroData
from probmetrics.calibrators import get_calibrator
from probmetrics.distributions import CategoricalLogits
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding


def _resolve_attr(module: torch.nn.Module, attr: str):
    if hasattr(module, attr):
        return getattr(module, attr)
    if hasattr(module, "module") and hasattr(module.module, attr):
        return getattr(module.module, attr)
    return None


def _forward_with_embedding(model: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    base = _resolve_attr(model, "base")
    head = _resolve_attr(model, "head")
    if base is not None and head is not None:
        rep = base(x)
        if isinstance(rep, tuple):
            rep = rep[0]
        rep = rep.contiguous()
        logits = head(rep)
        rep_flat = rep.view(rep.size(0), -1)
        return logits, rep_flat

    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = logits.contiguous()
    rep_flat = logits.view(logits.size(0), -1)
    return logits, rep_flat


def compute_meta_labels(probs, preds, labels, min_positive=5):
    mask = (preds == labels.unsqueeze(1)).clone()

    with torch.no_grad():
        true_cls_probs = probs[torch.arange(probs.size(0)), :, labels]  # [N, M]
        for i in range(probs.size(0)):
            needed = min_positive - int(mask[i].sum().item())
            if needed <= 0:
                continue
            for idx in torch.argsort(true_cls_probs[i], descending=True):
                if mask[i, idx]:
                    continue
                mask[i, idx] = True
                needed -= 1
                if needed <= 0:
                    break
    return mask.to(torch.uint8)


def get_diversity_embeddings(preds: np.ndarray, n_components: int = 8) -> np.ndarray:
    """
    Computes spectral embeddings based on classifier error correlation.
    Args:
        preds: [N_samples, M_classifiers] Binary array (1=Correct, 0=Wrong)
        n_components: Dimension of the output embedding.
    Returns:
        embeddings: [M_classifiers, n_components]
    """
    M = preds.shape[1]
    if M < n_components + 1:
        return np.zeros((M, n_components), dtype=np.float32)

    preds_std = preds - preds.mean(axis=0)
    std = preds.std(axis=0)
    std[std == 0] = 1.0
    preds_std = preds_std / std

    correlation_matrix = (preds_std.T @ preds_std) / preds.shape[0]
    affinity_matrix = (correlation_matrix + 1.0) / 2.0
    np.fill_diagonal(affinity_matrix, 0.0)

    try:
        embedder = SpectralEmbedding(n_components=n_components, affinity="precomputed")
        embeddings = embedder.fit_transform(affinity_matrix)
    except Exception as e:
        print(f"[Warning] Diversity embedding failed: {e}. Using zeros.")
        embeddings = np.zeros((M, n_components), dtype=np.float32)

    return embeddings.astype(np.float32)


def calibrate_pool(self, loader, classifier_pool):
    classifier_keys = self.global_clf_keys
    num_classifiers = len(classifier_keys)
    active_models = [
        classifier_pool[key].to(self.device).eval() for key in classifier_keys
    ]

    logits_all, labels_all = [], []
    batches = list(loader)

    with torch.no_grad():
        for batch in batches:
            x, y = process_batch(batch, self.device)
            per_model_logits = []
            for model in active_models:
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                per_model_logits.append(logits.cpu())
            logits_all.append(torch.stack(per_model_logits, dim=1))
            labels_all.append(y.cpu())

    for model in active_models:
        model.to("cpu").eval()

    if logits_all:
        logits = torch.cat(logits_all, dim=0)
        labels = torch.cat(labels_all, dim=0)
    else:
        logits = torch.empty((0, num_classifiers, 1), dtype=torch.float32)
        labels = torch.empty((0,), dtype=torch.long)

    temperatures: List[float] = []
    for i in range(num_classifiers):
        logits_slice = logits[:, i, :]
        if logits_slice.numel() == 0 or labels.numel() == 0:
            temperatures.append(1.0)
            continue
        T = fit_calibrator(logits_slice, labels)
        temperatures.append(T)

    setattr(self, "_pool_calibrators", temperatures)
    return temperatures


def project_to_DS(self, loader, classifier_pool, calibrate_probs = True):

    classifier_keys = self.global_clf_keys
    num_classifiers = len(classifier_keys)

    logits_all = []
    embeddings_all = []
    batches = list(loader)

    labels_all = []
    for batch in batches:
        _, y = batch
        y = y[0] if isinstance(y, (list, tuple)) else y
        labels_all.append(y.detach().cpu())

    with torch.no_grad():
        for key in classifier_keys:
            model = classifier_pool[key].to(self.device).eval()
            per_model_logits = []
            per_model_embeds = []
            for batch in batches:
                x, y = process_batch(batch, self.device)
                logits, embeds = _forward_with_embedding(model, x)
                per_model_embeds.append(embeds.cpu())
                per_model_logits.append(logits.cpu())
            model.to("cpu").eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logits_all.append(torch.cat(per_model_logits, dim=0))
            embeddings_all.append(torch.cat(per_model_embeds, dim=0))

    # Stack all models: [N, M, C]
    logits = torch.stack(logits_all, dim=1)
    labels = torch.cat(labels_all, dim=0)   # [N]
    embeddings = torch.stack(embeddings_all, dim=1)

    if calibrate_probs:
        # Choose calibration method (configurable; default to temp-scaling)
        calib_method = getattr(self.args, "pool_calib_method", "ts-mix")

        # ----------------------------------------------------
        # 1) Fit calibrators once (on the "train" split) and cache
        # ----------------------------------------------------
        pool_calibrators = getattr(self, "_pool_calibrators", None)
        if pool_calibrators is not None and len(pool_calibrators) != num_classifiers:
            pool_calibrators = None
            setattr(self, "_pool_calibrators", None)
        if pool_calibrators is None:
            pool_calibrators = []
            for i in range(num_classifiers):
                logits_slice = logits[:, i, :]  # [N, C]

                if calib_method == "temp-scaling":
                    T = fit_calibrator(logits_slice, labels)
                    pool_calibrators.append({"type": "temp-scaling", "T": float(T)})
                elif calib_method == "ts-mix":
                    calib = get_calibrator("ts-mix")
                    dist_fit = CategoricalLogits(logits_slice)
                    calib.fit_torch(dist_fit, labels)
                    pool_calibrators.append({"type": "ts-mix", "calib": calib})
                else:
                    raise ValueError(f"Unknown pool_calib_method: {calib_method!r}")
            self._pool_calibrators = pool_calibrators

        # ----------------------------------------------------
        # 2) Apply calibrators to get calibrated probabilities
        # ----------------------------------------------------
        probs_per_model = []
        for i in range(num_classifiers):
            logits_slice = logits[:, i, :]      # [N, C]
            cal = self._pool_calibrators[i]

            if cal["type"] == "temp-scaling":
                T = cal["T"]
                probs_i = torch.softmax(logits_slice / T, dim=1)
            elif cal["type"] == "ts-mix":
                dist_apply = CategoricalLogits(logits_slice)
                result = cal["calib"].predict_proba_torch(dist_apply)
                probs_i = result.get_probs()     # [N, C]

            probs_per_model.append(probs_i.unsqueeze(1))  # [N,1,C]

        probs = torch.cat(probs_per_model, dim=1)         # [N,M,C]
    else:
        probs = torch.softmax(logits, dim=2)

    preds = probs.argmax(dim=2)                 # [N, M]
    meta_labels = compute_meta_labels(
        probs, preds, labels,
        min_positive=int(getattr(self.args, "graph_meta_min_pos", 3)),
    )

    probs_flat = probs.reshape(probs.size(0), -1)

    feat_mode = str(getattr(self.args, "graph_sample_node_feats", "ds")).lower()

    if feat_mode == "ds":
        features = probs_flat
    elif feat_mode == "embedding_mean":
        features = embeddings.mean(dim=1)
    elif feat_mode == "embedding_concat":
        features = embeddings.reshape(embeddings.size(0), -1)
    elif feat_mode == "encoder":
        dataset_name = str(getattr(self.args, "data", getattr(self.args, "dataset", ""))).lower()
        if "eicu" in dataset_name:
            features = encode_eicu_features(self, batches)
        else:
            features = encode_with_graph_encoder(self, batches)
    elif feat_mode == "hybrid":
        dataset_name = str(getattr(self.args, "data", getattr(self.args, "dataset", ""))).lower()
        if "eicu" in dataset_name:
            enc_feats = encode_eicu_features(self, batches)
        else:
            enc_feats = encode_with_graph_encoder(self, batches)
        if enc_feats.dim() > 2:
            enc_feats = enc_feats.view(enc_feats.size(0), -1)
        enc_feats = F.normalize(enc_feats.float(), p=2, dim=1)
        features = torch.cat([enc_feats, probs_flat], dim=1)
    elif feat_mode == "meta_feats":
        # Placeholder; actual computation happens in build_train_eval_graph
        features = probs_flat
    else:
        features = probs_flat

    return probs_flat, preds, labels, meta_labels, features


from sklearn.cluster import KMeans

# You can name the first arg 'self' if you want, but 'client' is clearer 
# since this is a standalone function, not a class method.
def generate_prototypes(client, ds, feats, meta, y):
    """
    Generates clustered prototypes.
    Args:
        client: The Client instance (provides .args and .id)
        ds, feats, meta, y: Tensor data
    """
    # 1. Extract Config from Client
    # Uses defaults if args are missing from client.args
    max_k = int(getattr(client.args, "proto_max_k", 5))
    min_samples = int(getattr(client.args, "proto_min_samples", 5))
    
    # 2. Extract Seed from Client ID
    # Ensure client.id is usable as a seed
    try:
        random_seed = int(client.id)
    except (ValueError, TypeError):
        # Fallback if id is a string like 'client0'
        import hashlib
        random_seed = int(hashlib.sha256(str(client.id).encode('utf-8')).hexdigest(), 16) % (2**32)

    ds, feats, meta, y = ds.cpu(), feats.cpu(), meta.cpu(), y.cpu()
    prototypes = []
    unique_classes = torch.unique(y)

    for c in unique_classes:
        mask = (y == c)
        c_feats = feats[mask]
        c_ds = ds[mask]
        c_meta = meta[mask]
        n_samples = c_feats.shape[0]

        # Adaptive K calculation
        k = max(1, min(max_k, n_samples // min_samples))
        
        if k == 1:
            proto_feat = c_feats.mean(dim=0, keepdim=True)
            proto_ds = c_ds.mean(dim=0, keepdim=True)
            proto_meta = c_meta.mean(dim=0, keepdim=True)
        else:
            # Use client-specific seed
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_seed)
            cluster_ids = kmeans.fit_predict(c_feats.numpy())
            cluster_ids = torch.from_numpy(cluster_ids)
            
            proto_feat, proto_ds, proto_meta = [], [], []
            for i in range(k):
                c_mask = (cluster_ids == i)
                proto_feat.append(c_feats[c_mask].mean(dim=0))
                proto_ds.append(c_ds[c_mask].mean(dim=0))
                proto_meta.append(c_meta[c_mask].mean(dim=0))
            
            proto_feat = torch.stack(proto_feat)
            proto_ds = torch.stack(proto_ds)
            proto_meta = torch.stack(proto_meta)

        prototypes.append({
            "class": c.item(),
            "feats": proto_feat, 
            "ds": proto_ds,      
            "meta": proto_meta   
        })

    return prototypes


# --- META-DES-style meta-feature computation ---
def compute_meta_des_sample_feats(
    *,
    combined_ds: np.ndarray,           
    ss_edge_index: np.ndarray,         
    y_train: np.ndarray,               
    tr_meta_labels: np.ndarray,        
    n_train: int,
    n_total: int,
    num_classifiers: int,
    num_classes: int,
    K: int = 7,
    Kp: int = 5,
    eps: float = 1e-8,
) -> np.ndarray:
    # [Unchanged implementation of compute_meta_des_sample_feats]
    combined_ds = np.asarray(combined_ds, dtype=np.float32)
    ss_edge_index = np.asarray(ss_edge_index, dtype=np.int64)
    y_train = np.asarray(y_train, dtype=np.int64)
    tr_meta_labels = np.asarray(tr_meta_labels, dtype=np.uint8)

    M = int(num_classifiers)
    C = int(num_classes)

    probs_all = combined_ds.reshape(n_total, M, C).astype(np.float32)   
    preds_all = probs_all.argmax(axis=2).astype(np.int64)               

    train_probs = probs_all[:n_train]                                   
    train_preds = preds_all[:n_train]                                   

    if ss_edge_index.size == 0:
        neighbors_by_dest: Dict[int, np.ndarray] = {}
    else:
        src_ids = ss_edge_index[0]
        dst_ids = ss_edge_index[1]
        buckets: Dict[int, List[int]] = {}
        for s, d in zip(src_ids.tolist(), dst_ids.tolist()):
            if s >= n_train:
                continue
            buckets.setdefault(int(d), []).append(int(s))
        neighbors_by_dest = {d: np.asarray(v, dtype=np.int64) for d, v in buckets.items()}

    if n_train > 0:
        uniq_profiles, inv = np.unique(train_preds, axis=0, return_inverse=True) 
        P = int(uniq_profiles.shape[0])
        prof_to_indices: List[np.ndarray] = [None] * P
        tmp_lists: List[List[int]] = [[] for _ in range(P)]
        for idx, pid in enumerate(inv.tolist()):
            tmp_lists[int(pid)].append(int(idx))
        for pid in range(P):
            prof_to_indices[pid] = np.asarray(tmp_lists[pid], dtype=np.int64)
    else:
        uniq_profiles = np.zeros((0, M), dtype=np.int64)
        prof_to_indices = []

    feats = np.zeros((n_total, M * (2 * K + Kp + 2)), dtype=np.float32)
    clf_idx = np.arange(M, dtype=np.int64)

    for j in range(n_total):
        neigh = neighbors_by_dest.get(j, None)
        if neigh is None or neigh.size == 0:
            nn_idx = np.empty((0,), dtype=np.int64)
        else:
            if j < n_train:
                neigh = neigh[neigh != j]
            q_flat = combined_ds[j] 
            d = np.sum(np.abs(combined_ds[neigh] - q_flat[None, :]), axis=1)
            order = np.argsort(d, kind="stable")
            neigh_sorted = neigh[order]
            nn_idx = neigh_sorted[: min(int(K), int(neigh_sorted.size))]

        k_eff = int(nn_idx.size)

        if k_eff > 0:
            f1 = tr_meta_labels[nn_idx, :].astype(np.float32).T
        else:
            f1 = np.zeros((M, 0), dtype=np.float32)

        q_probs = probs_all[j]                                          
        w_l = q_probs.argmax(axis=1).astype(np.int64)                   

        if k_eff > 0:
            f2 = train_probs[nn_idx[:, None], clf_idx[None, :], w_l[None, :]].astype(np.float32).T
        else:
            f2 = np.zeros((M, 0), dtype=np.float32)

        if k_eff > 0:
            f3 = f1.mean(axis=1, keepdims=True).astype(np.float32)      
        else:
            f3 = np.zeros((M, 1), dtype=np.float32)

        kp_eff = min(int(Kp), int(n_train))
        if kp_eff > 0 and uniq_profiles.shape[0] > 0:
            op_q = preds_all[j].astype(np.int64)                        
            d_prof = (uniq_profiles != op_q[None, :]).mean(axis=1).astype(np.float32)  
            kp_prof = min(kp_eff, int(d_prof.size))
            sel_pids = np.argpartition(d_prof, kp_prof - 1)[:kp_prof]
            sel_pids = sel_pids[np.argsort(d_prof[sel_pids])]

            op_neighbors: List[int] = []
            for pid in sel_pids.tolist():
                ids = prof_to_indices[int(pid)]
                if ids.size == 0: continue
                op_neighbors.extend(ids.tolist())
                if len(op_neighbors) >= kp_eff: break

            if len(op_neighbors) > kp_eff:
                op_neighbors = op_neighbors[:kp_eff]

            op_idx = np.asarray(op_neighbors, dtype=np.int64)
            if j < n_train:
                op_idx = op_idx[op_idx != j]
                if op_idx.size > kp_eff:
                    op_idx = op_idx[:kp_eff]

            f4 = tr_meta_labels[op_idx, :].astype(np.float32).T if op_idx.size else np.zeros((M, 0), dtype=np.float32)
        else:
            f4 = np.zeros((M, 0), dtype=np.float32)

        if C >= 2:
            part = np.partition(-q_probs, kth=1, axis=1)
            top1 = -part[:, 0]
            top2 = -part[:, 1]
            margin = (top1 - top2)
        else:
            margin = np.ones((M,), dtype=np.float32)
        f5 = margin.reshape(M, 1).astype(np.float32)

        if k_eff < K:
            pad = K - k_eff
            f1 = np.pad(f1, ((0, 0), (0, pad)), mode="constant")
            f2 = np.pad(f2, ((0, 0), (0, pad)), mode="constant")

        if f4.shape[1] < Kp:
            padp = Kp - f4.shape[1]
            f4 = np.pad(f4, ((0, 0), (0, padp)), mode="constant")

        per_clf = np.concatenate([f1, f2, f3, f4, f5], axis=1)           
        feats[j] = per_clf.reshape(-1).astype(np.float32)

    return feats


def encode_with_graph_encoder(self, batches):
    encoder = getattr(self, "_graph_encoder", None)
    if encoder is None:
        encoder = init_graph_encoder()
        self._graph_encoder = encoder.to(self.device)
    encoder.eval()

    encoded_feats = []
    with torch.no_grad():
        for batch in batches:
            x, _ = process_batch(batch, self.device)
            x = preprocess_for_encoder(x)
            feat = encoder(x)
            feat = feat.reshape(feat.size(0), -1)
            encoded_feats.append(feat.cpu())

    if not encoded_feats:
        return torch.empty(0, 0)
    return torch.cat(encoded_feats, dim=0)


def encode_eicu_features(self, batches):
    encoded_feats = []
    with torch.no_grad():
        for batch in batches:
            x, _ = process_batch(batch, self.device)
            if x.dim() == 3:
                feat = x.mean(dim=1)
            else:
                feat = x.reshape(x.size(0), -1)
            encoded_feats.append(feat.cpu())

    if not encoded_feats:
        return torch.empty(0, 0)
    return torch.cat(encoded_feats, dim=0)


def preprocess_for_encoder(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.dim() == 4 and x.size(2) != 224:
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, -1, 1, 1)
    return (x - mean) / std


def init_graph_encoder(name: str = "resnet18"):
    if name != "resnet18":
        raise NotImplementedError(f"Encoder '{name}' is not supported yet.")
    try:
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        base = models.resnet18(pretrained=True)
    encoder = torch.nn.Sequential(*list(base.children())[:-1])
    return encoder


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
    prototypes: Optional[List[Dict[str, Any]]] = None
) -> HeteroData:
    """
    Constructs the Heterogeneous Graph for FedDES (Train + Eval).
    
    Arguments match the unpacked call in clientdes.py.
    """
    
    # Helper to convert between Torch/Numpy/Device
    def to_np(t):
        if isinstance(t, torch.Tensor): return t.cpu().numpy()
        return np.asarray(t)
    def to_cpu(t):
        if isinstance(t, torch.Tensor): return t.cpu()
        return torch.from_numpy(t)

    # 1. IDENTIFY LOCAL CLASSES & FILTER PROTOTYPES
    # ---------------------------------------------
    # We only want to inject prototypes for classes that this client actually 
    # has in its local training set (to anchor them).
    local_classes = torch.unique(y_tr.cpu())
    
    valid_protos = []
    if prototypes is not None:
        for p in prototypes:
            # p["class"] might be an int or scalar tensor
            p_cls = p["class"]
            if isinstance(p_cls, torch.Tensor): p_cls = p_cls.item()
            
            if p_cls in local_classes:
                valid_protos.append(p)

    # 2. PREPARE DATA STACKS
    # ---------------------------------------------
    # Structure: [Real Train, Ghost Protos (Optional), Real Eval]
    
    # Base Numpy Arrays for SS Edges / Stats
    tr_ds_np = to_np(tr_ds)
    eval_ds_np = to_np(eval_ds)
    y_tr_np = to_np(y_tr)
    y_eval_np = to_np(y_eval)
    
    # Counts
    n_train_real = tr_ds.shape[0]
    n_eval_real = eval_ds.shape[0]
    n_proto = 0
    
    # Check if we should use prototypes globally (Flag check)
    use_protos_globally = getattr(self.args, "proto_use", False)

    # --- CASE A: Valid Prototypes Found (Inject them) ---
    if len(valid_protos) > 0:
        print(f"[FedDES][{self.role}] Injecting {len(valid_protos)} valid prototypes as ghost nodes.")
        
        # Unpack Prototypes
        p_feats = torch.cat([to_cpu(p["feats"]) for p in valid_protos], dim=0)
        p_ds = torch.cat([to_cpu(p["ds"]) for p in valid_protos], dim=0)
        p_meta = torch.cat([to_cpu(p["meta"]) for p in valid_protos], dim=0) # Soft labels
        
        # Create 'y' for protos
        p_y_list = []
        for p in valid_protos:
            k = p["feats"].shape[0]
            p_y_list.append(torch.full((k,), p["class"], dtype=y_tr.dtype))
        p_y = torch.cat(p_y_list, dim=0)

        n_proto = p_feats.shape[0]

        # Stack DS & Y
        p_ds_np = to_np(p_ds)
        combined_ds = np.vstack([tr_ds_np, p_ds_np, eval_ds_np])
        combined_y = np.concatenate([y_tr_np, to_np(p_y), y_eval_np])
        
        # Stack Features
        combined_feats = torch.cat([to_cpu(tr_feats), p_feats, to_cpu(eval_feats)], dim=0)

        # Stack Meta Labels (Train=Binary, Proto=Soft, Eval=Dummy)
        num_classifiers = tr_meta_labels.shape[1]
        eval_meta_dummy = torch.zeros((n_eval_real, num_classifiers), dtype=torch.float32)
        
        combined_meta = torch.cat([
            to_cpu(tr_meta_labels).float(),
            p_meta.float(),
            eval_meta_dummy
        ], dim=0)

        # Append Indicator Feature: [Real=0, Proto=1]
        indicator = torch.cat([
            torch.zeros(n_train_real, 1),
            torch.ones(n_proto, 1),
            torch.zeros(n_eval_real, 1)
        ], dim=0)
        combined_feats = torch.cat([combined_feats, indicator], dim=1)

        # Create Hybrid Mask for Edge Builder
        is_proto_mask = torch.cat([
            torch.zeros(n_train_real, dtype=torch.bool),
            torch.ones(n_proto, dtype=torch.bool),
            torch.zeros(n_eval_real, dtype=torch.bool)
        ], dim=0)

    # --- CASE B: Flag is True, but No Valid Protos (Fallback) ---
    elif use_protos_globally:
        # We MUST still append the '0' indicator to match dimensions of peers
        # who might have found prototypes.
        
        combined_ds = np.vstack([tr_ds_np, eval_ds_np])
        combined_y = np.concatenate([y_tr_np, y_eval_np])
        
        combined_feats = torch.cat([to_cpu(tr_feats), to_cpu(eval_feats)], dim=0)
        
        # Append Zeros (Real Data)
        indicator = torch.zeros(combined_feats.shape[0], 1)
        combined_feats = torch.cat([combined_feats, indicator], dim=1) # Dim D+1
        
        # Standard Meta Labels
        num_classifiers = tr_meta_labels.shape[1]
        eval_meta_dummy = torch.zeros((n_eval_real, num_classifiers), dtype=torch.float32)
        combined_meta = torch.cat([to_cpu(tr_meta_labels).float(), eval_meta_dummy], dim=0)
        
        is_proto_mask = torch.zeros(combined_feats.shape[0], dtype=torch.bool)

    # --- CASE C: Legacy / Standard Mode (No Flag) ---
    else:
        combined_ds = np.vstack([tr_ds_np, eval_ds_np])
        combined_y = np.concatenate([y_tr_np, y_eval_np])
        combined_feats = torch.cat([to_cpu(tr_feats), to_cpu(eval_feats)], dim=0)
        
        num_classifiers = tr_meta_labels.shape[1]
        eval_meta_dummy = torch.zeros((n_eval_real, num_classifiers), dtype=torch.float32)
        combined_meta = torch.cat([to_cpu(tr_meta_labels).float(), eval_meta_dummy], dim=0)
        
        is_proto_mask = torch.zeros(combined_feats.shape[0], dtype=torch.bool)

    # 3. BUILD EDGES
    # ---------------------------------------------
    n_train_total = n_train_real + n_proto
    n_total = combined_ds.shape[0]

    # A. Sample-Sample Edges (CMDW)
    # Protos are treated as valid "Sources" (part of training set)
    ss_edge_index, ss_weights = build_ss_edges_cmdw(
        decision_matrix=combined_ds,
        label_vector=combined_y,
        source_indices=np.arange(n_train_total, dtype=np.int64), # Train + Proto
        destination_indices=np.arange(n_total, dtype=np.int64),  # All nodes
        k_per_class=int(getattr(self.args, "graph_k_per_class", 5)),
    )
    
    # Prepare Adjacency List for Hybrid Builder
    from collections import defaultdict
    adj_list = defaultdict(list)
    src_np = ss_edge_index[0]
    dst_np = ss_edge_index[1]
    for s, d in zip(src_np, dst_np):
        adj_list[d].append(s)
    neighbor_indices_list = [adj_list[i] for i in range(n_total)]

    # B. Classifier-Sample Edges (Hybrid)
    # cs_edge_index, cs_weights = build_cs_edges_hybrid(
    #     neighbor_indices=neighbor_indices_list,
    #     meta_labels=combined_meta, 
    #     is_proto_mask=is_proto_mask,
    #     threshold=getattr(self.args, "cs_threshold", 0.5)
    # )
    cs_mode = str(getattr(self.args, "graph_cs_mode", "balanced_acc:logloss")).lower()
    if ":" in cs_mode:
        score_mode, tie_break_mode = (part.strip() for part in cs_mode.split(":", 1))
    else:
        score_mode, tie_break_mode = cs_mode.strip(), "logloss"
    if tie_break_mode in {"none", ""}:
        tie_break_mode = None

    cs_topk = int(getattr(self.args, "graph_cs_topk", 3))
    if cs_topk == 0:
        cs_topk = max(1, int(0.25 * num_classifiers))

    cs_edge_index, cs_weights = build_cs_edges_v2(
        tr_meta_labels=combined_meta.numpy(), # Pass as numpy
        decision_all=combined_ds,
        y_train=combined_y[:n_train_total],   # Train+Proto labels
        ss_edge_index=ss_edge_index,
        ss_edge_attr=ss_weights,
        n_train=n_train_total,
        n_total=n_total,
        num_classes=self.args.num_classes,
        top_k=cs_topk,
        score_mode=score_mode,
        tie_break_mode=tie_break_mode,
        
        # --- THE NEW ARGS ---
        is_proto_mask=is_proto_mask.numpy(),
        proto_self_weight=1.0 # Give the prototype a strong vote for itself? Or 1.0?
    )


    # C. Classifier-Classifier Edges (Diversity)
    # Use real train data only for stability
    cc_edge_index, cc_weights = build_cc_edges(
         diversity_matrix=getattr(self.args, "diversity_matrix", None), 
         threshold=getattr(self.args, "cc_threshold", 0.0)
    ) if hasattr(self.args, "diversity_matrix") else (torch.empty((2, 0)), torch.empty(0))

    if cc_edge_index.numel() == 0:
         cc_edge_index = torch.empty((2, 0), dtype=torch.long)
         cc_weights = torch.empty((0,), dtype=torch.float)

    # 4. CONSTRUCT HETERODATA
    # ---------------------------------------------
    # Calculate classifier node features (stats based on real train only).
    tr_meta_np = to_np(tr_meta_labels).astype(np.float32)
    probs = tr_ds_np.reshape(-1, num_classifiers, self.args.num_classes)

    present_only = True
    if present_only:
        class_ids = np.unique(y_tr_np)
    else:
        class_ids = np.arange(self.args.num_classes)

    class_masks = [y_tr_np == cls for cls in class_ids]
    class_counts = np.array([mask.sum() for mask in class_masks], dtype=np.float32)
    has_support = class_counts > 0
    safe_counts = np.where(has_support, class_counts, 1.0)[:, None]

    def masked_mean(values: np.ndarray) -> np.ndarray:
        out = []
        for mask, count in zip(class_masks, class_counts):
            if count > 0:
                out.append(values[mask].mean(axis=0))
            else:
                out.append(np.zeros(values.shape[1:], dtype=np.float32))
        return np.asarray(out, dtype=np.float32)

    def agg_over_supported(matrix: np.ndarray) -> np.ndarray:
        supported = matrix[has_support]
        if supported.size == 0:
            return np.zeros((1, matrix.shape[1]), dtype=np.float32)
        return supported.mean(axis=0, keepdims=True).astype(np.float32)

    per_class_hard_recall = masked_mean(tr_meta_np)
    true_class_probs = probs[np.arange(probs.shape[0]), :, y_tr_np][:, :, None]
    per_class_true_prob = masked_mean(true_class_probs.squeeze(-1))
    # --- NEW: Calculate Average Margin per Class ---
    probs_temp = probs.copy()
    rows = np.arange(probs.shape[0])
    probs_temp[rows[:, None], :, y_tr_np[:, None]] = -1.0
    max_rival_probs = probs_temp.max(axis=2)
    margins = true_class_probs.squeeze(-1) - max_rival_probs
    per_class_avg_margin = masked_mean(margins)

    # --- NEW: Calculate Diversity Embeddings (optional) ---
    use_div_emb = bool(getattr(self.args, "graph_clf_div_emb", False))
    if use_div_emb:
        div_feats = get_diversity_embeddings(tr_meta_np, n_components=8)

    se = np.sqrt(per_class_hard_recall * (1.0 - per_class_hard_recall) / safe_counts)
    se = np.where(has_support[:, None], se, 0.0).astype(np.float32)

    overall_accuracy = tr_meta_np.mean(axis=0, keepdims=True).astype(np.float32)
    mean_true_prob = agg_over_supported(per_class_true_prob)
    balanced_accuracy = agg_over_supported(per_class_hard_recall)

    clf_x_stats = np.concatenate(
        [
            per_class_hard_recall.T,
            per_class_true_prob.T,
            per_class_avg_margin.T,
            se.T,
            overall_accuracy.T,
            mean_true_prob.T,
            balanced_accuracy.T,
        ],
        axis=1,
    ).astype(np.float32)

    # Append per-class home-client data distribution (sums to 1 per classifier).
    dataset_name = getattr(self.args, "dataset", "")
    label_counts_map = load_client_label_counts(dataset_name)
    dist_feats = np.zeros((num_classifiers, self.args.num_classes), dtype=np.float32)
    for idx in range(num_classifiers):
        if idx < len(self.global_clf_keys):
            key = self.global_clf_keys[idx]
            home_role = key[0] if isinstance(key, (list, tuple)) and key else str(key)
        else:
            home_role = f"Client_{idx}"
        home_counts = label_counts_map.get(home_role, {})
        total = float(sum(home_counts.values()))
        if total > 0:
            for cls, cnt in home_counts.items():
                if 0 <= int(cls) < self.args.num_classes:
                    dist_feats[idx, int(cls)] = float(cnt) / total

    clf_parts = [clf_x_stats, dist_feats]
    if use_div_emb:
        clf_parts.append(div_feats)
    clf_x = torch.from_numpy(np.concatenate(clf_parts, axis=1)).float()

    data = HeteroData()
    data['sample'].x = combined_feats.float()
    data['sample'].y = torch.from_numpy(combined_y).long()
    
    # Masks
    idx = torch.arange(n_total)
    data['sample'].train_mask = idx < n_train_total # Real Train + Protos
    data['sample'][f"{eval_type}_mask"] = idx >= n_train_total # Real Eval

    data['classifier'].x = clf_x
    data['classifier'].num_nodes = num_classifiers
    data['classifier'].clf_keys = list(self.global_clf_keys)

# ... inside build_train_eval_graph ...

    # --- Helper to handle mixed Numpy/Tensor outputs safely ---
    def to_tensor(x, dtype_fn):
        if isinstance(x, torch.Tensor):
            return dtype_fn(x)
        return dtype_fn(torch.from_numpy(x))

    # Edges
    # SS edges usually return Numpy
    data['sample', 'ss', 'sample'].edge_index = to_tensor(ss_edge_index, lambda t: t.long())
    data['sample', 'ss', 'sample'].edge_attr = to_tensor(ss_weights, lambda t: t.float())
    
    # CS edges: Handles both v2 (Numpy) and Hybrid (Tensor) returns
    data['classifier', 'cs', 'sample'].edge_index = to_tensor(cs_edge_index, lambda t: t.long())
    data['classifier', 'cs', 'sample'].edge_attr = to_tensor(cs_weights, lambda t: t.float())
    
    # CC edges: Handles build_cc_edges (Numpy) and fallback torch.empty (Tensor)
    data['classifier', 'cc', 'classifier'].edge_index = to_tensor(cc_edge_index, lambda t: t.long())
    data['classifier', 'cc', 'classifier'].edge_attr = to_tensor(cc_weights, lambda t: t.float())

    # 5. LOGGING
    # ...


    # 5. LOGGING
    # ---------------------------------------------
    # This visualization uses n_train=n_train_total to correctly visualize 
    # prototypes as part of the "Training" manifold distribution.
    save_graph_summaries(self,
        client_role=self.role,
        args=getattr(self, "args", None),
        n_train=n_train_total, 
        n_eval=n_eval_real,
        n_classifiers=num_classifiers,
        classifier_names=list(self.global_clf_keys),
        sample_labels=combined_y,
        ss_edge_index=ss_edge_index, ss_edge_attr=ss_weights,
        cs_edge_index=cs_edge_index, cs_edge_attr=cs_weights,
        cc_edge_index=cc_edge_index, cc_edge_attr=cc_weights,
        bins=50,
        eval_kind=eval_type,
    )
    
    warn_disconnected_classifier_cc(data, eval_type, getattr(self, "role", None))

    return data

def warn_disconnected_classifier_cc(
    graph: HeteroData, graph_label: str, client_role: str | None = None
):
    """
    Print classifiers that only connect through cc edges with no cs links to samples.
    """
    cc_rel = ("classifier", "cc", "classifier")
    if cc_rel not in graph.edge_index_dict:
        return

    cc_edge_index = graph[cc_rel].edge_index
    if cc_edge_index.numel() == 0:
        return

    num_classifiers = int(
        getattr(graph["classifier"], "num_nodes", graph["classifier"].x.size(0))
    )
    neighbors = [[] for _ in range(num_classifiers)]
    for src, dst in cc_edge_index.t().tolist():
        neighbors[src].append(dst)
        neighbors[dst].append(src)

    cs_rel = ("classifier", "cs", "sample")
    cs_sources = set()
    if cs_rel in graph.edge_index_dict:
        cs_edge_index = graph[cs_rel].edge_index
        if cs_edge_index.numel():
            cs_sources = set(cs_edge_index[0].tolist())

    visited = [False] * num_classifiers
    disconnected_components: list[list[int]] = []

    for node_idx in range(num_classifiers):
        if visited[node_idx] or not neighbors[node_idx]:
            continue
        stack = [node_idx]
        component = []
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for neighbor in neighbors[node]:
                if not visited[neighbor]:
                    stack.append(neighbor)
        if component and not any(member in cs_sources for member in component):
            disconnected_components.append(component)

    if not disconnected_components:
        return

    clf_names = getattr(graph["classifier"], "clf_keys", None)
    if not clf_names or len(clf_names) != num_classifiers:
        clf_names = [f"classifier[{i}]" for i in range(num_classifiers)]

    descriptions = []
    for component in disconnected_components:
        names = ", ".join(clf_names[idx] for idx in component)
        descriptions.append(f"[{len(component)}] {names}")

    role_label = f"Client {client_role}" if client_role else "Client"
    print(
        f"[FedDES][{role_label}] {graph_label} graph has "
        f"{len(disconnected_components)} classifier-only cc component(s): "
        f"{'; '.join(descriptions)}"
    )



# from __future__ import annotations

# from typing import Any, Dict, List, Optional, Tuple
# import torch
# import numpy as np
# import torch.nn.functional as F
# from pathlib import Path
# from torchvision import models
# from des.edge_builders import build_cs_edges, build_cc_edges, build_cs_edges_v2, build_ss_edges_cmdw, build_cs_edges_hybrid
# from des.viz import save_graph_summaries
# from des.base_clf_utils import fit_calibrator, process_batch
# from torch_geometric.data import HeteroData
# from probmetrics.calibrators import get_calibrator
# from probmetrics.distributions import CategoricalLogits


# def _resolve_attr(module: torch.nn.Module, attr: str):
#     if hasattr(module, attr):
#         return getattr(module, attr)
#     if hasattr(module, "module") and hasattr(module.module, attr):
#         return getattr(module.module, attr)
#     return None


# def _forward_with_embedding(model: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     base = _resolve_attr(model, "base")
#     head = _resolve_attr(model, "head")
#     if base is not None and head is not None:
#         rep = base(x)
#         if isinstance(rep, tuple):
#             rep = rep[0]
#         rep = rep.contiguous()
#         logits = head(rep)
#         rep_flat = rep.view(rep.size(0), -1)
#         return logits, rep_flat

#     logits = model(x)
#     if isinstance(logits, tuple):
#         logits = logits[0]
#     logits = logits.contiguous()
#     rep_flat = logits.view(logits.size(0), -1)
#     return logits, rep_flat


# def compute_meta_labels(probs, preds, labels, min_positive=5):
#     mask = (preds == labels.unsqueeze(1)).clone()

#     with torch.no_grad():
#         true_cls_probs = probs[torch.arange(probs.size(0)), :, labels]  # [N, M]
#         for i in range(probs.size(0)):
#             needed = min_positive - int(mask[i].sum().item())
#             if needed <= 0:
#                 continue
#             for idx in torch.argsort(true_cls_probs[i], descending=True):
#                 if mask[i, idx]:
#                     continue
#                 mask[i, idx] = True
#                 needed -= 1
#                 if needed <= 0:
#                     break
#     return mask.to(torch.uint8)


# def calibrate_pool(self, loader, classifier_pool):
#     classifier_keys = self.global_clf_keys
#     num_classifiers = len(classifier_keys)
#     active_models = [
#         classifier_pool[key].to(self.device).eval() for key in classifier_keys
#     ]

#     logits_all, labels_all = [], []
#     batches = list(loader)

#     with torch.no_grad():
#         for batch in batches:
#             x, y = process_batch(batch, self.device)
#             per_model_logits = []
#             for model in active_models:
#                 logits = model(x)
#                 if isinstance(logits, tuple):
#                     logits = logits[0]
#                 per_model_logits.append(logits.cpu())
#             logits_all.append(torch.stack(per_model_logits, dim=1))
#             labels_all.append(y.cpu())

#     for model in active_models:
#         model.to("cpu").eval()

#     if logits_all:
#         logits = torch.cat(logits_all, dim=0)
#         labels = torch.cat(labels_all, dim=0)
#     else:
#         logits = torch.empty((0, num_classifiers, 1), dtype=torch.float32)
#         labels = torch.empty((0,), dtype=torch.long)

#     temperatures: List[float] = []
#     for i in range(num_classifiers):
#         logits_slice = logits[:, i, :]
#         if logits_slice.numel() == 0 or labels.numel() == 0:
#             temperatures.append(1.0)
#             continue
#         T = fit_calibrator(logits_slice, labels)
#         temperatures.append(T)

#     setattr(self, "_pool_calibrators", temperatures)
#     return temperatures



# def project_to_DS(self, loader, classifier_pool, calibrate_probs = True):

#     classifier_keys = self.global_clf_keys
#     num_classifiers = len(classifier_keys)

#     # Activate models on device in a fixed order
#     active_models: List[torch.nn.Module] = [
#         classifier_pool[key].to(self.device).eval() for key in classifier_keys
#     ]


#     logits_all, labels_all = [], []
#     embeddings_all = [] 
#     batches = list(loader)

#     with torch.no_grad():
#         for batch in batches:
#             x, y = process_batch(batch, self.device)

#             # Collect raw logits from each model for this batch
#             per_model_logits = []
#             per_model_embeds = [] 
#             for model in active_models:
#                 logits, embeds = _forward_with_embedding(model, x)
#                 per_model_embeds.append(embeds.cpu())
#                 per_model_logits.append(logits.cpu())

#             # Stack to [B, M, C] and keep on CPU to save VRAM
#             batch_logits = torch.stack(per_model_logits, dim=1)  # [B, M, C]
#             logits_all.append(batch_logits)
#             labels_all.append(y.cpu())
#             batch_embeds = torch.stack(per_model_embeds, dim=1)  # [B, M, D]
#             embeddings_all.append(batch_embeds)

#     # Free GPU memory
#     for model in active_models:
#         model.to("cpu").eval()

#     # Concatenate all batches: [N, M, C]
#     logits = torch.cat(logits_all, dim=0)
#     labels = torch.cat(labels_all, dim=0)   # [N]
#     embeddings = torch.cat(embeddings_all, dim=0) 


#     if calibrate_probs:
#         # Choose calibration method (configurable; default to temp-scaling)
#         calib_method = getattr(self.args, "pool_calib_method", "ts-mix")

#         # ----------------------------------------------------
#         # 1) Fit calibrators once (on the "train" split) and cache
#         # ----------------------------------------------------
#         pool_calibrators = getattr(self, "_pool_calibrators", None)
#         if pool_calibrators is None:
#             pool_calibrators = []
#             for i in range(num_classifiers):
#                 logits_slice = logits[:, i, :]  # [N, C]

#                 if calib_method == "temp-scaling":
#                     # Your existing manual TS (LBFGS on CE)
#                     T = fit_calibrator(logits_slice, labels)
#                     pool_calibrators.append({"type": "temp-scaling", "T": float(T)})

#                 elif calib_method == "ts-mix":
#                     # Probmetrics TS + Laplace smoothing
#                     calib = get_calibrator("ts-mix")
#                     dist_fit = CategoricalLogits(logits_slice)
#                     calib.fit_torch(dist_fit, labels)
#                     pool_calibrators.append({"type": "ts-mix", "calib": calib})

#                 else:
#                     raise ValueError(f"Unknown pool_calib_method: {calib_method!r}")

#             # Cache for later splits (val/test)
#             self._pool_calibrators = pool_calibrators

#         # ----------------------------------------------------
#         # 2) Apply calibrators to get calibrated probabilities
#         # ----------------------------------------------------
#         probs_per_model = []
#         for i in range(num_classifiers):
#             logits_slice = logits[:, i, :]      # [N, C]
#             cal = self._pool_calibrators[i]

#             if cal["type"] == "temp-scaling":
#                 T = cal["T"]
#                 probs_i = torch.softmax(logits_slice / T, dim=1)

#             elif cal["type"] == "ts-mix":
#                 dist_apply = CategoricalLogits(logits_slice)
#                 result = cal["calib"].predict_proba_torch(dist_apply)
#                 probs_i = result.get_probs()     # [N, C]

#             probs_per_model.append(probs_i.unsqueeze(1))  # [N,1,C]

#         probs = torch.cat(probs_per_model, dim=1)         # [N,M,C]
#     else:
#         probs = torch.softmax(logits, dim=2)

#         # At init / first call
#     #     if getattr(self, "_pool_calibrators", None) is None:
#     #         self._pool_calibrators = []
#     #         for m in range(num_classifiers):
#     #             if calib_method == "temp-scaling":
#     #                 T = fit_calibrator(...)
#     #                 self._pool_calibrators.append({"type": "temp-scaling", "T": T})
#     #             elif calib_method == "ts-mix":
#     #                 calib = get_calibrator("ts-mix")
#     #                 calib.fit_torch(...)
#     #                 self._pool_calibrators.append({"type": "ts-mix", "calib": calib})

#     #     probs_per_model = []
#     #     for i in range(num_classifiers):
#     #         logits_slice = logits[:, i, :]
#     #         if reuse:
#     #             T = temperatures[i]
#     #         else:
#     #             T = fit_calibrator(logits_slice, labels)
#     #             temperatures.append(T)
#     #         probs_i = torch.softmax(logits_slice / T, dim=1)
#     #         probs_per_model.append(probs_i.unsqueeze(1))

#     #     if not reuse:
#     #         setattr(self, "_pool_calibrators", temperatures)

#     #     probs = torch.cat(probs_per_model, dim=1)
#     # else:
#     #     probs = torch.softmax(logits, dim=2)

#     preds = probs.argmax(dim=2)                 # [N, M]
#     meta_labels = compute_meta_labels(
#         probs, preds, labels,
#         min_positive=int(getattr(self.args, "graph_meta_min_pos", 3)),
#     )

#     probs_flat = probs.reshape(probs.size(0), -1)

#     feat_mode = str(getattr(self.args, "graph_sample_node_feats", "ds")).lower()

#     if feat_mode == "ds":
#         features = probs_flat
#     elif feat_mode == "embedding_mean":
#         features = embeddings.mean(dim=1)
#     elif feat_mode == "embedding_concat":
#         features = embeddings.reshape(embeddings.size(0), -1)
#     elif feat_mode == "encoder":
#         dataset_name = str(getattr(self.args, "data", getattr(self.args, "dataset", ""))).lower()
#         if "eicu" in dataset_name:
#             features = encode_eicu_features(self, batches)
#         else:
#             features = encode_with_graph_encoder(self, batches)
#     elif feat_mode == "meta_feats":
#         # META-DES style meta-features require the full (train + eval) context.
#         # We compute these later inside `build_train_eval_graph` where we have
#         # access to the combined graph (train+val/test) and train-only meta labels.
#         # Here we return a placeholder to keep bundle shapes consistent.
#         features = probs_flat
#     else:
#         # Fallback to decision-space features
#         features = probs_flat


#     return probs_flat, preds, labels, meta_labels, features



# from sklearn.cluster import KMeans

# def generate_prototypes(self, ds, feats, meta, y, max_k=5, min_samples_per_k=5):
#     """
#     Generates clustered prototypes. 
#     Returns a dict or list of prototype objects.
#     """
#     # Ensure CPU tensors
#     ds, feats, meta, y = ds.cpu(), feats.cpu(), meta.cpu(), y.cpu()
    
#     prototypes = []
#     unique_classes = torch.unique(y)

#     for c in unique_classes:
#         # 1. Filter data for this class
#         mask = (y == c)
#         c_feats = feats[mask]
#         c_ds = ds[mask]
#         c_meta = meta[mask]
#         n_samples = c_feats.shape[0]

#         # 2. Adaptive K
#         # e.g., if n=8, min=5 -> K=1. If n=50, min=5 -> K=5.
#         k = max(1, min(max_k, n_samples // min_samples_per_k))
        
#         # 3. Clustering
#         # If K=1, just take the mean directly to save time/stability
#         if k == 1:
#             proto_feat = c_feats.mean(dim=0, keepdim=True)
#             proto_ds = c_ds.mean(dim=0, keepdim=True)
#             proto_meta = c_meta.mean(dim=0, keepdim=True)
#         else:
#             kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.id)
#             cluster_ids = kmeans.fit_predict(c_feats.numpy())
#             cluster_ids = torch.from_numpy(cluster_ids)
            
#             # 4. Compute Centroids for ALL 3 types
#             proto_feat, proto_ds, proto_meta = [], [], []
#             for i in range(k):
#                 c_mask = (cluster_ids == i)
#                 proto_feat.append(c_feats[c_mask].mean(dim=0))
#                 proto_ds.append(c_ds[c_mask].mean(dim=0))
#                 proto_meta.append(c_meta[c_mask].mean(dim=0)) # Soft Label!
            
#             proto_feat = torch.stack(proto_feat)
#             proto_ds = torch.stack(proto_ds)
#             proto_meta = torch.stack(proto_meta)

#         # Store
#         prototypes.append({
#             "class": c.item(),
#             "feats": proto_feat, # [K, Feat_Dim]
#             "ds": proto_ds,      # [K, DS_Dim] - Needed for SS edges
#             "meta": proto_meta   # [K, N_Models] - Needed for CS edges/Loss
#         })

#     return prototypes

# # --- META-DES-style meta-feature computation ---
# def compute_meta_des_sample_feats(
#     *,
#     combined_ds: np.ndarray,           # [n_total, M*C]
#     ss_edge_index: np.ndarray,         # [2, E] SS edges (needed for RoC)
#     y_train: np.ndarray,               # [n_train]
#     tr_meta_labels: np.ndarray,        # [n_train, M] 0/1 correctness
#     n_train: int,
#     n_total: int,
#     num_classifiers: int,
#     num_classes: int,
#     K: int = 7,
#     Kp: int = 5,
#     eps: float = 1e-8,
# ) -> np.ndarray:
#     """Compute per-sample META-DES-style meta-feature vectors.

#     This computes the five META-DES meta-feature *sets* (f1..f5) for every sample node
#     using TRAIN samples as the labeled DSEL (dynamic selection dataset).

#     Optimization notes:
#       - f1/f2/f3 reuse the already-built SS neighborhoods (incoming TRAIN neighbors) as the RoC θ_j.
#         We do NOT use SS edge weights here (pure neighborhood membership only).
#       - f4 is computed properly in output-profile space (Hamming distance over the pool's hard predictions),
#         but we precompute unique output profiles among TRAIN samples to avoid full O(n_train) scans per node.

#     For a destination sample j:

#       - θ_j: Region of competence (RoC) = K SS-neighbors into j (TRAIN sources only).
#       - φ_j: Output-profile neighborhood = Kp closest TRAIN output profiles to query's output profile.

#     Meta-feature sets per classifier m:
#       1) f1: Neighbors' hard classification on θ_j
#            binary correctness of classifier m on each neighbor in θ_j  -> [M, K]
#       2) f2: Posterior probabilities on θ_j
#            p_m(w_l | x_k) on each neighbor x_k in θ_j, where w_l is classifier m's
#            predicted class on the query sample j                     -> [M, K]
#       3) f3: Overall local accuracy on θ_j
#            mean correctness of classifier m over θ_j                 -> [M, 1]
#       4) f4: Output profiles classification on φ_j
#            binary correctness of classifier m on each OP-neighbor     -> [M, Kp]
#       5) f5: Classifier confidence on the query
#            distance-to-boundary proxy via probability margin
#            (top1 - top2) for classifier m on query j                 -> [M, 1]

#     Final per-sample feature dim = M * (2*K + Kp + 2)
#       (f1:K) + (f2:K) + (f3:1) + (f4:Kp) + (f5:1)

#     Notes:
#       - θ_j is derived from SS edges, not recomputed via fresh KNN in DS.
#       - φ_j uses output-profile distance (Hamming) computed against UNIQUE TRAIN profiles, then expanded to
#         specific TRAIN samples.
#     """

#     combined_ds = np.asarray(combined_ds, dtype=np.float32)
#     ss_edge_index = np.asarray(ss_edge_index, dtype=np.int64)
#     y_train = np.asarray(y_train, dtype=np.int64)
#     tr_meta_labels = np.asarray(tr_meta_labels, dtype=np.uint8)

#     M = int(num_classifiers)
#     C = int(num_classes)

#     # Reshape decision space back to probabilities
#     probs_all = combined_ds.reshape(n_total, M, C).astype(np.float32)   # [n_total, M, C]
#     preds_all = probs_all.argmax(axis=2).astype(np.int64)               # [n_total, M]

#     # TRAIN-only blocks
#     train_probs = probs_all[:n_train]                                   # [n_train, M, C]
#     train_preds = preds_all[:n_train]                                   # [n_train, M]

#     # ------------------------------------------------------------------
#     # Build RoC bundles θ_j by reusing SS neighborhoods (TRAIN sources only)
#     # ------------------------------------------------------------------
#     # ss_edge_index is [2, E] with edges src -> dst over sample nodes
#     if ss_edge_index.size == 0:
#         neighbors_by_dest: Dict[int, np.ndarray] = {}
#     else:
#         src_ids = ss_edge_index[0]
#         dst_ids = ss_edge_index[1]
#         # collect incoming TRAIN sources for each destination
#         buckets: Dict[int, List[int]] = {}
#         for s, d in zip(src_ids.tolist(), dst_ids.tolist()):
#             if s >= n_train:
#                 continue
#             buckets.setdefault(int(d), []).append(int(s))
#         neighbors_by_dest = {d: np.asarray(v, dtype=np.int64) for d, v in buckets.items()}

#     # ------------------------------------------------------------------
#     # Precompute UNIQUE TRAIN output profiles to accelerate φ_j search
#     # ------------------------------------------------------------------
#     # Each output profile is a length-M vector of hard predictions across the pool.
#     # Distance in OP space is Hamming distance between profiles.
#     if n_train > 0:
#         uniq_profiles, inv = np.unique(train_preds, axis=0, return_inverse=True)  # [P,M], [n_train]
#         P = int(uniq_profiles.shape[0])
#         # Map profile id -> list of train indices with that profile
#         prof_to_indices: List[np.ndarray] = [None] * P
#         # Use a single pass gather to avoid repeated np.where calls
#         tmp_lists: List[List[int]] = [[] for _ in range(P)]
#         for idx, pid in enumerate(inv.tolist()):
#             tmp_lists[int(pid)].append(int(idx))
#         for pid in range(P):
#             prof_to_indices[pid] = np.asarray(tmp_lists[pid], dtype=np.int64)
#     else:
#         uniq_profiles = np.zeros((0, M), dtype=np.int64)
#         prof_to_indices = []

#     # Final feature matrix
#     feats = np.zeros((n_total, M * (2 * K + Kp + 2)), dtype=np.float32)

#     # Precompute classifier indices for vectorized gathers
#     clf_idx = np.arange(M, dtype=np.int64)

#     for j in range(n_total):
#         # ------------------------------------------------------------
#         # θ_j: Region of competence (reuse SS neighborhood)
#         # ------------------------------------------------------------
#         neigh = neighbors_by_dest.get(j, None)
#         if neigh is None or neigh.size == 0:
#             # If SS neighborhood is missing, keep zeros for f1/f2/f3 (robust fallback)
#             nn_idx = np.empty((0,), dtype=np.int64)
#         else:
#             # Ensure we don't include self for training nodes
#             if j < n_train:
#                 neigh = neigh[neigh != j]
#             # Sort incoming SS neighbors by DS distance (L1) ascending to mimic META-DES RoC as closely as possible.
#             # NOTE: This does NOT change SS edges/weights; it only chooses which neighbors populate f1/f2/f3.
#             q_flat = combined_ds[j]  # [M*C]
#             d = np.sum(np.abs(combined_ds[neigh] - q_flat[None, :]), axis=1)  # [|neigh|]
#             order = np.argsort(d, kind="stable")
#             neigh_sorted = neigh[order]
#             nn_idx = neigh_sorted[: min(int(K), int(neigh_sorted.size))]

#         k_eff = int(nn_idx.size)

#         # --- f1: correctness pattern on θ_j (binary) ---
#         # [k_eff, M] -> [M, k_eff]
#         if k_eff > 0:
#             f1 = tr_meta_labels[nn_idx, :].astype(np.float32).T
#         else:
#             f1 = np.zeros((M, 0), dtype=np.float32)

#         # --- f2: posterior pattern on θ_j ---
#         # For each classifier m, use its predicted class on the *query* as w_l
#         # and evaluate p_m(w_l | x_k) on each neighbor x_k.
#         q_probs = probs_all[j]                                          # [M, C]
#         w_l = q_probs.argmax(axis=1).astype(np.int64)                   # [M]

#         if k_eff > 0:
#             # Gather train_probs[nn_idx, m, w_l[m]] for each neighbor and classifier.
#             # Result shape: [k_eff, M] -> transpose to [M, k_eff]
#             f2 = train_probs[nn_idx[:, None], clf_idx[None, :], w_l[None, :]].astype(np.float32).T
#         else:
#             f2 = np.zeros((M, 0), dtype=np.float32)

#         # --- f3: overall local accuracy on θ_j (scalar per classifier) ---
#         if k_eff > 0:
#             f3 = f1.mean(axis=1, keepdims=True).astype(np.float32)      # [M, 1]
#         else:
#             f3 = np.zeros((M, 1), dtype=np.float32)

#         # ------------------------------------------------------------
#         # φ_j: Output-profile neighborhood (Kp nearest TRAIN OPs)
#         # ------------------------------------------------------------
#         kp_eff = min(int(Kp), int(n_train))
#         if kp_eff > 0 and uniq_profiles.shape[0] > 0:
#             op_q = preds_all[j].astype(np.int64)                        # [M]

#             # Hamming distance to UNIQUE profiles (P << n_train often)
#             # d_prof[p] = mean(op_q != uniq_profiles[p])
#             d_prof = (uniq_profiles != op_q[None, :]).mean(axis=1).astype(np.float32)  # [P]

#             # Pick best profile ids then expand to sample indices
#             kp_prof = min(kp_eff, int(d_prof.size))
#             sel_pids = np.argpartition(d_prof, kp_prof - 1)[:kp_prof]
#             # Sort selected profiles by distance (smallest first)
#             sel_pids = sel_pids[np.argsort(d_prof[sel_pids])]

#             op_neighbors: List[int] = []
#             for pid in sel_pids.tolist():
#                 ids = prof_to_indices[int(pid)]
#                 if ids.size == 0:
#                     continue
#                 # Add indices from this profile bucket
#                 op_neighbors.extend(ids.tolist())
#                 if len(op_neighbors) >= kp_eff:
#                     break

#             if len(op_neighbors) > kp_eff:
#                 op_neighbors = op_neighbors[:kp_eff]

#             op_idx = np.asarray(op_neighbors, dtype=np.int64)
#             # Avoid self if train node
#             if j < n_train:
#                 op_idx = op_idx[op_idx != j]
#                 if op_idx.size > kp_eff:
#                     op_idx = op_idx[:kp_eff]

#             # f4: correctness pattern on φ_j
#             f4 = tr_meta_labels[op_idx, :].astype(np.float32).T if op_idx.size else np.zeros((M, 0), dtype=np.float32)
#         else:
#             f4 = np.zeros((M, 0), dtype=np.float32)

#         # --- f5: classifier confidence on the query (margin proxy) ---
#         if C >= 2:
#             part = np.partition(-q_probs, kth=1, axis=1)
#             top1 = -part[:, 0]
#             top2 = -part[:, 1]
#             margin = (top1 - top2)
#         else:
#             margin = np.ones((M,), dtype=np.float32)
#         f5 = margin.reshape(M, 1).astype(np.float32)

#         # ------------------------------------------------------------
#         # Pad to fixed sizes K / Kp
#         # ------------------------------------------------------------
#         if k_eff < K:
#             pad = K - k_eff
#             f1 = np.pad(f1, ((0, 0), (0, pad)), mode="constant")
#             f2 = np.pad(f2, ((0, 0), (0, pad)), mode="constant")

#         # f4 padding
#         if f4.shape[1] < Kp:
#             padp = Kp - f4.shape[1]
#             f4 = np.pad(f4, ((0, 0), (0, padp)), mode="constant")

#         per_clf = np.concatenate([f1, f2, f3, f4, f5], axis=1)           # [M, 2K + Kp + 2]
#         feats[j] = per_clf.reshape(-1).astype(np.float32)

#     return feats


# def encode_with_graph_encoder(self, batches):
#     encoder = getattr(self, "_graph_encoder", None)
#     if encoder is None:
#         encoder = init_graph_encoder()
#         self._graph_encoder = encoder.to(self.device)
#     encoder.eval()

#     encoded_feats = []
#     with torch.no_grad():
#         for batch in batches:
#             x, _ = process_batch(batch, self.device)
#             x = preprocess_for_encoder(x)
#             feat = encoder(x)
#             feat = feat.reshape(feat.size(0), -1)
#             encoded_feats.append(feat.cpu())

#     if not encoded_feats:
#         return torch.empty(0, 0)
#     return torch.cat(encoded_feats, dim=0)


# def encode_eicu_features(self, batches):
#     encoded_feats = []
#     with torch.no_grad():
#         for batch in batches:
#             x, _ = process_batch(batch, self.device)
#             if x.dim() == 3:
#                 feat = x.mean(dim=1)
#             else:
#                 feat = x.reshape(x.size(0), -1)
#             encoded_feats.append(feat.cpu())

#     if not encoded_feats:
#         return torch.empty(0, 0)
#     return torch.cat(encoded_feats, dim=0)


# def preprocess_for_encoder(x: torch.Tensor) -> torch.Tensor:
#     x = x.to(torch.float32)
#     if x.dim() == 4 and x.size(2) != 224:
#         x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
#     mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, -1, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, -1, 1, 1)
#     return (x - mean) / std


# def init_graph_encoder(name: str = "resnet18"):
#     if name != "resnet18":
#         raise NotImplementedError(f"Encoder '{name}' is not supported yet.")
#     try:
#         base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#     except Exception:
#         base = models.resnet18(pretrained=True)
#     encoder = torch.nn.Sequential(*list(base.children())[:-1])
#     return encoder


# def build_train_eval_graph(self, 
#                  tr_ds, tr_preds, tr_meta_labels, y_tr, tr_feats,
#                  eval_ds, y_eval, eval_feats, eval_type: str) -> Tuple[Any, Any]:

#     n_train = tr_ds.shape[0]
#     n_eval = eval_ds.shape[0]
#     n_total = n_train + n_eval

#     combined_ds = np.vstack([tr_ds, eval_ds])
#     combined_y = np.concatenate([y_tr, y_eval])

#     # Build SS edges first so META-DES-style features can reuse SS neighborhoods (RoC bundles)
#     ss_edge_index, ss_edge_attr = build_ss_edges_cmdw(
#         decision_matrix=combined_ds,
#         label_vector=combined_y,
#         source_indices=np.arange(n_train, dtype=np.int64),
#         destination_indices=np.arange(n_total, dtype=np.int64),
#         k_per_class=int(getattr(self.args, "graph_k_per_class", 5)),
#     )

#     feat_mode = str(getattr(self.args, "graph_sample_node_feats", "ds")).lower()
#     if feat_mode == "meta_feats":
#         # Build META-DES-style meta-features using TRAIN as DSEL for *all* nodes.
#         # Reuse SS neighborhoods for f1/f2/f3, compute f4 properly in OP space, and f5 via margin.
#         K = int(getattr(self.args, "meta_des_K", 7))
#         Kp = int(getattr(self.args, "meta_des_Kp", 5))
#         sample_feats = compute_meta_des_sample_feats(
#             combined_ds=combined_ds,
#             ss_edge_index=ss_edge_index,
#             y_train=y_tr,
#             tr_meta_labels=tr_meta_labels,
#             n_train=n_train,
#             n_total=n_total,
#             num_classifiers=len(self.global_clf_keys),
#             num_classes=self.args.num_classes,
#             K=K,
#             Kp=Kp,
#         )
#     else:
#         sample_feats = np.vstack([tr_feats, eval_feats])

#     cs_mode = str(getattr(self.args, "graph_cs_mode", "balanced_acc:logloss")).lower()
#     if ":" in cs_mode:
#         score_mode, tie_break_mode = (part.strip() for part in cs_mode.split(":", 1))
#     else:
#         score_mode, tie_break_mode = cs_mode.strip(), "logloss"
#     if tie_break_mode in {"none", ""}:
#         tie_break_mode = None

#     cs_edge_index, cs_edge_attr = build_cs_edges_v2(
#         tr_meta_labels=tr_meta_labels,
#         decision_all=combined_ds,
#         y_train=y_tr,
#         ss_edge_index=ss_edge_index,
#         ss_edge_attr=ss_edge_attr,
#         n_train=n_train,
#         n_total=n_total,
#         num_classes=self.args.num_classes,
#         top_k=self.args.graph_cs_topk,
#         score_mode=score_mode,
#         tie_break_mode=tie_break_mode,
#     )
#     # cs_edge_index, cs_edge_attr = build_cs_edges(
#     #     tr_meta_labels=tr_meta_labels,                 # [N_train, M] 0/1
#     #     decision_all=combined_ds,                  # [N_total, M*C] probs
#     #     y_train=y_tr,                             # [N_train] labels (np array)
#     #     ss_edge_index=ss_edge_index,
#     #     ss_edge_attr=ss_edge_attr,
#     #     n_train=n_train,
#     #     n_total=n_total,
#     #     num_classes=self.args.num_classes,
#     #     top_k=int(getattr(self.args, "graph_cs_topk", 3)),
#     # )

#     cc_edge_index, cc_edge_attr = build_cc_edges(
#         y_true=y_tr,
#         y_pred_matrix=tr_preds.cpu().numpy(),
#         top_k=int(getattr(self.args, "graph_cc_topk", 3)),
#         verbose=False,
#     )

#     num_classifiers, num_classes = len(self.global_clf_keys), self.args.num_classes
#     probs = tr_ds.reshape(-1, num_classifiers, num_classes)

#     present_only = True  # only consider classes present in labels
#     if present_only:
#         class_ids = np.unique(y_tr)
#     else:
#         class_ids = np.arange(num_classes)

#     class_masks = [y_tr == cls for cls in class_ids]
#     class_counts = np.array([mask.sum() for mask in class_masks])  # [K]
#     has_support = class_counts > 0
#     safe_counts = np.where(has_support, class_counts, 1.0)[:, None]  # [K,1]

#     # helper: per-class mean over samples with mask
#     def masked_mean(values: np.ndarray) -> np.ndarray:
#         out = []
#         for mask, count in zip(class_masks, class_counts):
#             if count > 0: out.append(values[mask].mean(axis=0))
#             else: out.append(np.zeros(values.shape[1:]))
#         return np.asarray(out)  # [K, ...]
    
#     def agg_over_supported(matrix: np.ndarray) -> np.ndarray:
#         supported = matrix[has_support]
#         if supported.size == 0:
#             return np.zeros((1, matrix.shape[1]))
#         return supported.mean(axis=0, keepdims=True)
    
#     per_class_hard_recall = masked_mean(tr_meta_labels)  # [K, M]
#     true_class_probs = probs[np.arange(probs.shape[0]), :, y_tr][:, :, None]  # [N,M,1]
#     per_class_true_prob = masked_mean(true_class_probs.squeeze(-1))  # [K, M]

#     se = np.sqrt(per_class_hard_recall * (1.0 - per_class_hard_recall) / safe_counts)
#     se = np.where(has_support[:, None], se, 0.0)

#     overall_accuracy = tr_meta_labels.mean(axis=0, keepdims=True)  # [1, M]
#     mean_true_prob = agg_over_supported(per_class_true_prob)  # [1, M]
#     balanced_accuracy = agg_over_supported(per_class_hard_recall)  # [1, M]

#     clf_x = np.concatenate(
#         [
#             per_class_hard_recall.T,
#             per_class_true_prob.T,
#             se.T,
#             overall_accuracy.T,
#             mean_true_prob.T,
#             balanced_accuracy.T,
#         ],
#         axis=1,
#     )
 

#     # ---------- Compose HeteroData ----------
#     data = HeteroData()
#     data["sample"].x = torch.from_numpy(sample_feats).float()
#     data["sample"].y = torch.from_numpy(combined_y).long()
#     idx = torch.arange(n_total)
#     data["sample"].train_mask = idx < n_train
#     data["sample"][f"{eval_type}_mask"] = (idx >= n_train) & (n_eval > 0)

#     # classifier nodes
#     data["classifier"].x = torch.from_numpy(clf_x).float()             # [M, F]
#     data["classifier"].num_nodes = num_classifiers
#     data["classifier"].clf_keys = list(self.global_clf_keys)
#     # IMPORTANT: stable mapping from classifier node index -> column index in meta-labels.
#     # Pairwise decoders in the meta-learner (gnn_pair_decoder) assume logits[:, j] corresponds
#     # to classifier j in the same ordering used to build tr_preds / tr_meta_labels.
#     data["classifier"].clf_idx = torch.arange(num_classifiers, dtype=torch.long)

#     for key, edge_index, edge_attr in (
#         (("sample", "ss", "sample"), ss_edge_index, ss_edge_attr),
#         (("classifier", "cs", "sample"), cs_edge_index, cs_edge_attr),
#         (("classifier", "cc", "classifier"), cc_edge_index, cc_edge_attr),
#     ):
#         if edge_index.size:
#             data[key].edge_index = torch.from_numpy(edge_index).long()
#             data[key].edge_attr = torch.from_numpy(edge_attr).float()
#     graph = data

#     # ---------- Save graph summaries ----------
#     save_graph_summaries(self,
#         client_role=self.role,
#         args=getattr(self, "args", None),
#         n_train=n_train, n_eval=n_eval,
#         n_classifiers=int(clf_x.shape[0]),
#         classifier_names=list(self.global_clf_keys),
#         sample_labels=combined_y,
#         ss_edge_index=ss_edge_index, ss_edge_attr=ss_edge_attr,
#         cs_edge_index=cs_edge_index, cs_edge_attr=cs_edge_attr,
#         cc_edge_index=cc_edge_index, cc_edge_attr=cc_edge_attr,
#         bins=50,
#         eval_kind=eval_type,
#     )
    
#     warn_disconnected_classifier_cc(
#         graph, eval_type, getattr(self, "role", None)
#     )

#     return graph




# def warn_disconnected_classifier_cc(
#     graph: HeteroData, graph_label: str, client_role: str | None = None
# ):
#     """
#     Print classifiers that only connect through cc edges with no cs links to samples.
#     """
#     cc_rel = ("classifier", "cc", "classifier")
#     if cc_rel not in graph.edge_index_dict:
#         return

#     cc_edge_index = graph[cc_rel].edge_index
#     if cc_edge_index.numel() == 0:
#         return

#     num_classifiers = int(
#         getattr(graph["classifier"], "num_nodes", graph["classifier"].x.size(0))
#     )
#     neighbors = [[] for _ in range(num_classifiers)]
#     for src, dst in cc_edge_index.t().tolist():
#         neighbors[src].append(dst)
#         neighbors[dst].append(src)

#     cs_rel = ("classifier", "cs", "sample")
#     cs_sources = set()
#     if cs_rel in graph.edge_index_dict:
#         cs_edge_index = graph[cs_rel].edge_index
#         if cs_edge_index.numel():
#             cs_sources = set(cs_edge_index[0].tolist())

#     visited = [False] * num_classifiers
#     disconnected_components: list[list[int]] = []

#     for node_idx in range(num_classifiers):
#         if visited[node_idx] or not neighbors[node_idx]:
#             continue
#         stack = [node_idx]
#         component = []
#         while stack:
#             node = stack.pop()
#             if visited[node]:
#                 continue
#             visited[node] = True
#             component.append(node)
#             for neighbor in neighbors[node]:
#                 if not visited[neighbor]:
#                     stack.append(neighbor)
#         if component and not any(member in cs_sources for member in component):
#             disconnected_components.append(component)

#     if not disconnected_components:
#         return

#     clf_names = getattr(graph["classifier"], "clf_keys", None)
#     if not clf_names or len(clf_names) != num_classifiers:
#         clf_names = [f"classifier[{i}]" for i in range(num_classifiers)]

#     descriptions = []
#     for component in disconnected_components:
#         names = ", ".join(clf_names[idx] for idx in component)
#         descriptions.append(f"[{len(component)}] {names}")

#     role_label = f"Client {client_role}" if client_role else "Client"
#     print(
#         f"[FedDES][{role_label}] {graph_label} graph has "
#         f"{len(disconnected_components)} classifier-only cc component(s): "
#         f"{'; '.join(descriptions)}"
#     )

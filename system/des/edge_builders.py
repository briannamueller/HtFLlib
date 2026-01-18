import numpy as np
from typing import Tuple, List, Optional, Dict
from deslib.util.diversity import double_fault
from collections import defaultdict
import torch
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from typing import Tuple, Dict

# edge_builders.py
# des/edge_builders.py

# ... existing imports ...
import torch
from typing import List, Tuple

def build_cs_edges_hybrid(
    neighbor_indices: List[List[int]], 
    meta_labels: torch.Tensor, 
    is_proto_mask: torch.Tensor, 
    threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds Classifier-Sample edges with hybrid logic:
    - Real Samples: Score = Average of neighbors' meta-labels.
    - Prototypes: Score = (Self-label + Sum(neighbors)) / (1 + Count(neighbors))
    
    Args:
        neighbor_indices: Adjacency list [sample_idx] -> [list of neighbor_indices]
        meta_labels: Combined tensor [N_total, N_classifiers]. 
                     (Real samples = binary, Prototypes = soft).
        is_proto_mask: Boolean tensor [N_total] identifying prototype nodes.
        threshold: Minimum competence score to create an edge.
    """
    rows = []
    cols = []
    edge_weights = []
    
    num_samples = meta_labels.shape[0]
    num_classifiers = meta_labels.shape[1]
    
    # Ensure mask is boolean
    is_proto_mask = is_proto_mask.bool()

    for s_idx in range(num_samples):
        neighbors = neighbor_indices[s_idx]
        
        # 1. Aggregate Neighbors
        if len(neighbors) > 0:
            # If a neighbor is a prototype, this reads its soft label automatically
            neighbor_vals = meta_labels[neighbors] 
            sum_vals = neighbor_vals.sum(dim=0)
            count_vals = len(neighbors)
        else:
            sum_vals = torch.zeros(num_classifiers, device=meta_labels.device)
            count_vals = 0
            
        # 2. Hybrid Logic for Prototypes (Add Self-Vote)
        if is_proto_mask[s_idx]:
            # Add the prototype's own soft label to the aggregation
            sum_vals += meta_labels[s_idx]
            count_vals += 1
            
        # 3. Compute Final Score
        if count_vals > 0:
            scores = sum_vals / count_vals
        else:
            scores = torch.zeros(num_classifiers, device=meta_labels.device)
            
        # 4. Threshold & Create Edges
        # Find classifiers where score > threshold
        valid_clf_indices = (scores > threshold).nonzero(as_tuple=False).view(-1)
        
        for c_idx in valid_clf_indices:
            # Edge: Classifier (src) -> Sample (dst)
            rows.append(c_idx.item())
            cols.append(s_idx)
            edge_weights.append(scores[c_idx].item())

    if len(rows) == 0:
         return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    
    return edge_index, edge_attr

def build_ss_edges_cmdw(
    decision_matrix: np.ndarray,
    label_vector: np.ndarray,
    source_indices: np.ndarray,
    destination_indices: np.ndarray,
    k_per_class: int = 2,
    membership_mode: str = "soft",   # {"none", "hard", "soft"}
    membership_k: int = 7,
    eps: float = 1e-8,
    log_class_edge_stats: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sample→sample edges using CMDW-style class scores and within-class softmax.

    For each destination j and each class c among SOURCE samples:
      1) Select up to k_per_class nearest class-c sources to j (L1 in decision space), excluding j itself if same class.
      2) CMDW score s_c = m̄_c / (d̄_c + eps), where:
         - d̄_c is the mean L1 distance from j to cumulative means of the ordered neighbors.
         - m̄_c is either 1 (membership_mode="none") or the mean neighbor membership, where each neighbor's membership
           is the proportion ("soft") or hard majority ("hard") of same-label points in its own KNN over SOURCE.
      3) Split class mass across selected neighbors with a temperatured softmax over their raw L1 distances.
      4) Mix classes by π_c = s_c / Σ_c s_c and assign final edge weights γ_{c,r} = π_c * softmax_in_class[r].
    Returns:
        edge_index: int64 array of shape (2, E)
        edge_attr:  float32 array of shape (E,)
    """
    src_ids = np.asarray(source_indices, dtype=np.int64)
    dest_ids   = np.asarray(destination_indices, dtype=np.int64)
    label_vector = np.asarray(label_vector)

    # Map each class to its source ids for quick lookups
    src_labels = label_vector[src_ids]
    classes = np.unique(src_labels)
    sources_by_class: Dict[int, np.ndarray] = {c: src_ids[src_labels == c] for c in classes}

    # --- Helpers ----------------------------------------------------------------
    mem_cache: Dict[int, float] = {}

    def neighbor_membership(i: int) -> float:
        """Membership of source i w.r.t. its own KNN over SOURCE ids."""
        if membership_mode == "none":
            return 1.0
        if i in mem_cache:
            return mem_cache[i]

        xi = decision_matrix[i]
        pool = src_ids[src_ids != i]
        if pool.size == 0:
            mem_cache[i] = 1.0
            return 1.0

        d = np.sum(np.abs(decision_matrix[pool] - xi), axis=1)
        k = min(int(membership_k), pool.size)
        nn = pool[np.argpartition(d, k - 1)[:k]] if k > 0 else np.empty(0, dtype=np.int64)
        prop_same = float(np.mean(label_vector[nn] == label_vector[i])) if nn.size else 1.0

        val = 1.0 if (membership_mode == "hard" and prop_same > 0.5) else (prop_same if membership_mode == "soft" else 1.0)
        mem_cache[i] = val
        return val

    def softmax_over_neg(dist: np.ndarray) -> np.ndarray:
        """Compute softmax over -dist with temperature = median(dist)."""
        tau = max(float(np.median(dist)), eps)
        z = (-dist / tau)
        z -= z.max()  # numerical stability
        w = np.exp(z)
        return w / w.sum()

    # --- Build edges ------------------------------------------------------------
    src_list, dst_list, w_list = [], [], []

    for j in dest_ids.tolist():
        q = decision_matrix[j]

        # First pass: collect per-class neighbors and CMDW scores s_c
        per_class_neighbors: Dict[int, np.ndarray] = {}
        per_class_raw_dists: Dict[int, np.ndarray] = {}
        class_scores: Dict[int, float] = {}

        for c in classes.tolist():
            S_c = sources_by_class[c]
            if label_vector[j] == c:
                S_c = S_c[S_c != j]  # avoid self if same class
            if S_c.size == 0:
                continue

            # k nearest within class
            d_all = np.sum(np.abs(decision_matrix[S_c] - q), axis=1)
            k_c = min(int(k_per_class), S_c.size)
            idx = np.argpartition(d_all, k_c - 1)[:k_c]
            neigh_ids = S_c[idx]           # (k_c,)
            neigh_d   = d_all[idx]         # (k_c,)
            per_class_neighbors[c] = neigh_ids
            per_class_raw_dists[c] = neigh_d

            # CMDW: mean distance to cumulative means
            cum_means = np.cumsum(decision_matrix[neigh_ids], axis=0) / np.arange(1, k_c + 1)[:, None]
            dbar_c = float(np.mean(np.sum(np.abs(cum_means - q), axis=1)))

            # Membership average over neighbors
            mbar_c = 1.0 if membership_mode == "none" else float(np.mean([neighbor_membership(int(i)) for i in neigh_ids]))
            class_scores[c] = mbar_c / (dbar_c + eps)

        if not class_scores:
            continue

        # Normalize class masses π_c
        total_score = float(sum(class_scores.values()))
        class_mass = {c: class_scores[c] / total_score for c in class_scores}

        # Second pass: within-class softmax + mix with π_c
        for c, neigh_ids in per_class_neighbors.items():
            w_in_class = softmax_over_neg(per_class_raw_dists[c]).astype(np.float32)
            gamma = (class_mass[c] * w_in_class).astype(np.float32)

            src_list.extend(neigh_ids.tolist())
            dst_list.extend([j] * neigh_ids.size)
            w_list.extend(gamma.tolist())

    if not src_list:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    if log_class_edge_stats:
        pair_weights: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        for src_id, dst_id, weight in zip(src_list, dst_list, w_list):
            pair_weights[(int(label_vector[dst_id]), int(label_vector[src_id]))].append(weight)

        classes = np.unique(label_vector)
        for dst_cls in classes.tolist():
            for src_cls in classes.tolist():
                weights = pair_weights.get((int(dst_cls), int(src_cls)), [])
                avg_weight = float(np.mean(weights)) if weights else 0.0
                print(f"[SS edges] class {int(dst_cls)} incoming from {int(src_cls)} average weight: {avg_weight:.6f}")

    edge_index = np.vstack([np.asarray(src_list, dtype=np.int64),
                            np.asarray(dst_list, dtype=np.int64)])
    edge_attr = np.asarray(w_list, dtype=np.float32)
    return edge_index, edge_attr


def _to_numpy(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x.astype(dtype) if dtype is not None else x

import numpy as np
from typing import Tuple, Optional, Dict

def _to_numpy(x, dtype=None):
    if isinstance(x, np.ndarray):
        arr = x
    else:
        # torch.Tensor or list-like
        try:
            import torch
            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().numpy()
            else:
                arr = np.asarray(x)
        except Exception:
            arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr

def build_cs_edges_v2(
    *,
    tr_meta_labels: np.ndarray,      # [n_train, M] correctness (0/1 for real, 0..1 for protos)
    decision_all: np.ndarray,        # [n_total, M*C] flattened probs
    y_train: np.ndarray,             # [n_train]
    ss_edge_index: np.ndarray,       # [2, E] sample->sample
    ss_edge_attr: np.ndarray,        # [E] weights
    n_train: int,
    n_total: int,
    num_classes: int,
    top_k: int = 3,
    score_mode: str = "balanced_acc",
    tie_break_mode: Optional[str] = "true_prob",
    gain_baseline: str = "mean_pool",
    eps: float = 1e-8,
    is_proto_mask: Optional[np.ndarray] = None, # <--- NEW ARGUMENT
    proto_self_weight: float = 1.0              # <--- NEW ARGUMENT
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard FedDES CS edge builder, updated to support Prototypes.
    
    The Logic:
    - For Real Samples: Uses neighbors only (Standard DES).
    - For Prototypes: Uses neighbors + Self (The "Small Tweak").
      The prototype is added to its own neighborhood bundle, allowing it to 
      influence the 'logloss' or 'accuracy' score calculation directly.
    """

    # Remove the uint8 cast so we preserve soft labels for prototypes!
    # tr_meta_labels = _to_numpy(tr_meta_labels, np.uint8) <--- REMOVED
    tr_meta_labels = _to_numpy(tr_meta_labels, np.float32) # Keep as float
    decision_all   = _to_numpy(decision_all,   np.float32)
    y_train        = _to_numpy(y_train,        np.int64)
    ss_edge_index  = _to_numpy(ss_edge_index,  np.int64)
    ss_edge_attr   = _to_numpy(ss_edge_attr,   np.float32)
    
    if is_proto_mask is not None:
        is_proto_mask = _to_numpy(is_proto_mask, bool)

    M = int(tr_meta_labels.shape[1])
    C = int(num_classes)

    # ---------- build neighbor bundles ----------
    src, dst = ss_edge_index
    w = ss_edge_attr.astype(np.float32)

    neighbors_by_dest: Dict[int, Dict[str, List]] = {}
    
    # Standard neighbor collection
    for s, d, weight in zip(src.tolist(), dst.tolist(), w.tolist()):
        if s >= n_train: continue
        bundle = neighbors_by_dest.setdefault(d, {"idx": [], "w": []})
        bundle["idx"].append(s)
        bundle["w"].append(weight)

    # --- THE TWEAK: Add Self-Loop for Prototypes ---
    if is_proto_mask is not None:
        # Identify prototype indices (they must be < n_train to serve as sources)
        proto_indices = np.where(is_proto_mask[:n_train])[0]
        
        for p_idx in proto_indices:
            # Ensure bundle exists
            bundle = neighbors_by_dest.setdefault(p_idx, {"idx": [], "w": []})
            
            # Add self to the neighborhood
            bundle["idx"].append(p_idx)
            bundle["w"].append(proto_self_weight)

    # Convert lists to arrays for vectorized scoring
    final_bundles = {}
    for d, bundle in neighbors_by_dest.items():
        final_bundles[d] = {
            "idx": np.asarray(bundle["idx"], dtype=np.int64),
            "w":   np.asarray(bundle["w"],   dtype=np.float32)
        }

    # ---------- Internal Scoring Helpers (Unchanged Logic) ----------
    def neigh_true_prob_matrix(neigh_idx: np.ndarray) -> np.ndarray:
        neigh_label = y_train[neigh_idx]
        base = (np.arange(M, dtype=np.int64) * C)[None, :]
        idx  = base + neigh_label[:, None]
        return decision_all[neigh_idx[:, None], idx]

    def score_logloss(neigh_idx: np.ndarray, neigh_w: np.ndarray) -> np.ndarray:
        P = neigh_true_prob_matrix(neigh_idx)
        P = np.clip(P, eps, 1.0)
        return -(neigh_w[:, None] * np.log(P)).sum(axis=0)

    def score_gain(neigh_idx: np.ndarray, neigh_w: np.ndarray) -> np.ndarray:
        corr = tr_meta_labels[neigh_idx, :] # Now float compatible
        if gain_baseline == "mean_pool":
            base = corr.mean(axis=1, keepdims=True)
        else:
            base = 0.0
        return (neigh_w[:, None] * (corr - base)).sum(axis=0)

    def score_balanced_acc(neigh_idx: np.ndarray, neigh_w: np.ndarray) -> np.ndarray:
        neigh_label = y_train[neigh_idx]
        classes = np.unique(neigh_label)
        corr = tr_meta_labels[neigh_idx, :]
        out = np.zeros(M, dtype=np.float32)
        denom_classes = 0
        for c in classes:
            mask = (neigh_label == c)
            w_c = neigh_w[mask]
            denom = float(w_c.sum()) + eps
            a_c = (w_c[:, None] * corr[mask, :]).sum(axis=0) / denom
            out += a_c
            denom_classes += 1
        if denom_classes > 0: out /= denom_classes
        return out

    def score_true_prob(neigh_idx: np.ndarray, neigh_w: np.ndarray) -> np.ndarray:
        P = neigh_true_prob_matrix(neigh_idx)
        return (neigh_w[:, None] * P).sum(axis=0)

    def compute_score(mode: str, idx: np.ndarray, w: np.ndarray):
        if mode == "logloss": return score_logloss(idx, w)
        if mode == "gain": return score_gain(idx, w)
        if mode == "balanced_acc": return score_balanced_acc(idx, w)
        if mode == "true_prob": return score_true_prob(idx, w)
        raise ValueError(f"Unknown mode: {mode}")

    # ---------- Main Scoring Loop (Unchanged) ----------
    edge_src, edge_dst, edge_attr = [], [], []

    for dest_id in range(n_total):
        bundle = final_bundles.get(dest_id)
        if bundle is None or bundle["idx"].size == 0:
            continue

        idx, w_arr = bundle["idx"], bundle["w"]
        
        # Primary Score
        primary = compute_score(score_mode, idx, w_arr)
        
        # Secondary Score
        secondary = None
        if tie_break_mode:
            secondary = compute_score(tie_break_mode, idx, w_arr)

        # Sorting (Sorting logic remains exactly as in v2)
        # ... [Sort and Top-K logic] ...
        if score_mode == "logloss":
            pkey = primary
            if secondary is not None:
                skey = secondary if tie_break_mode == "logloss" else -secondary
                order = np.lexsort((skey, pkey))
            else:
                order = np.argsort(pkey)
        else:
            pkey = -primary
            if secondary is not None:
                skey = secondary if tie_break_mode == "logloss" else -secondary
                order = np.lexsort((skey, pkey))
            else:
                order = np.argsort(pkey)

        keep = order[:min(top_k, M)]
        
        # Edge Weight Calc (Monotone transformation)
        s = primary.astype(np.float32)
        if score_mode == "logloss":
            s0 = s - s.min()
            w_primary = np.exp(-s0)
        else:
            s0 = s - s.min()
            w_primary = s0 / (s0.max() + eps) if s0.max() > 0 else np.ones_like(s0)

        for clf_id in keep:
            edge_src.append(int(clf_id))
            edge_dst.append(int(dest_id))
            edge_attr.append(float(w_primary[clf_id]))

    if not edge_src:
        return np.zeros((2,0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    return np.vstack([edge_src, edge_dst]), np.array(edge_attr, dtype=np.float32)

# def build_cs_edges_v2(
#     *,
#     tr_meta_labels: np.ndarray,      # [n_train, M] correctness (0/1)
#     decision_all: np.ndarray,        # [n_total, M*C] flattened probs (or logits->probs upstream)
#     y_train: np.ndarray,             # [n_train]
#     ss_edge_index: np.ndarray,       # [2, E] sample->sample
#     ss_edge_attr: np.ndarray,        # [E] weights
#     n_train: int,
#     n_total: int,
#     num_classes: int,
#     top_k: int = 3,

#     # Primary / secondary scoring
#     score_mode: str = "balanced_acc",    # {"logloss", "gain", "balanced_acc"}
#     tie_break_mode: Optional[str] = "true_prob",  # {"true_prob","logloss","gain","balanced_acc",None}

#     # Gain mode options
#     gain_baseline: str = "mean_pool",    # {"mean_pool","none"}
#     eps: float = 1e-8,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Build classifier -> sample edges (CS) by scoring each classifier in the SS-neighborhood of each sample.

#     Supported primary criteria (score_mode):
#       1) "logloss":
#            score_m = - sum_{n in N(j)} w(n->j) * log p_m(y_n | x_n)       (lower is better)
#       2) "gain":
#            score_m = sum_{n in N(j)} w(n->j) * (correct(n,m) - baseline(n)) (higher is better)
#            baseline(n) defaults to mean correctness across classifiers on neighbor n.
#       3) "balanced_acc":
#            per-class weighted acc over neighborhood, then average across classes present in neighborhood
#            score_m = mean_{c in classes(N(j))} [ sum w * correct / sum w ]  (higher is better)

#     Optional tie-breaker (tie_break_mode):
#       - "true_prob": sum w * p_m(y_n|x_n)     (higher is better)
#       - or any of the 3 main modes again
#       - or None

#     Edge_attr:
#       - uses a normalized "confidence" derived from the primary score (monotone), so higher is always better.
#       - If you want raw score values, you can store them separately; for GNN edge weights, a [0,1]-ish
#         monotone value is usually safer.

#     Returns:
#       edge_index: (2, E_cs) int64, where rows are [clf_id, sample_id]
#       edge_attr : (E_cs,) float32, weight per CS edge (higher = better)
#     """

#     debug = True  # set True temporarily
#     debug_every = 200  # print every N destinations
#     max_debug_dests = 10  # cap verbose per-dest dumps

#     cnt_no_bundle = 0
#     cnt_empty_bundle = 0
#     cnt_s0max0 = 0                  # s0.max()==0 => all weights become 1 for that dest (for non-logloss)
#     cnt_all_one_weights = 0         # kept weights all == 1
#     cnt_num_neigh = []              # neighborhood sizes
#     cnt_ties_primary = 0
#     cnt_ties_primary_in_keep = 0
#     cnt_clip_prob = 0               # how often logloss needed clipping
#     # ---------- normalize inputs ----------
#     tr_meta_labels = _to_numpy(tr_meta_labels, np.uint8)
#     decision_all   = _to_numpy(decision_all,   np.float32)
#     y_train        = _to_numpy(y_train,        np.int64)
#     ss_edge_index  = _to_numpy(ss_edge_index,  np.int64)
#     ss_edge_attr   = _to_numpy(ss_edge_attr,   np.float32)

#     M = int(tr_meta_labels.shape[1])
#     C = int(num_classes)

#     # ---------- build neighbor bundles for each destination ----------
#     src, dst = ss_edge_index
#     w = ss_edge_attr.astype(np.float32)

#     neighbors_by_dest: Dict[int, Dict[str, np.ndarray]] = {}
#     for s, d, weight in zip(src.tolist(), dst.tolist(), w.tolist()):
#         # only allow TRAIN sources to define competence
#         if s >= n_train:
#             continue
#         bundle = neighbors_by_dest.setdefault(d, {"idx": [], "w": []})
#         bundle["idx"].append(s)
#         bundle["w"].append(weight)

#     for d, bundle in neighbors_by_dest.items():
#         bundle["idx"] = np.asarray(bundle["idx"], dtype=np.int64)
#         bundle["w"]   = np.asarray(bundle["w"],   dtype=np.float32)

#     # ---------- small helpers ----------
#     def neigh_true_prob_matrix(neigh_idx: np.ndarray) -> np.ndarray:
#         """
#         Returns P_true for each neighbor and classifier:
#             P_true[k, m] = p_m(y_k | x_k) where y_k is the true label of neighbor k.
#         Uses decision_all flattened indexing: index = m*C + y_k
#         """
#         neigh_label = y_train[neigh_idx]                     # [K]
#         # Build indices [K, M] of flattened decision vector positions
#         base = (np.arange(M, dtype=np.int64) * C)[None, :]   # [1, M]
#         idx  = base + neigh_label[:, None]                   # [K, M]
#         return decision_all[neigh_idx[:, None], idx]         # [K, M]

#     def score_logloss(neigh_idx: np.ndarray, neigh_w: np.ndarray) -> np.ndarray:
#         nonlocal cnt_clip_prob

#         P = neigh_true_prob_matrix(neigh_idx)                # [K, M]
#         # Track how often we need to clip tiny probabilities for numerical stability.
#         tiny_mask = (P <= eps * 1.01)
#         if np.any(tiny_mask):
#             cnt_clip_prob += 1

#         P = np.clip(P, eps, 1.0)

#         # Optional debug: print the fraction of tiny probs occasionally.
#         # (This helps diagnose why logloss may be dominated by near-zero true-class probs.)
#         if debug:
#             frac_tiny = float(np.mean(tiny_mask))
#             if frac_tiny > 0 and (dest_id % debug_every == 0):
#                 print(f"[CS][logloss][dest={dest_id}] frac probs <= ~eps: {frac_tiny:.4f}")

#         # score = - sum w * log(P)
#         return -(neigh_w[:, None] * np.log(P)).sum(axis=0).astype(np.float32)  # [M], lower better

#     def score_gain(neigh_idx: np.ndarray, neigh_w: np.ndarray) -> np.ndarray:
#         corr = tr_meta_labels[neigh_idx, :].astype(np.float32)   # [K, M]
#         if gain_baseline == "mean_pool":
#             base = corr.mean(axis=1, keepdims=True)              # [K, 1]
#         elif gain_baseline == "none":
#             base = 0.0
#         else:
#             raise ValueError(f"Unknown gain_baseline={gain_baseline}")
#         gain = (neigh_w[:, None] * (corr - base)).sum(axis=0).astype(np.float32)  # [M], higher better
#         return gain

#     def score_balanced_acc(neigh_idx: np.ndarray, neigh_w: np.ndarray) -> np.ndarray:
#         neigh_label = y_train[neigh_idx]                         # [K]
#         classes = np.unique(neigh_label)
#         corr = tr_meta_labels[neigh_idx, :].astype(np.float32)    # [K, M]
#         out = np.zeros(M, dtype=np.float32)
#         denom_classes = 0

#         for c in classes.tolist():
#             mask = (neigh_label == c)
#             if not np.any(mask):
#                 continue
#             w_c = neigh_w[mask]
#             denom = float(w_c.sum()) + eps
#             # weighted acc within class c
#             a_c = (w_c[:, None] * corr[mask, :]).sum(axis=0) / denom   # [M]
#             out += a_c.astype(np.float32)
#             denom_classes += 1

#         if denom_classes > 0:
#             out /= float(denom_classes)
#         return out.astype(np.float32)   # [M], higher better

#     def score_true_prob(neigh_idx: np.ndarray, neigh_w: np.ndarray) -> np.ndarray:
#         P = neigh_true_prob_matrix(neigh_idx)                     # [K, M]
#         return (neigh_w[:, None] * P).sum(axis=0).astype(np.float32)  # [M], higher better

#     def compute_score(mode: str, neigh_idx: np.ndarray, neigh_w: np.ndarray) -> np.ndarray:
#         if mode == "logloss":
#             return score_logloss(neigh_idx, neigh_w)
#         if mode == "gain":
#             return score_gain(neigh_idx, neigh_w)
#         if mode == "balanced_acc":
#             return score_balanced_acc(neigh_idx, neigh_w)
#         if mode == "true_prob":
#             return score_true_prob(neigh_idx, neigh_w)
#         raise ValueError(f"Unknown score mode: {mode}")

#     def order_from_scores(primary: np.ndarray, primary_mode: str,
#                           secondary: Optional[np.ndarray], secondary_mode: Optional[str]) -> np.ndarray:
#         """
#         Return indices sorted best->worst.
#         For 'logloss': smaller is better. For others: larger is better.
#         """
#         if primary_mode == "logloss":
#             pkey = primary
#             pkey = np.asarray(pkey, dtype=np.float32)
#             # smaller is better => sort ascending
#             if secondary is None:
#                 return np.argsort(pkey)
#             # secondary: handle direction too
#             if secondary_mode == "logloss":
#                 skey = secondary
#                 # lexsort uses last key as primary; want primary then secondary
#                 return np.lexsort((skey, pkey))
#             else:
#                 skey = -secondary
#                 return np.lexsort((skey, pkey))
#         else:
#             pkey = -primary  # larger better => sort ascending on -primary
#             if secondary is None:
#                 return np.argsort(pkey)
#             if secondary_mode == "logloss":
#                 skey = secondary  # smaller better
#             else:
#                 skey = -secondary
#             return np.lexsort((skey, pkey))

#     def edge_weight_from_primary(primary: np.ndarray, primary_mode: str) -> np.ndarray:
#         """
#         Convert the primary score to a monotone edge weight where larger is better.
#         Keeps things numerically tame.
#         """
#         s = primary.astype(np.float32)
#         if primary_mode == "logloss":
#             # lower logloss is better -> weight = exp(-logloss) in (0,1]
#             # subtract min for stability
#             s0 = s - s.min()
#             w = np.exp(-s0)
#         else:
#             # higher is better; shift to nonnegative then normalize-ish
#             s0 = s - s.min()
#             # avoid all-zeros
#             w = s0 / (s0.max() + eps) if s0.max() > 0 else np.ones_like(s0)
#         return w.astype(np.float32)

#     # ---------- build CS edges ----------
#     edge_src, edge_dst, edge_attr = [], [], []

#     for dest_id in range(n_total):
#         bundle = neighbors_by_dest.get(dest_id)

#         # --- debug stats ---
#         if bundle is None:
#             cnt_no_bundle += 1
#             continue
#         neigh_idx = bundle["idx"]
#         neigh_w   = bundle["w"]
#         if neigh_idx.size == 0:
#             cnt_empty_bundle += 1
#             continue

#         cnt_num_neigh.append(int(neigh_idx.size))

#         if debug and (dest_id % debug_every == 0) and (max_debug_dests > 0):
#             neigh_labels = y_train[neigh_idx]
#             uniq, counts = np.unique(neigh_labels, return_counts=True)
#             print(f"[CS][dest={dest_id}] K={neigh_idx.size} w_sum={neigh_w.sum():.4f} "
#                 f"label_counts={dict(zip(uniq.tolist(), counts.tolist()))}")
#         # --- debug stats ---

#         if bundle is None:
#             continue
#         neigh_idx = bundle["idx"]
#         neigh_w   = bundle["w"]
#         if neigh_idx.size == 0:
#             continue

#         primary = compute_score(score_mode, neigh_idx, neigh_w)  # [M]

#         # --- debug stats ---
#         if debug and (dest_id % debug_every == 0) and (max_debug_dests > 0):
#             p = primary
#             print(f"[CS][dest={dest_id}] primary({score_mode}) stats: "
#                 f"min={p.min():.6f} med={np.median(p):.6f} max={p.max():.6f} "
#                 f"std={p.std():.6f}")
#         # --- debug stats ---

#         if tie_break_mode is None:
#             secondary = None
#             sec_mode  = None
#         else:
#             secondary = compute_score(tie_break_mode, neigh_idx, neigh_w)  # [M]
#             sec_mode  = tie_break_mode

#         # --- debug stats ---
#         if secondary is not None and debug and (dest_id % debug_every == 0) and (max_debug_dests > 0):
#             s = secondary
#             print(f"[CS][dest={dest_id}] secondary({sec_mode}) stats: "
#                 f"min={s.min():.6f} med={np.median(s):.6f} max={s.max():.6f} "
#                 f"std={s.std():.6f}")
#         # --- debug stats ---


#         # --- tie diagnostics (ADD THIS HERE) ---
#         uniq = np.unique(primary)
#         if uniq.size < primary.size:
#             cnt_ties_primary += 1
#         # --- end tie diagnostics ---

#         order = order_from_scores(primary, score_mode, secondary, sec_mode)
#         keep = order[:min(int(top_k), M)]

#         # --- ties among top-k (ADD THIS TOO) ---
#         vals_keep = primary[keep]
#         if np.unique(vals_keep).size < vals_keep.size:
#             cnt_ties_primary_in_keep += 1
#         # --- end ---

#         # edge weights from primary score (monotone)
#         w_primary = edge_weight_from_primary(primary, score_mode)

#         # --- debug stats ---
#         if score_mode != "logloss":
#             s0 = primary - primary.min()
#             if float(s0.max()) == 0.0:
#                 cnt_s0max0 += 1

#         kept_w = w_primary[keep]
#         if np.allclose(kept_w, 1.0):
#             cnt_all_one_weights += 1

#         if debug and (dest_id % debug_every == 0) and (max_debug_dests > 0):
#             print(f"[CS][dest={dest_id}] keep={keep.tolist()}")
#             print(f"[CS][dest={dest_id}] primary_keep={primary[keep]}")
#             if secondary is not None:
#                 print(f"[CS][dest={dest_id}] secondary_keep={secondary[keep]}")
#             print(f"[CS][dest={dest_id}] w_keep={kept_w} | "
#                 f"frac_w_eq1={np.mean(np.isclose(kept_w, 1.0)):.2f} | "
#                 f"s0max={(primary-primary.min()).max():.6f}")
#             max_debug_dests -= 1
#         # --- end debug stats ---

#         for clf_id in keep.tolist():
#             edge_src.append(int(clf_id))
#             edge_dst.append(int(dest_id))
#             edge_attr.append(float(w_primary[clf_id]))

#     if not edge_src:
#         return (np.zeros((2, 0), dtype=np.int64),
#                 np.zeros((0,), dtype=np.float32))

#     edge_index = np.vstack([np.asarray(edge_src, dtype=np.int64),
#                             np.asarray(edge_dst, dtype=np.int64)])
#     edge_attr = np.asarray(edge_attr, dtype=np.float32)

#     if debug:
#         cnt_num_neigh_arr = np.asarray(cnt_num_neigh, dtype=np.int64)
#         print(
#             f"[CS][summary] dests_with_neighbors={cnt_num_neigh_arr.size} "
#             f"no_bundle={cnt_no_bundle} empty_bundle={cnt_empty_bundle} "
#             f"K: min={cnt_num_neigh_arr.min() if cnt_num_neigh_arr.size else -1} "
#             f"med={np.median(cnt_num_neigh_arr) if cnt_num_neigh_arr.size else -1} "
#             f"max={cnt_num_neigh_arr.max() if cnt_num_neigh_arr.size else -1}"
#         )
#         if score_mode != "logloss":
#             print(f"[CS][summary] s0.max==0 (all-primary-equal) count={cnt_s0max0}")
#         print(f"[CS][summary] kept weights all==1 count={cnt_all_one_weights}")
#         print(f"[CS][summary] primary ties (any) count={cnt_ties_primary} | ties_in_keep count={cnt_ties_primary_in_keep}")
#         # global edge_attr distribution quick check
#         print(f"[CS][summary] edge_attr: min={edge_attr.min():.4f} med={np.median(edge_attr):.4f} "
#             f"max={edge_attr.max():.4f} frac_eq1={np.mean(np.isclose(edge_attr,1.0)):.3f}")
        

#     return edge_index, edge_attr

def build_cs_edges(
    *,
    tr_meta_labels: np.ndarray,
    decision_all: np.ndarray,
    y_train: np.ndarray,
    ss_edge_index: np.ndarray,
    ss_edge_attr: np.ndarray,
    n_train: int,
    n_total: int,
    num_classes: int,
    top_k: int = 3,
    log_tie_break_usage: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified CS edges that select classifiers by highest f1 (local hard accuracy) with f2 as tie-breaker.

    Unlike the original build_cs_edges, this version retains only the f1 scalar as edge_attr and picks
    the top_k classifiers per destination purely on f1 (descending) and f2 (descending) as tie breaker.
    """
    tr_meta_labels = _to_numpy(tr_meta_labels, np.uint8)
    decision_all = _to_numpy(decision_all, np.float32)
    y_train = _to_numpy(y_train, np.int64)
    ss_edge_index = _to_numpy(ss_edge_index, np.int64)
    ss_edge_attr = _to_numpy(ss_edge_attr, np.float32)

    M, C = int(tr_meta_labels.shape[1]), int(num_classes)
    src, dst = ss_edge_index
    w = ss_edge_attr.astype(np.float32)

    neighbors_by_dest: Dict[int, Dict[str, np.ndarray]] = {}
    for s, d, weight in zip(src.tolist(), dst.tolist(), w.tolist()):
        if s >= n_train:
            continue
        bundle = neighbors_by_dest.setdefault(d, {"idx": [], "w": []})
        bundle["idx"].append(s)
        bundle["w"].append(weight)

    for d, bundle in neighbors_by_dest.items():
        bundle["idx"] = np.asarray(bundle["idx"], dtype=np.int64)
        bundle["w"] = np.asarray(bundle["w"], dtype=np.float32)

    edge_src, edge_dst, edge_attr = [], [], []
    tie_break_total = 0
    dest_with_neighbors = 0
    tie_candidate_fraction_sum = 0.0
    for dest_id in range(n_total):
        bundle = neighbors_by_dest.get(dest_id)
        if bundle is None or bundle["idx"].size == 0:
            continue

        neigh_idx = bundle["idx"]
        neigh_w = bundle["w"]
        neigh_label = y_train[neigh_idx]

        f1 = np.zeros(M, dtype=np.float32)
        f2 = np.zeros(M, dtype=np.float32)
        for clf_id in range(M):
            neigh_correct = tr_meta_labels[neigh_idx, clf_id].astype(np.float32)
            indexes = clf_id * C + neigh_label
            neigh_true_prob = decision_all[neigh_idx, indexes].astype(np.float32)

            f1[clf_id] = float((neigh_w * neigh_correct).sum())
            f2[clf_id] = float((neigh_w * neigh_true_prob).sum())

        order = np.lexsort((-f2, -f1))
        keep = order[:min(top_k, M)]

        for clf_id in keep:
            edge_src.append(int(clf_id))
            edge_dst.append(int(dest_id))
            edge_attr.append(float(f1[clf_id]))

        if log_tie_break_usage:
            dest_with_neighbors += 1
            vals = f1[keep]
            uniq, counts = np.unique(vals, return_counts=True)
            tie_sizes = counts[counts > 1]
            tie_candidate_count = int(tie_sizes.sum())
            if tie_candidate_count > 0:
                tie_break_total += 1
                tie_candidate_fraction_sum += tie_candidate_count / max(keep.size, 1)

    if not edge_src:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0,), dtype=np.float32))

    if log_tie_break_usage:
        total = dest_with_neighbors
        pct = 100.0 * tie_break_total / total if total > 0 else 0.0
        print(f"[CS edges] tie-break used for {pct:.2f}% of destinations ({tie_break_total}/{total}).")
        if tie_break_total > 0:
            avg_pct_candidates = 100.0 * (tie_candidate_fraction_sum / tie_break_total)
            print(f"[CS edges] average percent of kept classifiers involved in ties: {avg_pct_candidates:.2f}%")

    edge_index = np.vstack([np.asarray(edge_src, dtype=np.int64),
                            np.asarray(edge_dst, dtype=np.int64)])
    edge_attr = np.asarray(edge_attr, dtype=np.float32)
    return edge_index, edge_attr


def build_cc_edges(
    y_true: np.ndarray,
    y_pred_matrix: np.ndarray,   # shape [N, M]
    top_k: int = 3,
    threshold: Optional[float] = None,  # if set, connect pairs with (1-DF) >= threshold
    symmetric: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classifier <-> Classifier edges using DESlib's double_fault on TRAIN.
    Weight = 1 - DF(i,j) in [0,1]; larger means more diverse (lower joint error).
    Returns:
        edge_index: (2, E) int64
        edge_attr:  (E,)  float32
    """
    y_true = np.asarray(y_true)
    y_pred_matrix = np.asarray(y_pred_matrix)
    num_samples, num_classifiers = y_pred_matrix.shape

    df = np.zeros((num_classifiers, num_classifiers), dtype=np.float32)
    for i in range(num_classifiers):
        for j in range(i + 1, num_classifiers):
            df_ij = float(double_fault(y_true, y_pred_matrix[:, i], y_pred_matrix[:, j]))
            df[i, j] = df_ij
            df[j, i] = df_ij
    diversity = 1.0 - df
    np.fill_diagonal(diversity, 0.0)

    edges: List[Tuple[int, int]] = []
    weights: List[float] = []

    if threshold is not None:
        mask = diversity >= float(threshold)
        np.fill_diagonal(mask, False)
        src, dst = np.where(mask)
        for s, d in zip(src.tolist(), dst.tolist()):
            # if not symmetric, keep one direction (i->j)
            if not symmetric and s > d:
                continue
            edges.append((int(s), int(d)))
            weights.append(float(diversity[s, d]))
    else:
        # connect top_k diverse neighbors per classifier
        for i in range(num_classifiers):
            row = diversity[i]
            # indices sorted by diversity desc, excluding self
            order = np.argsort(-row)
            order = order[order != i]
            keep = order[:min(top_k, len(order))]
            for j in keep.tolist():
                if not symmetric and i > j:
                    continue
                if row[j] <= 0.0:
                    continue
                edges.append((int(i), int(j)))
                weights.append(float(row[j]))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    src = np.asarray([e[0] for e in edges], dtype=np.int64)
    dst = np.asarray([e[1] for e in edges], dtype=np.int64)
    edge_index = np.vstack([src, dst])              # <-- (2, E), int64
    edge_attr  = np.asarray(weights, dtype=np.float32)
    return edge_index, edge_attr

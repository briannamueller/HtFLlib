from __future__ import annotations

from typing import Tuple

import numpy as np

from des.edge_builders import build_ss_edges_cmdw


def build_ps_edges_cmdw(
    *,
    proto_decision_matrix: np.ndarray,
    proto_labels: np.ndarray,
    sample_decision_matrix: np.ndarray,
    sample_labels: np.ndarray,
    k_per_class: int = 2,
    membership_mode: str = "soft",
    membership_k: int = 7,
    eps: float = 1e-8,
    log_class_edge_stats: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build prototype -> sample edges using CMDW over the decision-space.

    Returns edge_index in prototype/sample index spaces.
    """
    if proto_decision_matrix is None:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    proto_decision_matrix = np.asarray(proto_decision_matrix, dtype=np.float32)
    sample_decision_matrix = np.asarray(sample_decision_matrix, dtype=np.float32)
    proto_labels = np.asarray(proto_labels, dtype=np.int64)
    sample_labels = np.asarray(sample_labels, dtype=np.int64)

    n_proto = int(proto_decision_matrix.shape[0])
    n_sample = int(sample_decision_matrix.shape[0])
    if n_proto == 0 or n_sample == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    combined_ds = np.vstack([proto_decision_matrix, sample_decision_matrix])
    combined_y = np.concatenate([proto_labels, sample_labels])
    source_indices = np.arange(n_proto, dtype=np.int64)
    destination_indices = np.arange(n_proto, n_proto + n_sample, dtype=np.int64)

    edge_index, edge_attr = build_ss_edges_cmdw(
        decision_matrix=combined_ds,
        label_vector=combined_y,
        source_indices=source_indices,
        destination_indices=destination_indices,
        k_per_class=k_per_class,
        membership_mode=membership_mode,
        membership_k=membership_k,
        eps=eps,
        log_class_edge_stats=log_class_edge_stats,
    )

    if edge_index.size == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    edge_index = edge_index.copy()
    edge_index[1] -= n_proto
    return edge_index, edge_attr

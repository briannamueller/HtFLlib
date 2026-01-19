from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected

from des import meta_learner_utils as _base
from des.meta_learner_utils import *  # noqa: F401,F403
from des.meta_learner_utils import _neighborloader_supports_weight_attr, iter_weighted_ss_minibatches


def _clone_args(args: Any) -> SimpleNamespace:
    try:
        payload = vars(args)
    except TypeError:
        payload = dict(args)
    return SimpleNamespace(**payload)


def _remap_metadata(metadata: Tuple[list[str], list[Tuple[str, str, str]]]) -> Tuple[list[str], list[Tuple[str, str, str]]]:
    node_types, edge_types = metadata
    if "prototype" not in node_types or "classifier" in node_types:
        return metadata
    remapped_nodes = ["classifier" if n == "prototype" else n for n in node_types]
    remapped_edges = []
    for src, rel, dst in edge_types:
        src_m = "classifier" if src == "prototype" else src
        dst_m = "classifier" if dst == "prototype" else dst
        if rel == "ps":
            rel = "cs"
        elif rel == "ps_rev":
            rel = "cs_rev"
        elif rel == "pp":
            rel = "cc"
        remapped_edges.append((src_m, rel, dst_m))
    return remapped_nodes, remapped_edges


def _remap_input_dims(input_dims: Dict[str, int]) -> Dict[str, int]:
    if "prototype" not in input_dims or "classifier" in input_dims:
        return dict(input_dims)
    remapped = dict(input_dims)
    remapped["classifier"] = remapped.pop("prototype")
    return remapped


def _remap_data_to_classifier(data: HeteroData) -> HeteroData:
    if "prototype" not in data.node_types or "classifier" in data.node_types:
        return data
    out = HeteroData()

    def _copy_node(src_type: str, dst_type: str) -> None:
        for key in data[src_type].keys():
            out[dst_type][key] = data[src_type][key]

    def _copy_edge(
        src_type: str,
        rel: str,
        dst_type: str,
        out_src: str,
        out_rel: str,
        out_dst: str,
    ) -> None:
        for key in data[(src_type, rel, dst_type)].keys():
            out[(out_src, out_rel, out_dst)][key] = data[(src_type, rel, dst_type)][key]

    _copy_node("sample", "sample")
    _copy_node("prototype", "classifier")

    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        src_m = "classifier" if src == "prototype" else src
        dst_m = "classifier" if dst == "prototype" else dst
        if rel == "ps":
            rel_m = "cs"
        elif rel == "ps_rev":
            rel_m = "cs_rev"
        elif rel == "pp":
            rel_m = "cc"
        else:
            rel_m = rel
        _copy_edge(src, rel, dst, src_m, rel_m, dst_m)

    return out


class PrototypeMetaLearnerWrapper(nn.Module):
    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, data: HeteroData) -> torch.Tensor:
        remapped = _remap_data_to_classifier(data)
        return self.base_model(remapped)


def build_meta_learner(
    args,
    metadata,
    input_dims,
    num_candidates: int,
):
    node_types = metadata[0]
    if "prototype" not in node_types or "classifier" in node_types:
        return _base.build_meta_learner(args, metadata, input_dims, num_candidates)

    remapped_metadata = _remap_metadata(metadata)
    remapped_input_dims = _remap_input_dims(input_dims)
    base_model = _base.build_meta_learner(args, remapped_metadata, remapped_input_dims, num_candidates)
    return PrototypeMetaLearnerWrapper(base_model)


def enforce_bidirectionality(data: HeteroData, bidirectional: bool) -> HeteroData:
    if not bidirectional:
        return data

    if "prototype" not in data.node_types or "classifier" in data.node_types:
        return _base.enforce_bidirectionality(data, bidirectional)

    def _num_nodes(ntype: str) -> int:
        nn_ = getattr(data[ntype], "num_nodes", None)
        if nn_ is not None:
            return int(nn_)
        x = getattr(data[ntype], "x", None)
        return int(x.size(0)) if x is not None else 0

    rel = ("sample", "ss", "sample")
    if rel in data.edge_index_dict:
        ei = data[rel].edge_index
        eattr = getattr(data[rel], "edge_attr", None)
        if eattr is not None:
            ei_ud, eattr_ud = to_undirected(ei, eattr, num_nodes=_num_nodes("sample"))
        else:
            ei_ud = to_undirected(ei, num_nodes=_num_nodes("sample"))
            eattr_ud = None

        train_mask = getattr(data["sample"], "train_mask", None)
        if train_mask is not None:
            train_mask = train_mask.bool()
            keep = train_mask[ei_ud[0]]
            data[rel].edge_index = ei_ud[:, keep]
            if eattr_ud is not None:
                data[rel].edge_attr = eattr_ud[keep]
        else:
            data[rel].edge_index = ei_ud[:, :0]
            if eattr_ud is not None:
                data[rel].edge_attr = eattr_ud[:0]

    for rel in (("prototype", "pp", "prototype"), ("prototype", "cc", "prototype")):
        if rel in data.edge_index_dict:
            ei = data[rel].edge_index
            eattr = getattr(data[rel], "edge_attr", None)
            if eattr is not None:
                ei_ud, eattr_ud = to_undirected(ei, eattr, num_nodes=_num_nodes("prototype"))
                data[rel].edge_index = ei_ud
                data[rel].edge_attr = eattr_ud
            else:
                data[rel].edge_index = to_undirected(ei, num_nodes=_num_nodes("prototype"))

    ps_rel = ("prototype", "ps", "sample")
    sp_rel = ("sample", "ps_rev", "prototype")

    if ps_rel in data.edge_index_dict:
        ps_ei = data[ps_rel].edge_index
        ps_eattr = getattr(data[ps_rel], "edge_attr", None)

        train_mask = getattr(data["sample"], "train_mask", None)
        if train_mask is not None:
            train_mask = train_mask.bool()
            keep = train_mask[ps_ei[1]]
            if keep.any():
                sp_src = ps_ei[1, keep]
                sp_dst = ps_ei[0, keep]
                data[sp_rel].edge_index = torch.stack([sp_src, sp_dst], dim=0)
                if ps_eattr is not None:
                    data[sp_rel].edge_attr = ps_eattr[keep]

    return data


def drop_cc_edges(data: HeteroData) -> HeteroData:
    for rel in (("prototype", "pp", "prototype"), ("prototype", "cc", "prototype")):
        if rel in data.edge_index_dict:
            data[rel].edge_index = data[rel].edge_index[:, :0]
            if hasattr(data[rel], "edge_attr"):
                data[rel].edge_attr = data[rel].edge_attr[:0]
            return data
    return _base.drop_cc_edges(data)

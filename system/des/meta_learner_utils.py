
from __future__ import annotations

import inspect
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv, HGTConv, HANConv
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborLoader
from entmax import entmax_bisect
from des.gatv3 import GATv3Conv


from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor


class HeteroGAT(nn.Module):
    """Lightweight heterograph GAT meta-learner.

    The model produces per-sample logits with one output per candidate classifier.

    Readout modes:
      - pair_decoder=None (or "none"): sample-only head -> logits = Linear(h_sample)  => [N, M]
      - pair_decoder in {"dot","cosine","bilinear","distmult","mlp"}: pairwise scoring
        between sample and classifier embeddings => logits = score(h_sample, h_classifier) => [N, M]

    NOTE: `use_classifier_dot_readout` is deprecated. If True and pair_decoder is None,
    it is treated as pair_decoder="mlp" for backwards compatibility.
    """

    def __init__(
        self,
        metadata,
        input_dims,
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
        use_sample_residual: bool = False,
        use_edge_attr: bool = False,
        # --- New API ---
        pair_decoder: str | None = None,
        pair_mlp_hidden: int | None = None,
        pair_mlp_layers: int = 2,
        pair_chunk_size: int = 256,
        # --- Debugging ---
        debug_classifier_embeddings: bool = True,
        clf_embed_var_thresh: float = 1e-4,
        debug_classifier_embeddings_maxprints: int = 3,
        # --- Old API (deprecated) ---
        use_classifier_dot_readout: bool = False,
        # --- Classifier metadata ---
        num_classifiers: int = 0,
        learned_classifier_attrs: bool = False,
        learned_classifier_attr_dim: int = 16,
    ) -> None:
        super().__init__()

        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.use_sample_residual = bool(use_sample_residual)
        self.use_edge_attr = bool(use_edge_attr)
        self.num_classifiers = int(num_classifiers)

        # Normalize decoder setting
        _pair = None if pair_decoder in (None, "", "none") else str(pair_decoder).lower()
        # Backwards compatibility: old boolean flag implies MLP pair decoder
        if _pair is None and bool(use_classifier_dot_readout):
            _pair = "mlp"
        self.pair_decoder = _pair

        self.pair_chunk_size = max(1, int(pair_chunk_size))
        self.pair_mlp_layers = max(1, int(pair_mlp_layers))
        self.pair_mlp_hidden = int(pair_mlp_hidden) if pair_mlp_hidden is not None else None

        # Debug: detect collapsed/near-identical classifier embeddings
        self.debug_classifier_embeddings = bool(debug_classifier_embeddings)
        self.clf_embed_var_thresh = float(clf_embed_var_thresh)
        self.debug_classifier_embeddings_maxprints = int(debug_classifier_embeddings_maxprints)
        self._debug_classifier_embeddings_prints = 0

        self._learned_classifier_attrs_enabled = bool(learned_classifier_attrs) and int(learned_classifier_attr_dim) > 0
        self.learned_classifier_attr_dim = int(learned_classifier_attr_dim) if self._learned_classifier_attrs_enabled else 0

        print(
            "HeteroGAT readout | "
            f"pair_decoder={self.pair_decoder!r} | "
            f"chunk={self.pair_chunk_size} | "
            f"mlp_layers={self.pair_mlp_layers}"
        )

        # Per-node-type input projections
        self.input_proj = nn.ModuleDict({ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in node_types})

        # GATv2 message passing per edge type
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = GATv2Conv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False,
                    edge_dim=1 if self.use_edge_attr else None,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Default (sample-only) head
        self.sample_head = nn.Linear(hidden_dim, out_dim)

        # Optional learned classifier attributes appended to classifier node features
        if self._learned_classifier_attrs_enabled:
            self.learned_classifier_attrs = nn.Parameter(torch.empty(self.num_classifiers, self.learned_classifier_attr_dim))
            nn.init.xavier_uniform_(self.learned_classifier_attrs)
        else:
            self.learned_classifier_attrs = None

        # Pairwise decoder parameters
        self.bilinear_W = None
        self.distmult_r = None
        self.pair_mlp = None

        if self.pair_decoder is not None:
            if self.num_classifiers <= 0:
                raise ValueError("num_classifiers must be > 0 when using a pairwise decoder.")
            if int(out_dim) != int(self.num_classifiers):
                # In your use-case, out_dim should equal num_classifiers (M)
                raise ValueError(
                    f"When using a pairwise decoder, expected out_dim == num_classifiers, got out_dim={out_dim} and "
                    f"num_classifiers={self.num_classifiers}."
                )

        if self.pair_decoder == "bilinear":
            self.bilinear_W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            nn.init.xavier_uniform_(self.bilinear_W)

        elif self.pair_decoder == "distmult":
            self.distmult_r = nn.Parameter(torch.empty(hidden_dim))
            nn.init.ones_(self.distmult_r)

        elif self.pair_decoder == "mlp":
            # Pair features: [h_s || h_c || h_s*h_c || |h_s-h_c|] => 4*hidden_dim
            in_dim = 4 * hidden_dim
            hidden = self.pair_mlp_hidden if self.pair_mlp_hidden is not None else hidden_dim

            mods: list[nn.Module] = []
            d = in_dim
            # (layers-1) hidden layers, then final scalar layer
            for _ in range(self.pair_mlp_layers - 1):
                mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
                d = hidden
            mods += [nn.Linear(d, 1)]
            self.pair_mlp = nn.Sequential(*mods)

        elif self.pair_decoder in {"dot", "cosine"}:
            # No additional parameters
            pass

        elif self.pair_decoder is None:
            pass

        else:
            raise ValueError(
                f"Unknown pair_decoder='{self.pair_decoder}'. Expected one of "
                "None/'none', 'dot', 'cosine', 'bilinear', 'distmult', 'mlp'."
            )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Returns logits.

        - If pair_decoder is None: logits shape [num_samples, out_dim]
        - Else: logits shape [num_samples, num_classifiers]
        """

        # Input projection (+ optional learned classifier attrs)
        x_dict: dict[str, torch.Tensor] = {}
        for ntype in self.node_types:
            node_x = data[ntype].x
            if ntype == "classifier" and self.learned_classifier_attrs is not None:
                # In full-batch graphs we expect all classifier nodes to be present.
                # In NeighborLoader mini-batches, only a subset of classifier nodes may be sampled.
                attrs = self.learned_classifier_attrs
                if node_x.size(0) != attrs.size(0):
                    idx = None
                    # NeighborLoader provides global node ids in `n_id` for sampled subgraphs.
                    if hasattr(data["classifier"], "n_id"):
                        idx = data["classifier"].n_id
                    else:
                        # Fallbacks if you store an explicit classifier index.
                        for _a in ("clf_idx", "classifier_idx", "clf_id"):
                            if hasattr(data["classifier"], _a):
                                idx = getattr(data["classifier"], _a)
                                break
                    if idx is None:
                        raise ValueError(
                            f"Expected {attrs.size(0)} classifier nodes but graph has {node_x.size(0)}. "
                            "When using learned classifier attrs with NeighborLoader, the batch must include "
                            "data['classifier'].n_id (or an equivalent clf_idx mapping) so we can index the attrs."
                        )
                    if idx.dim() == 2 and idx.size(1) == 1:
                        idx = idx.view(-1)
                    idx = idx.to(dtype=torch.long, device=attrs.device)
                    if idx.numel() != node_x.size(0):
                        raise ValueError(
                            f"Classifier attr index has {idx.numel()} entries but batch has {node_x.size(0)} classifier nodes."
                        )
                    attrs = attrs[idx]
                node_x = torch.cat([node_x, attrs], dim=1)
            x_dict[ntype] = self.input_proj[ntype](node_x)

        sample_residual = x_dict.get("sample", None) if self.use_sample_residual else None

        # Debug checkpoints (actual tensors used in forward)
        _dbg_capture = (
            self.debug_classifier_embeddings
            and (self._debug_classifier_embeddings_prints < self.debug_classifier_embeddings_maxprints)
        )
        clf_states = []
        if _dbg_capture and ("classifier" in x_dict):
            clf_states.append(("proj_actual", x_dict["classifier"].detach()))

        # Message passing
        for layer_idx, conv in enumerate(self.convs):
            if self.use_edge_attr:
                edge_attr_dict = {}
                for edge_type in data.edge_index_dict:
                    edge_attr = getattr(data[edge_type], "edge_attr", None)
                    if edge_attr is None:
                        continue
                    if edge_attr.dim() == 1:
                        edge_attr = edge_attr.view(-1, 1)
                    edge_attr_dict[edge_type] = edge_attr
                x_dict = conv(x_dict, data.edge_index_dict, edge_attr_dict)
            else:
                x_dict = conv(x_dict, data.edge_index_dict)
            if layer_idx != len(self.convs) - 1:
                x_dict = {ntype: self.dropout(self.activation(x)) for ntype, x in x_dict.items()}
            else:
                x_dict = {ntype: self.activation(x) for ntype, x in x_dict.items()}
            if _dbg_capture and ("classifier" in x_dict):
                clf_states.append((f"mp_layer{layer_idx}", x_dict["classifier"].detach()))

        sample_embeddings = x_dict["sample"]
        if sample_residual is not None:
            sample_embeddings = sample_embeddings + sample_residual

        # Default sample-only multi-output head
        if self.pair_decoder is None:
            return self.sample_head(sample_embeddings)

        # Pairwise scoring head
        classifier_embeddings = x_dict["classifier"]
        S = sample_embeddings  # [N, d]
        C = classifier_embeddings  # [M, d]

        # IMPORTANT: Pairwise decoders assume column j corresponds to classifier node index j.
        # If the graph stores an explicit mapping from classifier node -> column index (e.g., clf_idx),
        # reorder classifier embeddings here so logits align with your meta-label column ordering.
        idx = None
        for attr in ("clf_idx", "classifier_idx", "clf_id"):
            if hasattr(data["classifier"], attr):
                idx = getattr(data["classifier"], attr)
                break
        if idx is not None:
            if idx.dim() == 2 and idx.size(1) == 1:
                idx = idx.view(-1)
            idx = idx.to(dtype=torch.long, device=C.device)
            if idx.numel() == C.size(0):
                # Apply only if it looks like a permutation of [0..M-1]
                if int(idx.min().item()) == 0 and int(idx.max().item()) == int(C.size(0) - 1):
                    perm = torch.argsort(idx)
                    C = C[perm]

        # Debug: print similarity stats at multiple checkpoints (original clf_x, projected, final C)
        if (
            self.debug_classifier_embeddings
            and (self._debug_classifier_embeddings_prints < self.debug_classifier_embeddings_maxprints)
            and (C.numel() > 0)
            and (C.size(0) > 1)
        ):
            from typing import Tuple

            def _stats(T: torch.Tensor) -> Tuple[float, float, Tuple[float, float, float, float, float]]:
                """Return (std_mean, mean_offdiag_cos, (min,q25,med,q75,max)) for pairwise cosine."""
                T = T.float()
                std_mean = T.std(dim=0).mean().item()
                Tn = F.normalize(T, p=2, dim=-1)
                sim = Tn @ Tn.t()
                eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
                off = sim[~eye]
                if off.numel() == 0:
                    return std_mean, float("nan"), (float("nan"),) * 5
                mean_offdiag_cos = off.mean().item()
                qs = torch.quantile(
                    off,
                    torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=off.device, dtype=torch.float),
                ).tolist()
                q0, q25, q50, q75, q100 = (float(qs[0]), float(qs[1]), float(qs[2]), float(qs[3]), float(qs[4]))
                return std_mean, mean_offdiag_cos, (q0, q25, q50, q75, q100)

            # 1) Original classifier features (clf_x) as stored in the graph
            with torch.no_grad():
                x_raw = data["classifier"].x

            raw_std, raw_cos, raw_q = (float("nan"), float("nan"), (float("nan"),) * 5)
            if x_raw is not None and x_raw.numel() > 0 and x_raw.size(0) > 1:
                raw_std, raw_cos, raw_q = _stats(x_raw)

            # 2) Projected classifier features (best-effort)
            #    Note: if learned classifier attrs are concatenated upstream, x_raw may not match
            #    the linear's expected in_features. We pad/trim to compute a comparable projection.
            proj_std, proj_cos, proj_q = (float("nan"), float("nan"), (float("nan"),) * 5)
            x_proj = None
            if (
                x_raw is not None
                and hasattr(self, "input_proj")
                and isinstance(self.input_proj, torch.nn.ModuleDict)
                and ("classifier" in self.input_proj)
            ):
                lin = self.input_proj["classifier"]
                if hasattr(lin, "in_features") and x_raw.dim() == 2 and x_raw.size(0) > 1:
                    inF = int(lin.in_features)
                    xr = x_raw
                    if xr.size(1) < inF:
                        pad = torch.zeros((xr.size(0), inF - xr.size(1)), device=xr.device, dtype=xr.dtype)
                        xr_in = torch.cat([xr, pad], dim=1)
                    elif xr.size(1) > inF:
                        xr_in = xr[:, :inF]
                    else:
                        xr_in = xr
                    try:
                        x_proj = lin(xr_in)
                    except Exception:
                        x_proj = None

            if x_proj is not None and x_proj.numel() > 0 and x_proj.size(0) > 1:
                proj_std, proj_cos, proj_q = _stats(x_proj)

            # 3) Final classifier embeddings after message passing (C)
            fin_std, fin_cos, fin_q = _stats(C)

            collapse = (fin_std < self.clf_embed_var_thresh) or (fin_cos > 0.98)

            print(
                "[HeteroGAT][debug] classifier similarity checkpoints: "
                f"clf_x(std_mean={raw_std:.3e}, cos_mean={raw_cos:.4f}, cos[min/q25/med/q75/max]=[{raw_q[0]:.4f}/{raw_q[1]:.4f}/{raw_q[2]:.4f}/{raw_q[3]:.4f}/{raw_q[4]:.4f}]) | "
                f"proj(std_mean={proj_std:.3e}, cos_mean={proj_cos:.4f}, cos[min/q25/med/q75/max]=[{proj_q[0]:.4f}/{proj_q[1]:.4f}/{proj_q[2]:.4f}/{proj_q[3]:.4f}/{proj_q[4]:.4f}]) | "
                f"final(std_mean={fin_std:.3e}, cos_mean={fin_cos:.4f}, cos[min/q25/med/q75/max]=[{fin_q[0]:.4f}/{fin_q[1]:.4f}/{fin_q[2]:.4f}/{fin_q[3]:.4f}/{fin_q[4]:.4f}]) | "
                f"pair_decoder={self.pair_decoder!r}, learned_classifier_attrs={self.learned_classifier_attrs is not None}"
                + (" [COLLAPSE SUSPECTED]" if collapse else "")
            )

            # Increment on every print so this diagnostic is capped
            self._debug_classifier_embeddings_prints += 1

        if self.pair_decoder == "dot":
            return S @ C.t()

        if self.pair_decoder == "cosine":
            S_n = F.normalize(S, p=2, dim=-1)
            C_n = F.normalize(C, p=2, dim=-1)
            return S_n @ C_n.t()

        if self.pair_decoder == "bilinear":
            # score(i,m) = (S_i W) dot C_m
            return (S @ self.bilinear_W) @ C.t()

        if self.pair_decoder == "distmult":
            # score(i,m) = (S_i ⊙ r) dot C_m
            return (S * self.distmult_r) @ C.t()

        if self.pair_decoder == "mlp":
            # Chunk over classifiers to avoid [N, M, *] blowup for large M
            N, d = S.size()
            M = C.size(0)
            out = torch.empty((N, M), device=S.device, dtype=S.dtype)
            chunk = self.pair_chunk_size

            for j in range(0, M, chunk):
                Cj = C[j : j + chunk]  # [cj, d]
                cj = Cj.size(0)

                Sj = S.unsqueeze(1).expand(N, cj, d)
                Cjj = Cj.unsqueeze(0).expand(N, cj, d)

                feats = torch.cat([Sj, Cjj, Sj * Cjj, (Sj - Cjj).abs()], dim=-1)  # [N, cj, 4d]
                out[:, j : j + chunk] = self.pair_mlp(feats).squeeze(-1)
            return out

        raise RuntimeError(f"Unhandled pair_decoder='{self.pair_decoder}'")


class HeteroGATv3(nn.Module):
    """Lightweight heterograph GATv3 meta-learner.

    The model produces per-sample logits with one output per candidate classifier.

    Readout modes:
      - pair_decoder=None (or "none"): sample-only head -> logits = Linear(h_sample)  => [N, M]
      - pair_decoder in {"dot","cosine","bilinear","distmult","mlp"}: pairwise scoring
        between sample and classifier embeddings => logits = score(h_sample, h_classifier) => [N, M]

    NOTE: `use_classifier_dot_readout` is deprecated. If True and pair_decoder is None,
    it is treated as pair_decoder="mlp" for backwards compatibility.
    """

    def __init__(
        self,
        metadata,
        input_dims,
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
        use_sample_residual: bool = False,
        use_edge_attr: bool = False,
        # --- New API ---
        pair_decoder: str | None = None,
        pair_mlp_hidden: int | None = None,
        pair_mlp_layers: int = 2,
        pair_chunk_size: int = 256,
        # --- Debugging ---
        debug_classifier_embeddings: bool = True,
        clf_embed_var_thresh: float = 1e-4,
        debug_classifier_embeddings_maxprints: int = 3,
        # --- Old API (deprecated) ---
        use_classifier_dot_readout: bool = False,
        # --- Classifier metadata ---
        num_classifiers: int = 0,
        learned_classifier_attrs: bool = False,
        learned_classifier_attr_dim: int = 16,
    ) -> None:
        super().__init__()

        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.use_sample_residual = bool(use_sample_residual)
        self.use_edge_attr = bool(use_edge_attr)
        self.num_classifiers = int(num_classifiers)

        # Normalize decoder setting
        _pair = None if pair_decoder in (None, "", "none") else str(pair_decoder).lower()
        # Backwards compatibility: old boolean flag implies MLP pair decoder
        if _pair is None and bool(use_classifier_dot_readout):
            _pair = "mlp"
        self.pair_decoder = _pair

        self.pair_chunk_size = max(1, int(pair_chunk_size))
        self.pair_mlp_layers = max(1, int(pair_mlp_layers))
        self.pair_mlp_hidden = int(pair_mlp_hidden) if pair_mlp_hidden is not None else None

        # Debug: detect collapsed/near-identical classifier embeddings
        self.debug_classifier_embeddings = bool(debug_classifier_embeddings)
        self.clf_embed_var_thresh = float(clf_embed_var_thresh)
        self.debug_classifier_embeddings_maxprints = int(debug_classifier_embeddings_maxprints)
        self._debug_classifier_embeddings_prints = 0

        self._learned_classifier_attrs_enabled = bool(learned_classifier_attrs) and int(learned_classifier_attr_dim) > 0
        self.learned_classifier_attr_dim = int(learned_classifier_attr_dim) if self._learned_classifier_attrs_enabled else 0

        print(
            "HeteroGATv3 readout | "
            f"pair_decoder={self.pair_decoder!r} | "
            f"chunk={self.pair_chunk_size} | "
            f"mlp_layers={self.pair_mlp_layers}"
        )

        # Per-node-type input projections
        self.input_proj = nn.ModuleDict({ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in node_types})

        # GATv3 message passing per edge type
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = GATv3Conv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=1 if self.use_edge_attr else None,
                    name=str(edge_type),
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Default (sample-only) head
        self.sample_head = nn.Linear(hidden_dim, out_dim)

        # Optional learned classifier attributes appended to classifier node features
        if self._learned_classifier_attrs_enabled:
            self.learned_classifier_attrs = nn.Parameter(torch.empty(self.num_classifiers, self.learned_classifier_attr_dim))
            nn.init.xavier_uniform_(self.learned_classifier_attrs)
        else:
            self.learned_classifier_attrs = None

        # Pairwise decoder parameters
        self.bilinear_W = None
        self.distmult_r = None
        self.pair_mlp = None

        if self.pair_decoder is not None:
            if self.num_classifiers <= 0:
                raise ValueError("num_classifiers must be > 0 when using a pairwise decoder.")
            if int(out_dim) != int(self.num_classifiers):
                # In your use-case, out_dim should equal num_classifiers (M)
                raise ValueError(
                    f"When using a pairwise decoder, expected out_dim == num_classifiers, got out_dim={out_dim} and "
                    f"num_classifiers={self.num_classifiers}."
                )

        if self.pair_decoder == "bilinear":
            self.bilinear_W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            nn.init.xavier_uniform_(self.bilinear_W)

        elif self.pair_decoder == "distmult":
            self.distmult_r = nn.Parameter(torch.empty(hidden_dim))
            nn.init.ones_(self.distmult_r)

        elif self.pair_decoder == "mlp":
            # Pair features: [h_s || h_c || h_s*h_c || |h_s-h_c|] => 4*hidden_dim
            in_dim = 4 * hidden_dim
            hidden = self.pair_mlp_hidden if self.pair_mlp_hidden is not None else hidden_dim

            mods: list[nn.Module] = []
            d = in_dim
            # (layers-1) hidden layers, then final scalar layer
            for _ in range(self.pair_mlp_layers - 1):
                mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
                d = hidden
            mods += [nn.Linear(d, 1)]
            self.pair_mlp = nn.Sequential(*mods)

        elif self.pair_decoder in {"dot", "cosine"}:
            # No additional parameters
            pass

        elif self.pair_decoder is None:
            pass

        else:
            raise ValueError(
                f"Unknown pair_decoder='{self.pair_decoder}'. Expected one of "
                "None/'none', 'dot', 'cosine', 'bilinear', 'distmult', 'mlp'."
            )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Returns logits.

        - If pair_decoder is None: logits shape [num_samples, out_dim]
        - Else: logits shape [num_samples, num_classifiers]
        """

        # Input projection (+ optional learned classifier attrs)
        x_dict: dict[str, torch.Tensor] = {}
        for ntype in self.node_types:
            node_x = data[ntype].x
            if ntype == "classifier" and self.learned_classifier_attrs is not None:
                # In full-batch graphs we expect all classifier nodes to be present.
                # In NeighborLoader mini-batches, only a subset of classifier nodes may be sampled.
                attrs = self.learned_classifier_attrs
                if node_x.size(0) != attrs.size(0):
                    idx = None
                    # NeighborLoader provides global node ids in `n_id` for sampled subgraphs.
                    if hasattr(data["classifier"], "n_id"):
                        idx = data["classifier"].n_id
                    else:
                        # Fallbacks if you store an explicit classifier index.
                        for _a in ("clf_idx", "classifier_idx", "clf_id"):
                            if hasattr(data["classifier"], _a):
                                idx = getattr(data["classifier"], _a)
                                break
                    if idx is None:
                        raise ValueError(
                            f"Expected {attrs.size(0)} classifier nodes but graph has {node_x.size(0)}. "
                            "When using learned classifier attrs with NeighborLoader, the batch must include "
                            "data['classifier'].n_id (or an equivalent clf_idx mapping) so we can index the attrs."
                        )
                    if idx.dim() == 2 and idx.size(1) == 1:
                        idx = idx.view(-1)
                    idx = idx.to(dtype=torch.long, device=attrs.device)
                    if idx.numel() != node_x.size(0):
                        raise ValueError(
                            f"Classifier attr index has {idx.numel()} entries but batch has {node_x.size(0)} classifier nodes."
                        )
                    attrs = attrs[idx]
                node_x = torch.cat([node_x, attrs], dim=1)
            x_dict[ntype] = self.input_proj[ntype](node_x)

        sample_residual = x_dict.get("sample", None) if self.use_sample_residual else None

        # Debug checkpoints (actual tensors used in forward)
        _dbg_capture = (
            self.debug_classifier_embeddings
            and (self._debug_classifier_embeddings_prints < self.debug_classifier_embeddings_maxprints)
        )
        clf_states = []
        if _dbg_capture and ("classifier" in x_dict):
            clf_states.append(("proj_actual", x_dict["classifier"].detach()))

        # Message passing
        for layer_idx, conv in enumerate(self.convs):
            if self.use_edge_attr:
                edge_attr_dict = {}
                for edge_type in data.edge_index_dict:
                    edge_attr = getattr(data[edge_type], "edge_attr", None)
                    if edge_attr is None:
                        continue
                    if edge_attr.dim() == 1:
                        edge_attr = edge_attr.view(-1, 1)
                    edge_attr_dict[edge_type] = edge_attr
                x_dict = conv(x_dict, data.edge_index_dict, edge_attr_dict)
            else:
                x_dict = conv(x_dict, data.edge_index_dict)
            if layer_idx != len(self.convs) - 1:
                x_dict = {ntype: self.dropout(self.activation(x)) for ntype, x in x_dict.items()}
            else:
                x_dict = {ntype: self.activation(x) for ntype, x in x_dict.items()}
            if _dbg_capture and ("classifier" in x_dict):
                clf_states.append((f"mp_layer{layer_idx}", x_dict["classifier"].detach()))

        sample_embeddings = x_dict["sample"]
        if sample_residual is not None:
            sample_embeddings = sample_embeddings + sample_residual

        # Default sample-only multi-output head
        if self.pair_decoder is None:
            return self.sample_head(sample_embeddings)

        # Pairwise scoring head
        classifier_embeddings = x_dict["classifier"]
        S = sample_embeddings  # [N, d]
        C = classifier_embeddings  # [M, d]

        # IMPORTANT: Pairwise decoders assume column j corresponds to classifier node index j.
        # If the graph stores an explicit mapping from classifier node -> column index (e.g., clf_idx),
        # reorder classifier embeddings here so logits align with your meta-label column ordering.
        idx = None
        for attr in ("clf_idx", "classifier_idx", "clf_id"):
            if hasattr(data["classifier"], attr):
                idx = getattr(data["classifier"], attr)
                break
        if idx is not None:
            if idx.dim() == 2 and idx.size(1) == 1:
                idx = idx.view(-1)
            idx = idx.to(dtype=torch.long, device=C.device)
            if idx.numel() == C.size(0):
                # Apply only if it looks like a permutation of [0..M-1]
                if int(idx.min().item()) == 0 and int(idx.max().item()) == int(C.size(0) - 1):
                    perm = torch.argsort(idx)
                    C = C[perm]

        # Debug: print similarity stats at multiple checkpoints (original clf_x, projected, final C)
        if (
            self.debug_classifier_embeddings
            and (self._debug_classifier_embeddings_prints < self.debug_classifier_embeddings_maxprints)
            and (C.numel() > 0)
            and (C.size(0) > 1)
        ):
            from typing import Tuple

            def _stats(T: torch.Tensor) -> Tuple[float, float, Tuple[float, float, float, float, float]]:
                """Return (std_mean, mean_offdiag_cos, (min,q25,med,q75,max)) for pairwise cosine."""
                T = T.float()
                std_mean = T.std(dim=0).mean().item()
                Tn = F.normalize(T, p=2, dim=-1)
                sim = Tn @ Tn.t()
                eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
                off = sim[~eye]
                if off.numel() == 0:
                    return std_mean, float("nan"), (float("nan"),) * 5
                mean_offdiag_cos = off.mean().item()
                qs = torch.quantile(
                    off,
                    torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=off.device, dtype=torch.float),
                ).tolist()
                q0, q25, q50, q75, q100 = (float(qs[0]), float(qs[1]), float(qs[2]), float(qs[3]), float(qs[4]))
                return std_mean, mean_offdiag_cos, (q0, q25, q50, q75, q100)

            # 1) Original classifier features (clf_x) as stored in the graph
            with torch.no_grad():
                x_raw = data["classifier"].x

            raw_std, raw_cos, raw_q = (float("nan"), float("nan"), (float("nan"),) * 5)
            if x_raw is not None and x_raw.numel() > 0 and x_raw.size(0) > 1:
                raw_std, raw_cos, raw_q = _stats(x_raw)

            # 2) Projected classifier features (best-effort)
            #    Note: if learned classifier attrs are concatenated upstream, x_raw may not match
            #    the linear's expected in_features. We pad/trim to compute a comparable projection.
            proj_std, proj_cos, proj_q = (float("nan"), float("nan"), (float("nan"),) * 5)
            x_proj = None
            if (
                x_raw is not None
                and hasattr(self, "input_proj")
                and isinstance(self.input_proj, torch.nn.ModuleDict)
                and ("classifier" in self.input_proj)
            ):
                lin = self.input_proj["classifier"]
                if hasattr(lin, "in_features") and x_raw.dim() == 2 and x_raw.size(0) > 1:
                    inF = int(lin.in_features)
                    xr = x_raw
                    if xr.size(1) < inF:
                        pad = torch.zeros((xr.size(0), inF - xr.size(1)), device=xr.device, dtype=xr.dtype)
                        xr_in = torch.cat([xr, pad], dim=1)
                    elif xr.size(1) > inF:
                        xr_in = xr[:, :inF]
                    else:
                        xr_in = xr
                    try:
                        x_proj = lin(xr_in)
                    except Exception:
                        x_proj = None

            if x_proj is not None and x_proj.numel() > 0 and x_proj.size(0) > 1:
                proj_std, proj_cos, proj_q = _stats(x_proj)

            # 3) Final classifier embeddings after message passing (C)
            fin_std, fin_cos, fin_q = _stats(C)

            collapse = (fin_std < self.clf_embed_var_thresh) or (fin_cos > 0.98)

            print(
                "[HeteroGATv3][debug] classifier similarity checkpoints: "
                f"clf_x(std_mean={raw_std:.3e}, cos_mean={raw_cos:.4f}, cos[min/q25/med/q75/max]=[{raw_q[0]:.4f}/{raw_q[1]:.4f}/{raw_q[2]:.4f}/{raw_q[3]:.4f}/{raw_q[4]:.4f}]) | "
                f"proj(std_mean={proj_std:.3e}, cos_mean={proj_cos:.4f}, cos[min/q25/med/q75/max]=[{proj_q[0]:.4f}/{proj_q[1]:.4f}/{proj_q[2]:.4f}/{proj_q[3]:.4f}/{proj_q[4]:.4f}]) | "
                f"final(std_mean={fin_std:.3e}, cos_mean={fin_cos:.4f}, cos[min/q25/med/q75/max]=[{fin_q[0]:.4f}/{fin_q[1]:.4f}/{fin_q[2]:.4f}/{fin_q[3]:.4f}/{fin_q[4]:.4f}]) | "
                f"pair_decoder={self.pair_decoder!r}, learned_classifier_attrs={self.learned_classifier_attrs is not None}"
                + (" [COLLAPSE SUSPECTED]" if collapse else "")
            )

            # Increment on every print so this diagnostic is capped
            self._debug_classifier_embeddings_prints += 1

        if self.pair_decoder == "dot":
            return S @ C.t()

        if self.pair_decoder == "cosine":
            S_n = F.normalize(S, p=2, dim=-1)
            C_n = F.normalize(C, p=2, dim=-1)
            return S_n @ C_n.t()

        if self.pair_decoder == "bilinear":
            # score(i,m) = (S_i W) dot C_m
            return (S @ self.bilinear_W) @ C.t()

        if self.pair_decoder == "distmult":
            # score(i,m) = (S_i ⊙ r) dot C_m
            return (S * self.distmult_r) @ C.t()

        if self.pair_decoder == "mlp":
            # Chunk over classifiers to avoid [N, M, *] blowup for large M
            N, d = S.size()
            M = C.size(0)
            out = torch.empty((N, M), device=S.device, dtype=S.dtype)
            chunk = self.pair_chunk_size

            for j in range(0, M, chunk):
                Cj = C[j : j + chunk]  # [cj, d]
                cj = Cj.size(0)

                Sj = S.unsqueeze(1).expand(N, cj, d)
                Cjj = Cj.unsqueeze(0).expand(N, cj, d)

                feats = torch.cat([Sj, Cjj, Sj * Cjj, (Sj - Cjj).abs()], dim=-1)  # [N, cj, 4d]
                out[:, j : j + chunk] = self.pair_mlp(feats).squeeze(-1)
            return out

        raise RuntimeError(f"Unhandled pair_decoder='{self.pair_decoder}'")


class GAT(nn.Module):
    """Homogeneous GAT that only uses the S-S edge relation."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
        use_sample_residual: bool = False,
        use_edge_attr: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.use_sample_residual = bool(use_sample_residual)
        self.use_edge_attr = bool(use_edge_attr)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList(
            GATv2Conv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=1 if self.use_edge_attr else None,
            )
            for _ in range(num_layers)
        )
        self.sample_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: HeteroData) -> torch.Tensor:
        x = self.input_proj(data["sample"].x)
        sample_residual = x if self.use_sample_residual else None
        edge_index = data[("sample", "ss", "sample")].edge_index
        edge_attr = None
        if self.use_edge_attr:
            edge_attr = getattr(data[("sample", "ss", "sample")], "edge_attr", None)
            if edge_attr is not None and edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if layer_idx != len(self.convs) - 1:
                x = self.dropout(self.activation(x))
            else:
                x = self.activation(x)
        if sample_residual is not None:
            x = x + sample_residual
        return self.sample_head(x)


class GATv3(nn.Module):
    """Homogeneous GATv3 that only uses the S-S edge relation."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
        use_sample_residual: bool = False,
        use_edge_attr: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.use_sample_residual = bool(use_sample_residual)
        self.use_edge_attr = bool(use_edge_attr)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList(
            GATv3Conv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                edge_dim=1 if self.use_edge_attr else None,
                name="sample_ss",
            )
            for _ in range(num_layers)
        )
        self.sample_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: HeteroData) -> torch.Tensor:
        x = self.input_proj(data["sample"].x)
        sample_residual = x if self.use_sample_residual else None
        edge_index = data[("sample", "ss", "sample")].edge_index
        edge_attr = None
        if self.use_edge_attr:
            edge_attr = getattr(data[("sample", "ss", "sample")], "edge_attr", None)
            if edge_attr is not None and edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if layer_idx != len(self.convs) - 1:
                x = self.dropout(self.activation(x))
            else:
                x = self.activation(x)
        if sample_residual is not None:
            x = x + sample_residual
        return self.sample_head(x)


class HeteroHGT(nn.Module):
    """Heterogeneous meta-learner using HGTConv.

    Produces per-sample logits of shape [num_samples, out_dim].
    """

    def __init__(
        self,
        metadata,
        input_dims,
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
        use_sample_residual: bool = False,
        # --- classifier metadata ---
        num_classifiers: int = 0,
        learned_classifier_attrs: bool = False,
        learned_classifier_attr_dim: int = 16,
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.use_sample_residual = bool(use_sample_residual)
        self.num_classifiers = int(num_classifiers)

        self._learned_classifier_attrs_enabled = bool(learned_classifier_attrs) and int(learned_classifier_attr_dim) > 0
        self.learned_classifier_attr_dim = int(learned_classifier_attr_dim) if self._learned_classifier_attrs_enabled else 0

        # Per-node-type input projection to common hidden_dim
        self.input_proj = nn.ModuleDict(
            {ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in node_types}
        )

        # Optional learned classifier attributes appended to classifier node features
        if self._learned_classifier_attrs_enabled:
            if self.num_classifiers <= 0:
                raise ValueError("num_classifiers must be > 0 when using learned classifier attributes.")
            self.learned_classifier_attrs = nn.Parameter(torch.empty(self.num_classifiers, self.learned_classifier_attr_dim))
            nn.init.xavier_uniform_(self.learned_classifier_attrs)
        else:
            self.learned_classifier_attrs = None

        # Stack of HGTConv layers
        self.convs = nn.ModuleList(
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=heads,
            )
            for _ in range(num_layers)
        )

        self.sample_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {}
        for ntype in self.node_types:
            node_x = data[ntype].x
            if ntype == "classifier" and self.learned_classifier_attrs is not None:
                # Full-batch: classifier nodes == num_classifiers.
                # Mini-batch (NeighborLoader): only a subset of classifier nodes may be sampled.
                attrs = self.learned_classifier_attrs
                if node_x.size(0) != attrs.size(0):
                    idx = None
                    if hasattr(data["classifier"], "n_id"):
                        idx = data["classifier"].n_id
                    else:
                        for _a in ("clf_idx", "classifier_idx", "clf_id"):
                            if hasattr(data["classifier"], _a):
                                idx = getattr(data["classifier"], _a)
                                break
                    if idx is None:
                        raise ValueError(
                            f"Expected {attrs.size(0)} classifier nodes but graph has {node_x.size(0)}. "
                            "When using learned classifier attrs with NeighborLoader, the batch must include "
                            "data['classifier'].n_id (or an equivalent clf_idx mapping) so we can index the attrs."
                        )
                    if idx.dim() == 2 and idx.size(1) == 1:
                        idx = idx.view(-1)
                    idx = idx.to(dtype=torch.long, device=attrs.device)
                    if idx.numel() != node_x.size(0):
                        raise ValueError(
                            f"Classifier attr index has {idx.numel()} entries but batch has {node_x.size(0)} classifier nodes."
                        )
                    attrs = attrs[idx]
                node_x = torch.cat([node_x, attrs], dim=1)
            x_dict[ntype] = self.input_proj[ntype](node_x)

        sample_residual = x_dict.get("sample", None) if self.use_sample_residual else None

        for layer_idx, conv in enumerate(self.convs):
            x_dict = conv(x_dict, data.edge_index_dict)
            if layer_idx != len(self.convs) - 1:
                x_dict = {ntype: self.dropout(self.activation(x)) for ntype, x in x_dict.items()}
            else:
                x_dict = {ntype: self.activation(x) for ntype, x in x_dict.items()}

        sample_embeddings = x_dict["sample"]
        if sample_residual is not None:
            sample_embeddings = sample_embeddings + sample_residual
        return self.sample_head(sample_embeddings)


class HeteroHAN(nn.Module):
    """Heterogeneous meta-learner using HANConv.

    Produces per-sample logits of shape [num_samples, out_dim].
    """

    def __init__(
        self,
        metadata,
        input_dims,
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
        use_sample_residual: bool = False,
        # --- classifier metadata ---
        num_classifiers: int = 0,
        learned_classifier_attrs: bool = False,
        learned_classifier_attr_dim: int = 16,
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.use_sample_residual = bool(use_sample_residual)
        self.num_classifiers = int(num_classifiers)

        self._learned_classifier_attrs_enabled = bool(learned_classifier_attrs) and int(learned_classifier_attr_dim) > 0
        self.learned_classifier_attr_dim = int(learned_classifier_attr_dim) if self._learned_classifier_attrs_enabled else 0

        self.input_proj = nn.ModuleDict(
            {ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in node_types}
        )

        # Optional learned classifier attributes appended to classifier node features
        if self._learned_classifier_attrs_enabled:
            if self.num_classifiers <= 0:
                raise ValueError("num_classifiers must be > 0 when using learned classifier attributes.")
            self.learned_classifier_attrs = nn.Parameter(torch.empty(self.num_classifiers, self.learned_classifier_attr_dim))
            nn.init.xavier_uniform_(self.learned_classifier_attrs)
        else:
            self.learned_classifier_attrs = None

        self.convs = nn.ModuleList(
            HANConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )

        self.sample_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {}
        for ntype in self.node_types:
            node_x = data[ntype].x
            if ntype == "classifier" and self.learned_classifier_attrs is not None:
                # Full-batch: classifier nodes == num_classifiers.
                # Mini-batch (NeighborLoader): only a subset of classifier nodes may be sampled.
                attrs = self.learned_classifier_attrs
                if node_x.size(0) != attrs.size(0):
                    idx = None
                    if hasattr(data["classifier"], "n_id"):
                        idx = data["classifier"].n_id
                    else:
                        for _a in ("clf_idx", "classifier_idx", "clf_id"):
                            if hasattr(data["classifier"], _a):
                                idx = getattr(data["classifier"], _a)
                                break
                    if idx is None:
                        raise ValueError(
                            f"Expected {attrs.size(0)} classifier nodes but graph has {node_x.size(0)}. "
                            "When using learned classifier attrs with NeighborLoader, the batch must include "
                            "data['classifier'].n_id (or an equivalent clf_idx mapping) so we can index the attrs."
                        )
                    if idx.dim() == 2 and idx.size(1) == 1:
                        idx = idx.view(-1)
                    idx = idx.to(dtype=torch.long, device=attrs.device)
                    if idx.numel() != node_x.size(0):
                        raise ValueError(
                            f"Classifier attr index has {idx.numel()} entries but batch has {node_x.size(0)} classifier nodes."
                        )
                    attrs = attrs[idx]
                node_x = torch.cat([node_x, attrs], dim=1)
            x_dict[ntype] = self.input_proj[ntype](node_x)

        sample_residual = x_dict.get("sample", None) if self.use_sample_residual else None

        for layer_idx, conv in enumerate(self.convs):
            x_dict = conv(x_dict, data.edge_index_dict)
            if layer_idx != len(self.convs) - 1:
                x_dict = {ntype: self.dropout(self.activation(x)) for ntype, x in x_dict.items()}
            else:
                x_dict = {ntype: self.activation(x) for ntype, x in x_dict.items()}

        sample_embeddings = x_dict["sample"]
        if sample_residual is not None:
            sample_embeddings = sample_embeddings + sample_residual
        return self.sample_head(sample_embeddings)

class _SAGEAttnBlock(nn.Module):
    """GraphSAGE-style update with an attention aggregator.

    We compute an attention-weighted neighbor summary (via GATv2Conv), compute a self projection,
    then combine them with a SAGE-like CONCAT + Linear.

    This is intentionally lightweight and works in full-batch and sampled mini-batch settings.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int = 4,
        dropout: float = 0.1,
        use_edge_attr: bool = False,
    ) -> None:
        super().__init__()
        self.attn = GATv2Conv(
            in_dim,
            out_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            add_self_loops=False,
            edge_dim=1 if use_edge_attr else None,
        )
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.combine = nn.Linear(2 * out_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        neigh = self.attn(x, edge_index, edge_attr=edge_attr)  # [N, out_dim]
        self_msg = self.self_lin(x)                   # [N, out_dim]
        out = self.combine(torch.cat([self_msg, neigh], dim=-1))
        return out


class SAGEAttn(nn.Module):
    """Homogeneous GraphSAGE+attention meta-learner.

    Uses only the Sample-Sample (S-S) relation: ('sample','ss','sample').

    Output: per-sample logits of shape [num_sample_nodes, out_dim].

    Intended for DES settings where you want neighbor sampling during training (NeighborLoader),
    but can still run full-batch.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
        use_sample_residual: bool = False,
        use_edge_attr: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.use_sample_residual = bool(use_sample_residual)
        self.use_edge_attr = bool(use_edge_attr)

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            _SAGEAttnBlock(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                heads=heads,
                dropout=dropout,
                use_edge_attr=self.use_edge_attr,
            )
            for _ in range(num_layers)
        )

        self.sample_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: HeteroData) -> torch.Tensor:
        # data may be a full graph OR a sampled subgraph from NeighborLoader.
        x = self.input_proj(data["sample"].x)
        sample_residual = x if self.use_sample_residual else None

        edge_index = data[("sample", "ss", "sample")].edge_index
        edge_attr = None
        if self.use_edge_attr:
            edge_attr = getattr(data[("sample", "ss", "sample")], "edge_attr", None)
            if edge_attr is not None and edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)

        for layer_idx, block in enumerate(self.blocks):
            x = block(x, edge_index, edge_attr=edge_attr)
            if layer_idx != len(self.blocks) - 1:
                x = self.dropout(self.activation(x))
            else:
                x = self.activation(x)

        if sample_residual is not None:
            x = x + sample_residual

        return self.sample_head(x)


# --------------------------------------------------------------------------
# Heterogeneous SAGE+Attention meta-learner (HeteroSAGEAttn)
# --------------------------------------------------------------------------
class HeteroSAGEAttn(nn.Module):
    """Heterogeneous GraphSAGE+attention meta-learner.

    This is the hetero analogue of `SAGEAttn`.

    Key feature: `update_classifier_nodes` controls whether classifier nodes are updated.
      - If False: classifier nodes only influence sample nodes via (classifier, cs, sample) edges.
      - If True: classifier nodes are updated as well via optional (sample, cs_rev, classifier)
                 and (classifier, cc, classifier) edges.

    Readout:
      - pair_decoder=None/'none': logits = Linear(h_sample) -> [N_samples, out_dim]
      - pair_decoder in {"dot","cosine","bilinear","distmult","mlp"}: score sample vs classifier
        embeddings -> [N_samples, num_classifiers] (requires `num_classifiers == out_dim`).

    Supports learned classifier attributes (`gnn_learned_cc_attrs`) for identifiability.
    """

    def __init__(
        self,
        metadata,
        input_dims,
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
        use_sample_residual: bool = False,
        use_edge_attr: bool = False,
        # --- behavior ---
        update_classifier_nodes: bool = False,
        # --- readout ---
        pair_decoder: str | None = None,
        pair_mlp_hidden: int | None = None,
        pair_mlp_layers: int = 2,
        pair_chunk_size: int = 256,
        # --- classifier metadata ---
        num_classifiers: int = 0,
        learned_classifier_attrs: bool = False,
        learned_classifier_attr_dim: int = 16,
    ) -> None:
        super().__init__()

        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.use_sample_residual = bool(use_sample_residual)
        self.use_edge_attr = bool(use_edge_attr)
        self.update_classifier_nodes = bool(update_classifier_nodes)

        self.num_classifiers = int(num_classifiers)

        # Normalize decoder setting
        _pair = None if pair_decoder in (None, "", "none") else str(pair_decoder).lower()
        self.pair_decoder = _pair

        self.pair_chunk_size = max(1, int(pair_chunk_size))
        self.pair_mlp_layers = max(1, int(pair_mlp_layers))
        self.pair_mlp_hidden = int(pair_mlp_hidden) if pair_mlp_hidden is not None else None

        self._learned_classifier_attrs_enabled = bool(learned_classifier_attrs) and int(learned_classifier_attr_dim) > 0
        self.learned_classifier_attr_dim = int(learned_classifier_attr_dim) if self._learned_classifier_attrs_enabled else 0

        # Per-node-type input projections to common hidden_dim
        self.input_proj = nn.ModuleDict({ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in node_types})

        # Optional learned classifier attributes appended to classifier node features
        if self._learned_classifier_attrs_enabled:
            if self.num_classifiers <= 0:
                raise ValueError("num_classifiers must be > 0 when using learned classifier attributes.")
            self.learned_classifier_attrs = nn.Parameter(torch.empty(self.num_classifiers, self.learned_classifier_attr_dim))
            nn.init.xavier_uniform_(self.learned_classifier_attrs)
        else:
            self.learned_classifier_attrs = None

        # Relation-specific attention aggregators (neighbor summaries)
        # We create convs only for relations we might use.
        # The model will skip any relation not present in the input graph.
        self.convs = nn.ModuleList()
        for _ in range(int(num_layers)):
            conv_dict = nn.ModuleDict()
            # sample <- sample
            conv_dict["ss"] = GATv2Conv(
                (hidden_dim, hidden_dim),
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=1 if self.use_edge_attr else None,
            )
            # sample <- classifier
            conv_dict["cs"] = GATv2Conv(
                (hidden_dim, hidden_dim),
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=1 if self.use_edge_attr else None,
            )
            # classifier <- sample (only used if update_classifier_nodes)
            conv_dict["cs_rev"] = GATv2Conv(
                (hidden_dim, hidden_dim),
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=1 if self.use_edge_attr else None,
            )
            # classifier <- classifier (only used if update_classifier_nodes)
            conv_dict["cc"] = GATv2Conv(
                (hidden_dim, hidden_dim),
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=1 if self.use_edge_attr else None,
            )
            self.convs.append(conv_dict)

        # SAGE-style self projections and combine layers per node type we may update
        self.self_lin = nn.ModuleDict()
        self.combine = nn.ModuleDict()

        # sample always updated
        self.self_lin["sample"] = nn.Linear(hidden_dim, hidden_dim)
        self.combine["sample"] = nn.Linear(2 * hidden_dim, hidden_dim)

        # classifier updated only if requested
        if self.update_classifier_nodes:
            self.self_lin["classifier"] = nn.Linear(hidden_dim, hidden_dim)
            self.combine["classifier"] = nn.Linear(2 * hidden_dim, hidden_dim)

        # Default (sample-only) head
        self.sample_head = nn.Linear(hidden_dim, out_dim)

        # Pairwise decoder parameters (mirrors HeteroGAT)
        self.bilinear_W = None
        self.distmult_r = None
        self.pair_mlp = None

        if self.pair_decoder is not None:
            if self.num_classifiers <= 0:
                raise ValueError("num_classifiers must be > 0 when using a pairwise decoder.")
            if int(out_dim) != int(self.num_classifiers):
                raise ValueError(
                    f"When using a pairwise decoder, expected out_dim == num_classifiers, got out_dim={out_dim} and "
                    f"num_classifiers={self.num_classifiers}."
                )

        if self.pair_decoder == "bilinear":
            self.bilinear_W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            nn.init.xavier_uniform_(self.bilinear_W)

        elif self.pair_decoder == "distmult":
            self.distmult_r = nn.Parameter(torch.empty(hidden_dim))
            nn.init.ones_(self.distmult_r)

        elif self.pair_decoder == "mlp":
            in_dim = 4 * hidden_dim
            hidden = self.pair_mlp_hidden if self.pair_mlp_hidden is not None else hidden_dim
            mods: list[nn.Module] = []
            d = in_dim
            for _ in range(self.pair_mlp_layers - 1):
                mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
                d = hidden
            mods += [nn.Linear(d, 1)]
            self.pair_mlp = nn.Sequential(*mods)

        elif self.pair_decoder in {"dot", "cosine"}:
            pass

        elif self.pair_decoder is None:
            pass

        else:
            raise ValueError(
                f"Unknown pair_decoder='{self.pair_decoder}'. Expected one of "
                "None/'none', 'dot', 'cosine', 'bilinear', 'distmult', 'mlp'."
            )

    def _maybe_reorder_classifier_embeddings(self, data: HeteroData, C: torch.Tensor) -> torch.Tensor:
        """Best-effort: reorder classifier embeddings so column j aligns with meta-label column j."""
        idx = None
        for attr in ("clf_idx", "classifier_idx", "clf_id"):
            if hasattr(data["classifier"], attr):
                idx = getattr(data["classifier"], attr)
                break
        if idx is None:
            return C
        if idx.dim() == 2 and idx.size(1) == 1:
            idx = idx.view(-1)
        idx = idx.to(dtype=torch.long, device=C.device)
        if idx.numel() != C.size(0):
            return C
        if int(idx.min().item()) == 0 and int(idx.max().item()) == int(C.size(0) - 1):
            perm = torch.argsort(idx)
            return C[perm]
        return C

    def forward(self, data: HeteroData) -> torch.Tensor:
        # Input projection (+ optional learned classifier attrs)
        x_dict: dict[str, torch.Tensor] = {}
        for ntype in self.node_types:
            node_x = data[ntype].x
            if ntype == "classifier" and self.learned_classifier_attrs is not None:
                # Full-batch: classifier nodes == num_classifiers.
                # Mini-batch (NeighborLoader): only a subset of classifier nodes may be sampled.
                attrs = self.learned_classifier_attrs
                if node_x.size(0) != attrs.size(0):
                    idx = None
                    if hasattr(data["classifier"], "n_id"):
                        idx = data["classifier"].n_id
                    else:
                        for _a in ("clf_idx", "classifier_idx", "clf_id"):
                            if hasattr(data["classifier"], _a):
                                idx = getattr(data["classifier"], _a)
                                break
                    if idx is None:
                        raise ValueError(
                            f"Expected {attrs.size(0)} classifier nodes but graph has {node_x.size(0)}. "
                            "When using learned classifier attrs with NeighborLoader, the batch must include "
                            "data['classifier'].n_id (or an equivalent clf_idx mapping) so we can index the attrs."
                        )
                    if idx.dim() == 2 and idx.size(1) == 1:
                        idx = idx.view(-1)
                    idx = idx.to(dtype=torch.long, device=attrs.device)
                    if idx.numel() != node_x.size(0):
                        raise ValueError(
                            f"Classifier attr index has {idx.numel()} entries but batch has {node_x.size(0)} classifier nodes."
                        )
                    attrs = attrs[idx]
                node_x = torch.cat([node_x, attrs], dim=1)
            x_dict[ntype] = self.input_proj[ntype](node_x)

        sample_residual = x_dict.get("sample", None) if self.use_sample_residual else None

        # Message passing (hetero SAGE-style update)
        for layer_idx, conv_dict in enumerate(self.convs):
            x_new: dict[str, torch.Tensor] = {}

            # --- update SAMPLE ---
            s = x_dict.get("sample", None)
            if s is None:
                raise KeyError("HeteroSAGEAttn requires node type 'sample'.")

            neigh_sum = torch.zeros_like(s)

            # sample <- sample (ss)
            rel_ss = ("sample", "ss", "sample")
            if rel_ss in data.edge_index_dict:
                ei = data[rel_ss].edge_index
                edge_attr = None
                if self.use_edge_attr:
                    edge_attr = getattr(data[rel_ss], "edge_attr", None)
                    if edge_attr is not None and edge_attr.dim() == 1:
                        edge_attr = edge_attr.view(-1, 1)
                neigh_sum = neigh_sum + conv_dict["ss"]((s, s), ei, edge_attr=edge_attr)

            # sample <- classifier (cs)
            rel_cs = ("classifier", "cs", "sample")
            if ("classifier" in x_dict) and (rel_cs in data.edge_index_dict):
                c = x_dict["classifier"]
                ei = data[rel_cs].edge_index
                edge_attr = None
                if self.use_edge_attr:
                    edge_attr = getattr(data[rel_cs], "edge_attr", None)
                    if edge_attr is not None and edge_attr.dim() == 1:
                        edge_attr = edge_attr.view(-1, 1)
                neigh_sum = neigh_sum + conv_dict["cs"]((c, s), ei, edge_attr=edge_attr)

            self_msg = self.self_lin["sample"](s)
            out_s = self.combine["sample"](torch.cat([self_msg, neigh_sum], dim=-1))
            x_new["sample"] = out_s

            # --- update CLASSIFIER (optional) ---
            if self.update_classifier_nodes and ("classifier" in x_dict):
                c = x_dict["classifier"]
                neigh_c = torch.zeros_like(c)

                # classifier <- classifier (cc)
                rel_cc = ("classifier", "cc", "classifier")
                if rel_cc in data.edge_index_dict:
                    ei = data[rel_cc].edge_index
                    edge_attr = None
                    if self.use_edge_attr:
                        edge_attr = getattr(data[rel_cc], "edge_attr", None)
                        if edge_attr is not None and edge_attr.dim() == 1:
                            edge_attr = edge_attr.view(-1, 1)
                    neigh_c = neigh_c + conv_dict["cc"]((c, c), ei, edge_attr=edge_attr)

                # classifier <- sample (cs_rev)
                rel_sc = ("sample", "cs_rev", "classifier")
                if rel_sc in data.edge_index_dict:
                    ei = data[rel_sc].edge_index
                    edge_attr = None
                    if self.use_edge_attr:
                        edge_attr = getattr(data[rel_sc], "edge_attr", None)
                        if edge_attr is not None and edge_attr.dim() == 1:
                            edge_attr = edge_attr.view(-1, 1)
                    neigh_c = neigh_c + conv_dict["cs_rev"]((s, c), ei, edge_attr=edge_attr)

                self_c = self.self_lin["classifier"](c)
                out_c = self.combine["classifier"](torch.cat([self_c, neigh_c], dim=-1))
                x_new["classifier"] = out_c
            elif "classifier" in x_dict:
                # frozen classifier embeddings (still used for cs into samples and/or pairwise readout)
                x_new["classifier"] = x_dict["classifier"]

            # activation/dropout
            if layer_idx != len(self.convs) - 1:
                x_dict = {nt: self.dropout(self.activation(xx)) for nt, xx in x_new.items()}
            else:
                x_dict = {nt: self.activation(xx) for nt, xx in x_new.items()}

        sample_embeddings = x_dict["sample"]
        if sample_residual is not None:
            sample_embeddings = sample_embeddings + sample_residual

        # Default sample-only multi-output head
        if self.pair_decoder is None:
            return self.sample_head(sample_embeddings)

        # Pairwise scoring head
        if "classifier" not in x_dict:
            raise KeyError("pair_decoder requires node type 'classifier' in the graph.")

        classifier_embeddings = x_dict["classifier"]
        S = sample_embeddings
        C = self._maybe_reorder_classifier_embeddings(data, classifier_embeddings)

        if self.pair_decoder == "dot":
            return S @ C.t()

        if self.pair_decoder == "cosine":
            S_n = F.normalize(S, p=2, dim=-1)
            C_n = F.normalize(C, p=2, dim=-1)
            return S_n @ C_n.t()

        if self.pair_decoder == "bilinear":
            return (S @ self.bilinear_W) @ C.t()

        if self.pair_decoder == "distmult":
            return (S * self.distmult_r) @ C.t()

        if self.pair_decoder == "mlp":
            N, d = S.size()
            M = C.size(0)
            out = torch.empty((N, M), device=S.device, dtype=S.dtype)
            chunk = self.pair_chunk_size

            for j in range(0, M, chunk):
                Cj = C[j : j + chunk]
                cj = Cj.size(0)

                Sj = S.unsqueeze(1).expand(N, cj, d)
                Cjj = Cj.unsqueeze(0).expand(N, cj, d)

                feats = torch.cat([Sj, Cjj, Sj * Cjj, (Sj - Cjj).abs()], dim=-1)
                out[:, j : j + chunk] = self.pair_mlp(feats).squeeze(-1)
            return out

        raise RuntimeError(f"Unhandled pair_decoder='{self.pair_decoder}'")

def build_meta_learner(
    args,
    metadata,
    input_dims,
    num_candidates: int,
):
    """Construct a meta-learner GNN configured from args and graph metadata.

    Supports args.gnn_arch in {"hetero_gat", "gat", "hgt", "han"}.

    New readout config (hetero_gat only):
      - args.gnn_pair_decoder in {None/"none", "dot", "cosine", "bilinear", "distmult", "mlp"}
      - args.gnn_pair_mlp_hidden (optional)
      - args.gnn_pair_mlp_layers (default 2)
      - args.gnn_pair_chunk_size (default 256)

"""

    learned_classifier_attrs = bool(getattr(args, "gnn_learned_cc_attrs", False))
    learned_classifier_attr_dim = int(getattr(args, "gnn_learned_cc_attr_dim", 64))

    input_dims = dict(input_dims)
    if learned_classifier_attrs and learned_classifier_attr_dim > 0:
        input_dims["classifier"] = input_dims.get("classifier", 0) + learned_classifier_attr_dim

    # Resolve architecture
    arch = getattr(args, "gnn_arch", None)
    if arch is None or arch == "":
        arch = "hetero_gat"
    arch = str(arch).lower()

    hidden_dim = getattr(args, "gnn_hidden_dim", 128)
    num_layers = getattr(args, "gnn_layers", 2)
    heads = getattr(args, "gnn_heads", 4)
    dropout = getattr(args, "gnn_dropout", 0.1)
    use_sample_residual = bool(getattr(args, "gnn_use_sample_residual", False))
    use_edge_attr = bool(getattr(args, "gnn_use_edge_attr", False))

    # Homogeneous GAT (sample-only graph)
    if arch in {"gat", "homog_gat", "homogeneous_gat"}:
        sample_input_dim = input_dims.get("sample", 0)
        return GAT(
            input_dim=sample_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            out_dim=num_candidates,
            use_sample_residual=use_sample_residual,
            use_edge_attr=use_edge_attr,
        )
    if arch in {"gatv3"}:
        sample_input_dim = input_dims.get("sample", 0)
        return GATv3(
            input_dim=sample_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            out_dim=num_candidates,
            use_sample_residual=use_sample_residual,
            use_edge_attr=use_edge_attr,
        )
    if arch in {"sage", "graphsage", "sage_attn", "sage+attn"}:
        sample_input_dim = input_dims.get("sample", 0)
        return SAGEAttn(
            input_dim=sample_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            out_dim=num_candidates,
            use_sample_residual=use_sample_residual,
            use_edge_attr=use_edge_attr,
        )

    # Heterogeneous SAGE+attention
    if arch in {"hetero_sage", "hetero_sage_attn", "sage_attn_hetero", "sage_hetero"}:
        pair_decoder = getattr(args, "gnn_pair_decoder", None)
        if pair_decoder is not None:
            pair_decoder = None if str(pair_decoder).lower() in {"", "none"} else str(pair_decoder).lower()

        # NOTE: With pairwise decoders, the model must be able to distinguish classifier nodes.
        # If classifier node features are identical/weak, dot/cosine/bilinear/distmult/mlp can
        # collapse to near-constant scores. Enabling learned classifier attributes provides a
        # per-classifier ID embedding and makes the pairwise head identifiable.
        if (pair_decoder is not None) and (not learned_classifier_attrs):
            learned_classifier_attrs = True
            if learned_classifier_attr_dim <= 0:
                learned_classifier_attr_dim = 16
            input_dims["classifier"] = input_dims.get("classifier", 0) + learned_classifier_attr_dim
            print(
                "[meta_learner_utils] pair_decoder is set but gnn_learned_cc_attrs is False; "
                "enabling learned classifier attributes for identifiability."
            )

        pair_mlp_hidden = getattr(args, "gnn_pair_mlp_hidden", None)
        pair_mlp_layers = int(getattr(args, "gnn_pair_mlp_layers", 2))
        pair_chunk_size = int(getattr(args, "gnn_pair_chunk_size", 256))

        update_classifier_nodes = bool(getattr(args, "gnn_update_classifier_nodes", False))

        return HeteroSAGEAttn(
            metadata=metadata,
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            out_dim=num_candidates,
            use_sample_residual=use_sample_residual,
            use_edge_attr=use_edge_attr,
            update_classifier_nodes=update_classifier_nodes,
            pair_decoder=pair_decoder,
            pair_mlp_hidden=pair_mlp_hidden,
            pair_mlp_layers=pair_mlp_layers,
            pair_chunk_size=pair_chunk_size,
            num_classifiers=num_candidates,
            learned_classifier_attrs=learned_classifier_attrs,
            learned_classifier_attr_dim=learned_classifier_attr_dim,
        )
    # Heterogeneous variants below
    if arch in {"hetero_gatv3"}:
        pair_decoder = getattr(args, "gnn_pair_decoder", None)
        if pair_decoder is not None:
            pair_decoder = None if str(pair_decoder).lower() in {"", "none"} else str(pair_decoder).lower()

        # NOTE: With pairwise decoders, the model must be able to distinguish classifier nodes.
        # If classifier node features are identical/weak, dot/cosine/bilinear/distmult/mlp can
        # collapse to near-constant scores. Enabling learned classifier attributes provides a
        # per-classifier ID embedding and makes the pairwise head identifiable.
        if (pair_decoder is not None) and (not learned_classifier_attrs):
            learned_classifier_attrs = True
            if learned_classifier_attr_dim <= 0:
                learned_classifier_attr_dim = 16
            input_dims["classifier"] = input_dims.get("classifier", 0) + learned_classifier_attr_dim
            print(
                "[meta_learner_utils] pair_decoder is set but gnn_learned_cc_attrs is False; "
                "enabling learned classifier attributes for identifiability."
            )

        pair_mlp_hidden = getattr(args, "gnn_pair_mlp_hidden", None)
        pair_mlp_layers = int(getattr(args, "gnn_pair_mlp_layers", 2))
        pair_chunk_size = int(getattr(args, "gnn_pair_chunk_size", 256))
        debug_classifier_embeddings = bool(getattr(args, "gnn_debug_classifier_embeddings", True))
        clf_embed_var_thresh = float(getattr(args, "gnn_clf_embed_var_thresh", 1e-4))
        debug_classifier_embeddings_maxprints = int(getattr(args, "gnn_debug_classifier_embeddings_maxprints", 3))

        return HeteroGATv3(
            metadata=metadata,
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            out_dim=num_candidates,
            use_sample_residual=use_sample_residual,
            use_edge_attr=use_edge_attr,
            pair_decoder=pair_decoder,
            pair_mlp_hidden=pair_mlp_hidden,
            pair_mlp_layers=pair_mlp_layers,
            pair_chunk_size=pair_chunk_size,
            debug_classifier_embeddings=debug_classifier_embeddings,
            clf_embed_var_thresh=clf_embed_var_thresh,
            debug_classifier_embeddings_maxprints=debug_classifier_embeddings_maxprints,
            # keep accepting the old arg in case something instantiates HeteroGATv3 directly
            num_classifiers=num_candidates,
            learned_classifier_attrs=learned_classifier_attrs,
            learned_classifier_attr_dim=learned_classifier_attr_dim,
        )
    if arch in {"hetero_gat", "gatv2", "hetero"}:
        pair_decoder = getattr(args, "gnn_pair_decoder", None)
        if pair_decoder is not None:
            pair_decoder = None if str(pair_decoder).lower() in {"", "none"} else str(pair_decoder).lower()

        # NOTE: With pairwise decoders, the model must be able to distinguish classifier nodes.
        # If classifier node features are identical/weak, dot/cosine/bilinear/distmult/mlp can
        # collapse to near-constant scores. Enabling learned classifier attributes provides a
        # per-classifier ID embedding and makes the pairwise head identifiable.
        if (pair_decoder is not None) and (not learned_classifier_attrs):
            learned_classifier_attrs = True
            if learned_classifier_attr_dim <= 0:
                learned_classifier_attr_dim = 16
            input_dims["classifier"] = input_dims.get("classifier", 0) + learned_classifier_attr_dim
            print(
                "[meta_learner_utils] pair_decoder is set but gnn_learned_cc_attrs is False; "
                "enabling learned classifier attributes for identifiability."
            )

        pair_mlp_hidden = getattr(args, "gnn_pair_mlp_hidden", None)
        pair_mlp_layers = int(getattr(args, "gnn_pair_mlp_layers", 2))
        pair_chunk_size = int(getattr(args, "gnn_pair_chunk_size", 256))
        debug_classifier_embeddings = bool(getattr(args, "gnn_debug_classifier_embeddings", True))
        clf_embed_var_thresh = float(getattr(args, "gnn_clf_embed_var_thresh", 1e-4))
        debug_classifier_embeddings_maxprints = int(getattr(args, "gnn_debug_classifier_embeddings_maxprints", 3))

        return HeteroGAT(
            metadata=metadata,
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            out_dim=num_candidates,
            use_sample_residual=use_sample_residual,
            use_edge_attr=use_edge_attr,
            pair_decoder=pair_decoder,
            pair_mlp_hidden=pair_mlp_hidden,
            pair_mlp_layers=pair_mlp_layers,
            pair_chunk_size=pair_chunk_size,
            debug_classifier_embeddings=debug_classifier_embeddings,
            clf_embed_var_thresh=clf_embed_var_thresh,
            debug_classifier_embeddings_maxprints=debug_classifier_embeddings_maxprints,
            # keep accepting the old arg in case something instantiates HeteroGAT directly
            num_classifiers=num_candidates,
            learned_classifier_attrs=learned_classifier_attrs,
            learned_classifier_attr_dim=learned_classifier_attr_dim,
        )

    if arch in {"hgt", "hetero_hgt"}:
        return HeteroHGT(
            metadata=metadata,
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            out_dim=num_candidates,
            use_sample_residual=use_sample_residual,
            num_classifiers=num_candidates,
            learned_classifier_attrs=learned_classifier_attrs,
            learned_classifier_attr_dim=learned_classifier_attr_dim,
        )

    if arch in {"han", "hetero_han"}:
        return HeteroHAN(
            metadata=metadata,
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            out_dim=num_candidates,
            use_sample_residual=use_sample_residual,
            num_classifiers=num_candidates,
            learned_classifier_attrs=learned_classifier_attrs,
            learned_classifier_attr_dim=learned_classifier_attr_dim,
        )

    raise ValueError(
        f"Unknown gnn_arch='{arch}'. Expected one of ['hetero_gat', 'hetero_gatv3', 'gat', 'gatv3', "
        "'sage_attn', 'hetero_sage_attn', 'hgt', 'han']."
    )



def drop_cc_edges(data):
    """Remove classifier-classifier edges from the heterograph so the GNN never sees that relation."""
    rel = ("classifier", "cc", "classifier")
    if rel in data.edge_index_dict:
        del data.edge_index_dict[rel]
        if rel in data:
            del data[rel]
    return data


# NEW: drop_classifier_nodes helper for SAGE-style training
def drop_classifier_nodes(data: HeteroData) -> HeteroData:
    """Return a sample-only HeteroData graph.

    This is useful for homogeneous meta-learners (e.g., GraphSAGE/SAGEAttn) that
    only use the Sample-Sample relation and should not trigger NeighborLoader to
    expect neighbor counts for classifier-related edge types.

    Keeps:
      - Node type: "sample" (all attributes, including masks)
      - Edge type: ("sample", "ss", "sample") (edge_index and optional edge_attr)

    Drops:
      - Node type: "classifier"
      - All non-SS edge types (e.g., cs, cs_rev, cc)
    """

    out = HeteroData()

    # Copy all sample node attributes (x, masks, etc.)
    if "sample" not in data.node_types:
        raise KeyError("drop_classifier_nodes expected node type 'sample' in the input graph.")

    for key in data["sample"].keys():
        out["sample"][key] = data["sample"][key]

    # Copy sample-sample relation
    rel = ("sample", "ss", "sample")
    if rel not in data.edge_index_dict:
        raise KeyError(f"drop_classifier_nodes expected edge type {rel} in the input graph.")

    out[rel].edge_index = data[rel].edge_index
    if "edge_attr" in data[rel]:
        out[rel].edge_attr = data[rel].edge_attr

    return out


def compute_sample_weights(client, labels: torch.Tensor, meta_labels: torch.Tensor, mode: str):
    """Compute per-sample weights for the meta learner.

    mode: "class_prevalence" (inverse class frequency) or "difficulty" (1 - frac_correct).
    """

    mode = str(mode).lower()
    if mode == "class_prevalence":
        class_counts = torch.bincount(labels.detach().cpu(), minlength=client.args.num_classes).float()
        class_counts = class_counts.to(labels.device).clamp(min=1.0)
        inv = (class_counts.sum() / class_counts.numel()) / class_counts
        return inv[labels]
    if mode == "difficulty":
        frac_correct = meta_labels.float().mean(dim=1)
        return (1.0 - frac_correct).clamp(min=0.0)
    return None

def enforce_bidirectionality(data: HeteroData, bidirectional: bool) -> HeteroData:
    """If bidirectional is True, make selected relations undirected while ensuring
    val/test sample nodes only receive messages (no outgoing edges from them).

    - Sample-Sample: undirect, then keep edges with train samples as sources.
    - Classifier-Classifier: undirect fully.
    - Classifier->Sample (CS): add reverse Sample->Classifier edges, but ONLY for TRAIN sample nodes.
      (KEY NOTE: val/test sample nodes must NOT send messages to classifier nodes.)
    """

    if not bidirectional:
        return data

    def _num_nodes(ntype: str) -> int:
        nn_ = getattr(data[ntype], "num_nodes", None)
        if nn_ is not None:
            return int(nn_)
        x = getattr(data[ntype], "x", None)
        return int(x.size(0)) if x is not None else 0

    # Sample-Sample
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
            keep = train_mask[ei_ud[0]]  # only TRAIN as sources
            data[rel].edge_index = ei_ud[:, keep]
            if eattr_ud is not None:
                data[rel].edge_attr = eattr_ud[keep]
        else:
            # fail closed
            data[rel].edge_index = ei_ud[:, :0]
            if eattr_ud is not None:
                data[rel].edge_attr = eattr_ud[:0]

    # Classifier-Classifier
    rel = ("classifier", "cc", "classifier")
    if rel in data.edge_index_dict:
        ei = data[rel].edge_index
        eattr = getattr(data[rel], "edge_attr", None)
        if eattr is not None:
            ei_ud, eattr_ud = to_undirected(ei, eattr, num_nodes=_num_nodes("classifier"))
            data[rel].edge_index = ei_ud
            data[rel].edge_attr = eattr_ud
        else:
            data[rel].edge_index = to_undirected(ei, num_nodes=_num_nodes("classifier"))

    # Classifier->Sample (CS): add reverse Sample->Classifier edges, TRAIN ONLY
    cs_rel = ("classifier", "cs", "sample")
    sc_rel = ("sample", "cs_rev", "classifier")

    if cs_rel in data.edge_index_dict:
        cs_ei = data[cs_rel].edge_index  # [2, E] rows: [clf_id, sample_id]
        cs_eattr = getattr(data[cs_rel], "edge_attr", None)

        train_mask = getattr(data["sample"], "train_mask", None)
        if train_mask is not None:
            train_mask = train_mask.bool()
            keep = train_mask[cs_ei[1]]  # only edges to TRAIN sample nodes
            if keep.any():
                sc_src = cs_ei[1, keep]  # sample ids (train only)
                sc_dst = cs_ei[0, keep]  # classifier ids
                data[sc_rel].edge_index = torch.stack([sc_src, sc_dst], dim=0)
                if cs_eattr is not None:
                    data[sc_rel].edge_attr = cs_eattr[keep]

    return data

def _topk_indices_from_logits(logits: torch.Tensor, k: int | None, alpha: float):
    """Return indices [N, k] of top-k classifiers per sample based on entmax weights."""
    if k is None or k <= 0 or k >= logits.size(1):
        return None
    with torch.no_grad():
        w = entmax_bisect(logits, alpha=alpha, dim=-1)  # [N, M]
        _, idx = torch.topk(w, k=k, dim=1, largest=True, sorted=False)
    return idx


def _logdet_gram(V: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """V: [d, k]. Returns logdet(V^T V + eps I) (scalar)."""
    k = V.size(1)
    G = V.transpose(0, 1) @ V  # [k, k]
    G = G + eps * torch.eye(k, device=V.device, dtype=V.dtype)
    sign, logabsdet = torch.linalg.slogdet(G)
    return torch.where(sign > 0, logabsdet, torch.full_like(logabsdet, -1e6))


def _div_penalty_from_logdet(logdet: torch.Tensor) -> torch.Tensor:
    """Convert logdet to a penalty via 1 - determinant."""
    det = torch.exp(logdet)
    return 1.0 - det


def compute_gnn_diversity_loss(
    logits: torch.Tensor,
    ds: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    *,
    entmax_alpha: float = 1.5,
    top_k: int | None = None,
    eps: float = 1e-6,
    ss_edge_index_train: torch.Tensor | None = None,
    binary_neighbor_k_cap: int = 25,
) -> torch.Tensor:
    """Compute a diversity regularization loss for the HetGAT meta-learner.

    This returns a *loss term to ADD* to the training objective (1 - determinant,
    so it ranges in [0, 1] and minimizing it encourages higher diversity).

    Multi-class (C>2): Pang-style non-maximal probability diversity (per-sample).
    Binary (C==2): logit-response diversity over a local neighborhood in the sample-sample graph.

    Args:
      logits: [N, M] meta-learner outputs for N training samples.
      ds:     [N, M*C] flattened per-class probabilities from the frozen base classifiers.
      y:      [N] true labels for the N training samples (needed for multi-class non-max removal).
      num_classes: C.
      ss_edge_index_train: optional [2, E] sample-sample edges *in train-local indexing*
                           (only used for binary). If None in binary, raises.
      binary_neighbor_k_cap: cap on neighborhood size to bound cost.

    Returns:
      diversity_loss: scalar tensor
    """

    assert logits.dim() == 2, f"logits must be [N,M], got {logits.shape}"
    N, M = logits.shape
    C = int(num_classes)
    assert ds.shape[0] == N, "ds and logits must align on N"
    assert ds.shape[1] == M * C, f"ds second dim must be M*C={M*C}, got {ds.shape[1]}"

    alpha = float(entmax_alpha)
    idx = _topk_indices_from_logits(logits, top_k, alpha=alpha)  # [N,k] or None
    ds_3d = ds.view(N, M, C)

    if C > 2:
        losses = []
        for i in range(N):
            if idx is None:
                sel = torch.arange(M, device=logits.device)
            else:
                sel = idx[i]
            Pi = ds_3d[i, sel, :]  # [k, C]
            yi = int(y[i].item())
            mask = torch.ones(C, device=Pi.device, dtype=torch.bool)
            mask[yi] = False
            nonmax = Pi[:, mask]  # [k, C-1]
            V = nonmax.transpose(0, 1).contiguous()  # [C-1, k]
            V = F.normalize(V, p=2, dim=0)
            losses.append(_div_penalty_from_logdet(_logdet_gram(V, eps=eps)))
        return torch.stack(losses).mean()

    # Binary case: C == 2
    if ss_edge_index_train is None:
        raise ValueError("Binary diversity requires ss_edge_index_train (train-local indexing).")

    p_pos = ds_3d[:, :, 1].clamp(min=1e-6, max=1 - 1e-6)
    z = torch.log(p_pos) - torch.log1p(-p_pos)  # logit(p): [N, M]

    src = ss_edge_index_train[0]
    dst = ss_edge_index_train[1]

    neighbors = [[] for _ in range(N)]
    for s, d in zip(src.tolist(), dst.tolist()):
        if 0 <= s < N and 0 <= d < N:
            neighbors[s].append(d)

    losses = []
    for i in range(N):
        neigh = neighbors[i]
        if len(neigh) == 0:
            continue
        if len(neigh) > binary_neighbor_k_cap:
            neigh = neigh[:binary_neighbor_k_cap]
        neigh_idx = torch.tensor([i] + neigh, device=logits.device, dtype=torch.long)  # [t]
        Zi = z[neigh_idx, :]  # [t, M]

        if idx is None:
            sel = torch.arange(M, device=logits.device)
        else:
            sel = idx[i]
        R = Zi[:, sel]  # [t, k]
        V = R.contiguous()  # [t, k]
        V = F.normalize(V, p=2, dim=0)
        losses.append(_div_penalty_from_logdet(_logdet_gram(V, eps=eps)))
    if len(losses) == 0:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


#
# Remove duplicate import block and ensure REL_SS is defined once for helpers below.

REL_SS = ("sample", "ss", "sample")

def _neighborloader_supports_weight_attr() -> bool:
    try:
        sig = inspect.signature(NeighborLoader.__init__)
        return "weight_attr" in sig.parameters
    except Exception:
        return False

# def _build_ss_adjacency(edge_index: torch.Tensor,
#                         edge_weight: torch.Tensor | None,
#                         num_nodes: int):
#     """Build per-source adjacency lists for weighted sampling (CPU tensors)."""
#     ei = edge_index.detach().cpu()
#     src = ei[0].long()
#     dst = ei[1].long()

#     if edge_weight is None:
#         w = torch.ones(src.numel(), dtype=torch.float)
#     else:
#         w = edge_weight.detach().view(-1).cpu().float().clamp(min=0.0)

#     order = torch.argsort(src)
#     src, dst, w = src[order], dst[order], w[order]

#     nbrs: list[torch.Tensor | None] = [None] * int(num_nodes)
#     wts:  list[torch.Tensor | None] = [None] * int(num_nodes)

#     if src.numel() == 0:
#         return nbrs, wts

#     uniq, counts = torch.unique_consecutive(src, return_counts=True)
#     ptr = 0
#     for u, c in zip(uniq.tolist(), counts.tolist()):
#         u = int(u); c = int(c)
#         nbrs[u] = dst[ptr:ptr+c]
#         wts[u]  = w[ptr:ptr+c]
#         ptr += c

#     return nbrs, wts

def _build_ss_adjacency(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    allowed_mask: torch.Tensor | None = None,
):
    """Build per-source adjacency lists for weighted sampling (CPU tensors).

    If `allowed_mask` is provided (bool tensor [num_nodes]), we ONLY keep edges
    where BOTH endpoints are allowed. This guarantees we never traverse into
    disallowed nodes (e.g., val/test) when training on train seeds.

    Returns:
      nbrs:  List[List[int]] where nbrs[u] are destination nodes v.
      probs: List[torch.Tensor] where probs[u] are normalized sampling probs over nbrs[u].
    """
    ei = edge_index.detach().cpu()
    src = ei[0].long()
    dst = ei[1].long()

    w = None
    if edge_weight is not None:
        w = edge_weight.detach().cpu().float().view(-1)

    if allowed_mask is not None:
        am = allowed_mask.detach().cpu().bool()
        if am.numel() != num_nodes:
            raise ValueError(f"allowed_mask must have length num_nodes={num_nodes}, got {am.numel()}")
        keep = am[src] & am[dst]
        src = src[keep]
        dst = dst[keep]
        if w is not None:
            w = w[keep]

    nbrs: list[list[int]] = [[] for _ in range(num_nodes)]
    wlists: list[list[float]] = [[] for _ in range(num_nodes)]

    if src.numel() == 0:
        probs: list[torch.Tensor] = [torch.empty((0,), dtype=torch.float) for _ in range(num_nodes)]
        return nbrs, probs

    if w is None:
        for s, d in zip(src.tolist(), dst.tolist()):
            if 0 <= s < num_nodes and 0 <= d < num_nodes:
                nbrs[s].append(d)
                wlists[s].append(1.0)
    else:
        for s, d, ww in zip(src.tolist(), dst.tolist(), w.tolist()):
            if 0 <= s < num_nodes and 0 <= d < num_nodes:
                nbrs[s].append(d)
                wlists[s].append(float(max(0.0, ww)))  # clamp non-neg

    probs: list[torch.Tensor] = []
    for u in range(num_nodes):
        if len(nbrs[u]) == 0:
            probs.append(torch.empty((0,), dtype=torch.float))
            continue
        wu = torch.tensor(wlists[u], dtype=torch.float)
        s = float(wu.sum().item())
        if s <= 0.0:
            wu = torch.ones_like(wu)
            s = float(wu.sum().item())
        probs.append(wu / s)

    return nbrs, probs

def _weighted_sample_neighbors(sources: torch.Tensor,
                               nbrs: list[torch.Tensor | None],
                               wts:  list[torch.Tensor | None],
                               k: int):
    """Sample up to k outgoing neighbors per source node, biased by weights."""
    sources_cpu = sources.detach().cpu().long()

    src_out, dst_out, w_out = [], [], []
    for s in sources_cpu.tolist():
        neigh = nbrs[s]
        if neigh is None or neigh.numel() == 0:
            continue

        wt = wts[s]
        if wt is None or wt.numel() != neigh.numel():
            wt = torch.ones(neigh.numel(), dtype=torch.float)

        if float(wt.sum().item()) <= 0.0:
            prob = torch.full((neigh.numel(),), 1.0 / float(neigh.numel()))
        else:
            prob = wt / wt.sum()

        kk = min(int(k), int(neigh.numel()))
        if kk <= 0:
            continue

        idx = torch.multinomial(prob, num_samples=kk, replacement=False)
        picked = neigh[idx]
        picked_w = wt[idx]

        src_out.append(torch.full((picked.numel(),), s, dtype=torch.long))
        dst_out.append(picked.long())
        w_out.append(picked_w.float())

    if len(src_out) == 0:
        return None, None, None

    return torch.cat(src_out), torch.cat(dst_out), torch.cat(w_out)



# def iter_weighted_ss_minibatches(
#     full_graph: HeteroData,
#     seed_nodes: torch.Tensor,
#     num_neighbors: list[int],
#     batch_size: int,
#     device: torch.device,
#     shuffle: bool = True,
# ):
#     """Yield sampled minibatch subgraphs; sampling is biased by edge_attr.

#     This is a manual fallback for weighted neighbor sampling when PyG's NeighborLoader
#     cannot use `weight_attr` (e.g., missing pyg-lib>=0.3.0).

#     Key invariants:
#       - All sampling/index construction is done on CPU (Python lists + torch.cat/unique).
#       - We only move the *constructed batch tensors* (x, edge_index, edge_attr, ids) to `device`.
#       - Seed nodes are guaranteed to be the first `batch_size` entries of `batch['sample'].n_id`.
#       - We also attach `batch['sample'].seed_nodes` (global ids) for downstream slicing.
#     """

#     assert REL_SS in full_graph.edge_index_dict, f"Missing {REL_SS} edges"

#     # Total number of sample nodes in the *full* graph.
#     num_nodes = int(full_graph["sample"].num_nodes)

#     # Read full-graph S-S edge index and optional edge weights.
#     ei = full_graph[REL_SS].edge_index
#     ew = getattr(full_graph[REL_SS], "edge_attr", None)

#     # Build CPU adjacency lists for weighted neighbor sampling.
#     nbrs, wts = _build_ss_adjacency(ei, ew, num_nodes)

#     # IMPORTANT: keep seeds on CPU so everything we concatenate is on the same device.
#     seeds = seed_nodes.detach().cpu().long()
#     if shuffle:
#         perm = torch.randperm(seeds.numel())
#         seeds = seeds[perm]

#     for start in range(0, seeds.numel(), int(batch_size)):
#         seed = seeds[start : start + int(batch_size)]
#         if seed.numel() == 0:
#             continue

#         frontier = seed  # CPU
#         all_nodes: list[torch.Tensor] = [seed]
#         all_src: list[torch.Tensor] = []
#         all_dst: list[torch.Tensor] = []
#         all_w: list[torch.Tensor] = []

#         # Multi-hop weighted sampling.
#         for k in num_neighbors:
#             ssrc, sdst, sw = _weighted_sample_neighbors(frontier, nbrs, wts, int(k))
#             if ssrc is None:
#                 frontier = frontier.new_empty((0,))
#                 continue

#             # ssrc/sdst/sw are CPU tensors
#             all_src.append(ssrc)
#             all_dst.append(sdst)
#             all_w.append(sw)

#             # Next frontier: unique neighbors sampled at this hop.
#             frontier = torch.unique(sdst)
#             all_nodes.append(frontier)

#         # All node ids involved in this minibatch subgraph (CPU).
#         # Ensure seeds come first in n_id.
#         nodes_cat = torch.unique(torch.cat(all_nodes, dim=0))
#         seed_set = set(seed.tolist())
#         rest = [n for n in nodes_cat.tolist() if n not in seed_set]
#         n_id_cpu = torch.tensor(seed.tolist() + rest, dtype=torch.long)

#         # Map global node ids -> local [0..num_batch_nodes-1] ids.
#         mapping = torch.full((num_nodes,), -1, dtype=torch.long)
#         mapping[n_id_cpu] = torch.arange(n_id_cpu.numel(), dtype=torch.long)

#         # Build local edge_index (+ edge_attr) on CPU.
#         if len(all_src) == 0:
#             edge_index_local_cpu = torch.empty((2, 0), dtype=torch.long)
#             edge_attr_local_cpu = None
#         else:
#             src_all = torch.cat(all_src)  # CPU
#             dst_all = torch.cat(all_dst)  # CPU

#             src_l = mapping[src_all]
#             dst_l = mapping[dst_all]
#             keep = (src_l >= 0) & (dst_l >= 0)

#             edge_index_local_cpu = torch.stack([src_l[keep], dst_l[keep]], dim=0)

#             w_all = torch.cat(all_w)
#             edge_attr_local_cpu = w_all[keep].view(-1, 1)

#         # Construct HeteroData batch.
#         batch = HeteroData()

#         # Slice node features. IMPORTANT: indices must be on the same device as x.
#         x_all = full_graph["sample"].x
#         n_id_for_x = n_id_cpu.to(device=x_all.device)
#         batch["sample"].x = x_all[n_id_for_x].to(device)

#         # Attach ids/metadata on requested device.
#         batch["sample"].n_id = n_id_cpu.to(device)
#         batch["sample"].batch_size = int(seed.numel())
#         batch["sample"].seed_nodes = seed.to(device)

#         # Attach sampled S-S edges.
#         batch[REL_SS].edge_index = edge_index_local_cpu.to(device)
#         if edge_attr_local_cpu is not None:
#             batch[REL_SS].edge_attr = edge_attr_local_cpu.to(device)

#         yield batch




EdgeType = Tuple[str, str, str]


def iter_weighted_ss_minibatches(
    data: HeteroData,
    *,
    seed_nodes: Tensor,
    num_neighbors: Union[int, Sequence[int]],
    batch_size: int,
    device: Union[str, torch.device] = "cpu",
    shuffle: bool = True,
    seed: Optional[int] = None,
    ss_edge_type: EdgeType = ("sample", "ss", "sample"),
    weight_attr: str = "edge_attr",
    include_all_edge_types: bool = True,
    allowed_nodes: torch.Tensor | Sequence[int] | None = None,
) -> Iterator[HeteroData]:
    """
    Manual neighbor sampler for HeteroData that performs WEIGHTED sampling on the
    ('sample','ss','sample') relation using edge weights, without requiring pyg-lib.

    Key behaviors (chosen to match NeighborLoader expectations used in clientDES):
      - Returns a HeteroData mini-batch with:
          batch["sample"].batch_size = B
          batch["sample"].n_id = global node ids for the sampled 'sample' nodes,
                                with the first B entries corresponding to seed nodes.
      - Sampling is layer-wise for `num_neighbors` hops (GraphSAGE-style fanout).
      - Weighted sampling only applies to the S-S edge type `ss_edge_type`
        using `data[ss_edge_type][weight_attr]` when present; otherwise uniform.

    Parameters
    ----------
    data:
        Full HeteroData graph (e.g., your train_val_graph).
    seed_nodes:
        1D tensor of GLOBAL node ids for node type 'sample' to draw batches from.
        (Typically train_nodes = train_mask.nonzero().view(-1))
    num_neighbors:
        Fanout per layer. Either int or list/tuple of ints (one per layer).
        Example: [15, 15] for 2-layer SAGE.
    batch_size:
        Number of seed nodes per mini-batch.
    device:
        Device for internal tensor bookkeeping (NOT where the returned batch lives).
        You typically yield on CPU and call batch.to(device) in the train loop.
    shuffle:
        Whether to shuffle the order of seed nodes each epoch call.
    seed:
        RNG seed for reproducibility.
    ss_edge_type:
        Edge type to perform weighted sampling on.
    weight_attr:
        Edge attribute name holding weights for ss_edge_type.
    include_all_edge_types:
        If True, include induced edges for *all* edge types among sampled nodes.
        If False, only include the sampled S-S edges plus whatever is needed to
        keep node stores consistent.

    Yields
    ------
    HeteroData mini-batches.
    """

    if not isinstance(data, HeteroData):
        raise TypeError(f"iter_weighted_ss_minibatches expects HeteroData, got {type(data)}")

    if "sample" not in data.node_types:
        raise KeyError("HeteroData must contain node type 'sample'.")

    if ss_edge_type not in data.edge_types:
        raise KeyError(f"Expected ss_edge_type={ss_edge_type} in data.edge_types, but it was missing.")

    # Normalize fanout into a list (one per layer)
    if isinstance(num_neighbors, int):
        fanouts: List[int] = [int(num_neighbors)]
    else:
        fanouts = [int(x) for x in num_neighbors]
    if any(k < 0 for k in fanouts):
        raise ValueError(f"num_neighbors must be >= 0 per layer, got {fanouts}")

    if seed_nodes.dim() != 1:
        seed_nodes = seed_nodes.view(-1)

    # IMPORTANT: keep ids on CPU for stable indexing and to avoid device-mismatch cats.
    seed_nodes_cpu = seed_nodes.detach().to("cpu", non_blocking=True).long()

    # RNG
    g_cpu = torch.Generator(device="cpu")
    if seed is not None:
        g_cpu.manual_seed(int(seed))

    # Optionally shuffle seed order
    if shuffle:
        perm = torch.randperm(seed_nodes_cpu.numel(), generator=g_cpu)
        seed_nodes_cpu = seed_nodes_cpu[perm]

    # ---- Build adjacency for S-S edge type ----
    # We build "out-neighbors" lists keyed by source node id (global indexing).
    ss_ei = data[ss_edge_type].edge_index
    if ss_ei.device.type != "cpu":
        ss_ei = ss_ei.detach().to("cpu", non_blocking=True)
    ss_src = ss_ei[0].long()
    ss_dst = ss_ei[1].long()

    ss_w = None
    if hasattr(data[ss_edge_type], weight_attr):
        ss_w = getattr(data[ss_edge_type], weight_attr)
        if ss_w is not None:
            ss_w = ss_w.detach()
            if ss_w.device.type != "cpu":
                ss_w = ss_w.to("cpu", non_blocking=True)
            ss_w = ss_w.float()

    # Build CSR-like index: for each node u, edges are in [ptr[u], ptr[u+1])
    # using a sort by source.
    num_sample_nodes = int(data["sample"].num_nodes)



    # Allowed node filtering (safety): restrict sampling strictly to train nodes.
    num_nodes = int(data["sample"].num_nodes) if getattr(data["sample"], "num_nodes", None) is not None else int(data["sample"].x.size(0))

    if allowed_nodes is None:
        # Default to train_mask when present (safe-by-default).
        tm = getattr(data["sample"], "train_mask", None)
        if tm is not None:
            allowed_mask = tm.detach().cpu().bool()
        else:
            allowed_mask = None
    else:
        if isinstance(allowed_nodes, torch.Tensor):
            an = allowed_nodes.detach().cpu().long().view(-1)
        else:
            an = torch.tensor(list(allowed_nodes), dtype=torch.long)
        allowed_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        if an.numel() > 0:
            allowed_mask[an.clamp(min=0, max=num_nodes - 1)] = True

    # Ensure seeds are allowed (fail closed if not)
    if allowed_mask is not None:
        s_cpu = seed_nodes.detach().cpu().long().view(-1)
        if s_cpu.numel() > 0 and (not bool(allowed_mask[s_cpu].all().item())):
            bad = s_cpu[~allowed_mask[s_cpu]]
            raise ValueError(
                "iter_weighted_ss_minibatches: seed_nodes contains disallowed nodes (e.g., val/test). "
                f"First few bad ids: {bad[:10].tolist()}"
            )

    # If allowed_mask is set, restrict the S-S edge list so we can NEVER traverse
    # into disallowed nodes (e.g., val/test) during sampling.
    if allowed_mask is not None:
        keep_e = allowed_mask[ss_src] & allowed_mask[ss_dst]
        ss_src = ss_src[keep_e]
        ss_dst = ss_dst[keep_e]
        if ss_w is not None:
            ss_w = ss_w[keep_e]

    order = torch.argsort(ss_src)
    ss_src_sorted = ss_src[order]
    ss_dst_sorted = ss_dst[order]
    ss_w_sorted = ss_w[order] if ss_w is not None else None

    ptr = torch.zeros(num_sample_nodes + 1, dtype=torch.long)
    # counts per src
    ones = torch.ones_like(ss_src_sorted, dtype=torch.long)
    ptr = ptr.index_add(0, ss_src_sorted.clamp(min=0, max=num_sample_nodes - 1), ones)
    ptr = torch.cumsum(ptr, dim=0)
    # ptr[u] now is end pointer; convert to start pointers by shifting right
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), ptr])  # length num_sample_nodes+2
    # start = ptr[u], end = ptr[u+1]
    # (we created one extra; fix to exact length num_sample_nodes+1)
    ptr = ptr[: num_sample_nodes + 1]

    def _neighbors_and_weights(u: int) -> Tuple[Tensor, Optional[Tensor]]:
        """Return (nbrs, w) for source node u, both on CPU."""
        if u < 0 or u >= num_sample_nodes:
            return (torch.empty(0, dtype=torch.long), None)
        start = int(ptr[u].item())
        end = int(ptr[u + 1].item()) if (u + 1) < ptr.numel() else start
        if end <= start:
            return (torch.empty(0, dtype=torch.long), None)
        nbrs = ss_dst_sorted[start:end]
        w = None if ss_w_sorted is None else ss_w_sorted[start:end]

        # Extra safety: even if edges are filtered above, ensure we never return
        # disallowed neighbors.
        if allowed_mask is not None and nbrs.numel() > 0:
            keep_n = allowed_mask[nbrs]
            nbrs = nbrs[keep_n]
            if w is not None:
                w = w[keep_n]

        return (nbrs, w)

    def _weighted_sample(nbrs: Tensor, w: Optional[Tensor], k: int) -> Tensor:
        """Sample up to k neighbors (CPU tensors)."""
        if k <= 0 or nbrs.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        if w is None:
            # Uniform without replacement if possible
            if nbrs.numel() <= k:
                return nbrs
            idx = torch.randperm(nbrs.numel(), generator=g_cpu)[:k]
            return nbrs[idx]

        # Clean weights: clamp negatives, handle all-zeros.
        ww = w.clone()
        ww = torch.clamp(ww, min=0.0)
        if float(ww.sum().item()) <= 0.0:
            # fallback uniform
            if nbrs.numel() <= k:
                return nbrs
            idx = torch.randperm(nbrs.numel(), generator=g_cpu)[:k]
            return nbrs[idx]

        # If k >= deg, take all.
        if nbrs.numel() <= k:
            return nbrs

        # Sample without replacement using torch.multinomial
        probs = ww / ww.sum()
        idx = torch.multinomial(probs, num_samples=k, replacement=False, generator=g_cpu)
        return nbrs[idx]

    # ---- helper: build induced subgraph for hetero data ----
    def _build_batch_from_nodes(
        node_ids_by_type: Dict[str, Tensor],
        sample_seed_count: int,
    ) -> HeteroData:
        """
        Construct a HeteroData batch containing node features sliced from `data`
        and edges restricted to the sampled nodes.

        Ensures:
          - batch['sample'].n_id exists and first sample_seed_count entries are seeds.
          - batch['sample'].batch_size is set.
        """
        batch = HeteroData()

        # Mapping global -> local per node type
        maps: Dict[str, Tensor] = {}

        for ntype, nids in node_ids_by_type.items():
            nids_cpu = nids.detach().to("cpu", non_blocking=True).long()
            # preserve order exactly as provided
            maps[ntype] = torch.full(
                (int(data[ntype].num_nodes),),
                -1,
                dtype=torch.long,
            )
            maps[ntype][nids_cpu] = torch.arange(nids_cpu.numel(), dtype=torch.long)

            # Copy node store attributes
            for key, val in data[ntype].items():
                if torch.is_tensor(val) and val.size(0) == int(data[ntype].num_nodes):
                    batch[ntype][key] = val[nids_cpu]
                else:
                    # keep non-node-aligned attrs as-is if simple types
                    # (rarely needed; safe to skip)
                    pass

            batch[ntype].n_id = nids_cpu  # global ids (NeighborLoader convention)

        # Attach batch_size for sample seeds
        batch["sample"].batch_size = int(sample_seed_count)

        # Add edges
        if include_all_edge_types:
            edge_types = list(data.edge_types)
        else:
            edge_types = [ss_edge_type]

        for et in edge_types:
            src_t, rel, dst_t = et
            if src_t not in node_ids_by_type or dst_t not in node_ids_by_type:
                continue

            ei = data[et].edge_index
            if ei.device.type != "cpu":
                ei_cpu = ei.detach().to("cpu", non_blocking=True)
            else:
                ei_cpu = ei.detach()

            src_g = ei_cpu[0].long()
            dst_g = ei_cpu[1].long()

            src_l = maps[src_t][src_g]
            dst_l = maps[dst_t][dst_g]
            keep = (src_l >= 0) & (dst_l >= 0)
            if not bool(keep.any()):
                continue

            new_ei = torch.stack([src_l[keep], dst_l[keep]], dim=0)
            batch[et].edge_index = new_ei

            # edge_attr (if present and edge-aligned)
            if hasattr(data[et], weight_attr):
                ea = getattr(data[et], weight_attr)
                if torch.is_tensor(ea) and ea.size(0) == ei_cpu.size(1):
                    batch[et][weight_attr] = ea.detach().to("cpu", non_blocking=True)[keep]

            # copy any other edge-aligned tensors if you rely on them elsewhere
            for key, val in data[et].items():
                if key in {"edge_index", weight_attr}:
                    continue
                if torch.is_tensor(val) and val.size(0) == ei_cpu.size(1):
                    batch[et][key] = val.detach().to("cpu", non_blocking=True)[keep]

        return batch

    # ---- iterate over mini-batches of seeds ----
    n = seed_nodes_cpu.numel()
    if n == 0:
        return

    for start in range(0, n, batch_size):
        seeds = seed_nodes_cpu[start : start + batch_size]
        if seeds.numel() == 0:
            continue

        # Layer-wise expansion on SAMPLE nodes using S-S edges (weighted).
        # We only *sample additional SAMPLE nodes* here; then we induce all other edge types
        # among the final sampled node sets (which will include classifier nodes only if
        # they are already in node_ids_by_type, see below).
        sampled_sample_nodes: List[Tensor] = [seeds]  # keep order: seeds first
        frontier = seeds

        sampled_ss_edges_src: List[Tensor] = []
        sampled_ss_edges_dst: List[Tensor] = []
        sampled_ss_edges_w: List[Tensor] = []

        for k in fanouts:
            if k <= 0:
                frontier = torch.empty(0, dtype=torch.long)
                continue

            next_nodes: List[Tensor] = []
            # For each node in frontier, sample up to k neighbors
            for u in frontier.tolist():
                nbrs, w = _neighbors_and_weights(int(u))
                picked = _weighted_sample(nbrs, w, k)
                if picked.numel() > 0:
                    next_nodes.append(picked)
                    # record sampled edges (global ids)
                    sampled_ss_edges_src.append(torch.full((picked.numel(),), int(u), dtype=torch.long))
                    sampled_ss_edges_dst.append(picked)
                    if w is None:
                        sampled_ss_edges_w.append(torch.ones((picked.numel(),), dtype=torch.float))
                    else:
                        w_map = {int(n.item()): float(wi.item()) for n, wi in zip(nbrs, w)}
                        sampled_ss_edges_w.append(
                            torch.tensor(
                                [w_map.get(int(n.item()), 0.0) for n in picked],
                                dtype=torch.float,
                            )
                        )

            if len(next_nodes) == 0:
                frontier = torch.empty(0, dtype=torch.long)
                continue

            frontier = torch.unique(torch.cat(next_nodes, dim=0), sorted=False)
            if allowed_mask is not None and frontier.numel() > 0:
                frontier = frontier[allowed_mask[frontier]]
            sampled_sample_nodes.append(frontier)

        # Final SAMPLE node set in the batch:
        # seeds first (in original order), then the unique extra nodes.
        extra = torch.unique(torch.cat(sampled_sample_nodes[1:], dim=0), sorted=False) if len(sampled_sample_nodes) > 1 else torch.empty(0, dtype=torch.long)
        # remove any duplicates of seeds from extra
        if extra.numel() > 0:
            seed_set = torch.zeros(num_sample_nodes, dtype=torch.bool)
            seed_set[seeds] = True
            extra = extra[~seed_set[extra]]
        if allowed_mask is not None and extra.numel() > 0:
            extra = extra[allowed_mask[extra]]
        sample_nodes_batch = torch.cat([seeds, extra], dim=0) if extra.numel() > 0 else seeds

        if allowed_mask is not None and sample_nodes_batch.numel() > 0:
            if not bool(allowed_mask[sample_nodes_batch].all().item()):
                bad = sample_nodes_batch[~allowed_mask[sample_nodes_batch]]
                raise RuntimeError(
                    "iter_weighted_ss_minibatches: sampled disallowed nodes despite filtering. "
                    f"First few bad ids: {bad[:10].tolist()}"
                )

        node_ids_by_type: Dict[str, Tensor] = {"sample": sample_nodes_batch}

        # NOTE:
        # If your model requires other node types (e.g., 'classifier'), we include them by
        # inducing edges among the sampled SAMPLE nodes only. That means classifier nodes
        # will only appear if you also add them here.
        #
        # If you DO need classifiers in the batch (common for hetero models),
        # uncomment the block below to pull in 1-hop classifier neighbors of sampled samples.
        #
        # This is uniform (not weighted). It is usually cheap because #classifiers is small.
        #
        # ---- begin optional classifier expansion ----
        if "classifier" in data.node_types:
            clf_nodes: List[Tensor] = []
            for et in data.edge_types:
                src_t, _, dst_t = et
                if src_t == "sample" and dst_t == "classifier":
                    ei = data[et].edge_index
                    ei_cpu = ei.detach().to("cpu", non_blocking=True) if ei.device.type != "cpu" else ei.detach()
                    src_g = ei_cpu[0].long()
                    dst_g = ei_cpu[1].long()
                    # keep classifier neighbors for sampled samples
                    mask = torch.zeros(num_sample_nodes, dtype=torch.bool)
                    mask[sample_nodes_batch] = True
                    keep = mask[src_g]
                    if bool(keep.any()):
                        clf_nodes.append(dst_g[keep])
            if len(clf_nodes) > 0:
                clf_nodes_u = torch.unique(torch.cat(clf_nodes, dim=0), sorted=False)
                node_ids_by_type["classifier"] = clf_nodes_u
        # ---- end optional classifier expansion ----

        batch = _build_batch_from_nodes(node_ids_by_type, sample_seed_count=int(seeds.numel()))

        # Also include the S-S sampled edges explicitly (optional, but can be useful if you
        # want the batch to reflect *only sampled* S-S edges rather than induced-all edges).
        # If you prefer induced edges only, you can remove this override block.
        if len(sampled_ss_edges_src) > 0:
            ss_src_g = torch.cat(sampled_ss_edges_src, dim=0)
            ss_dst_g = torch.cat(sampled_ss_edges_dst, dim=0)
            ss_w_g = torch.cat(sampled_ss_edges_w, dim=0) if sampled_ss_edges_w else None

            # remap to local sample indices
            sample_map = torch.full((num_sample_nodes,), -1, dtype=torch.long)
            sample_map[sample_nodes_batch] = torch.arange(sample_nodes_batch.numel(), dtype=torch.long)

            ss_src_l = sample_map[ss_src_g]
            ss_dst_l = sample_map[ss_dst_g]
            keep = (ss_src_l >= 0) & (ss_dst_l >= 0)
            if bool(keep.any()):
                batch[ss_edge_type].edge_index = torch.stack([ss_src_l[keep], ss_dst_l[keep]], dim=0)
                if ss_w_g is not None:
                    batch[ss_edge_type][weight_attr] = ss_w_g[keep].to(torch.float)

        yield batch

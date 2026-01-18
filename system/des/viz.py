# des/viz.py
from __future__ import annotations
import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple
import json
import torch
import tempfile
from contextlib import nullcontext
# headless backend once, not per-call
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from des.dataset_stats import load_client_label_counts


def plot_weight_grid(
    heads: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    names: list[str],
    local_mask: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    """
    Draw 1–2 rows (meta head, optional ensemble head). Each row: pre logits, post weights, sparsity.
    heads: [(head_title, pre[T,M], post[T,M], nnz[T])]
    """
    ensure_dir(out_path)
    nrows = len(heads)
    fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(12, 4 * nrows), constrained_layout=True)
    if nrows == 1:
        axs = np.asarray(axs)[None, :]  # (1,3)

    for r, (h_title, pre, post, nnz) in enumerate(heads):
        xs = np.arange(1, pre.shape[0] + 1)
        idx_local    = np.where(local_mask[:pre.shape[1]])[0]
        idx_nonlocal = np.where(~local_mask[:pre.shape[1]])[0]

        # pre
        ax = axs[r, 0]
        for m in idx_nonlocal: ax.plot(xs, pre[:, m], linewidth=0.9, alpha=0.65)
        for m in idx_local:    ax.plot(xs, pre[:, m], linewidth=2.0)
        ax.set_xlabel("epoch"); ax.set_ylabel("mean logit"); ax.set_title(f"{h_title} (pre)")

        # post (+ top-3)
        ax = axs[r, 1]
        for m in idx_nonlocal: ax.plot(xs, post[:, m], linewidth=0.9, alpha=0.65)
        for m in idx_local:    ax.plot(xs, post[:, m], linewidth=2.0)
        final_w   = post[-1]
        top_idx   = np.argsort(final_w)[::-1][:min(3, len(final_w))]
        top_names = [str(names[int(i)]) for i in top_idx.tolist()]
        ax.set_xlabel("epoch"); ax.set_ylabel("mean weight")
        ax.set_title(f"{h_title} (post) — Top-{len(top_names)}: " + ", ".join(top_names))

        # sparsity
        ax = axs[r, 2]
        ax.plot(xs, nnz, linewidth=1.5)
        ax.set_xlabel("epoch"); ax.set_ylabel("# active experts")
        ax.set_title(f"{h_title} (sparsity)")

    # legend via line width
    h_local, h_nonlocal = Line2D([0],[0], lw=2), Line2D([0],[0], lw=1)
    axs[0, 0].legend([h_local, h_nonlocal], ["local", "non-local"], loc="best", fontsize=9)

    fig.suptitle(title)
    fig.savefig(out_path)
    plt.close(fig)


def _format_classifier_name(name: Any) -> str:
    if isinstance(name, (list, tuple)):
        return "_".join(str(part) for part in name)
    return str(name)


def save_graph_summaries(self,
    client_role: str,
    args: object | None,
    n_train: int,
    n_eval: int,
    n_classifiers: int,
    classifier_names: "list[str] | None" = None,
    sample_labels: "np.ndarray | None" = None,
    ss_edge_index: "np.ndarray | None" = None,   # shape (2, E_ss)
    ss_edge_attr:  "np.ndarray | None" = None,   # shape (E_ss,)
    cs_edge_index: "np.ndarray | None" = None,   # shape (2, E_cs)
    cs_edge_attr:  "np.ndarray | None" = None,   # shape (E_cs,)
    cc_edge_index: "np.ndarray | None" = None,   # shape (2, E_cc)
    cc_edge_attr:  "np.ndarray | None" = None,   # shape (E_cc,)
    bins: int = 50,
    eval_kind: str = "val",
) -> str:
    """
    Create text + histogram summaries for a client's graph artifacts for a given eval split.

    Args:
        n_train: number of TRAIN sample nodes (sources for SS; CS destinations)
        n_eval:  number of EVAL sample nodes (VAL or TEST; destinations only)
        n_classifiers: number of classifier nodes
        eval_kind: 'val' or 'test' (used in filenames)

    Saves under:
      {args.graph_dir}/graph_summaries/{client_role}/texts/summary_{eval_kind}.txt|json
      {args.graph_dir}/graph_summaries/{client_role}/plots/*
    """

    def as_np(x):
        if x is None:
            return None
        x = np.asarray(x)
        return x

    outputs_dir = Path(self.graph_outputs_dir) / client_role
    # graph_root = Path(getattr(args, "graph_dir", save_folder_name))
    # base_dir  = Path(graph_root) / "graph_summaries" / client_role
    texts_dir = outputs_dir / "texts"
    plots_dir = outputs_dir / "plots"
    cache_path = outputs_dir / "plot_cache.pt"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(self, "_graph_plot_registry"):
        self._graph_plot_registry = {}

    plots_by_type_root = Path(self.graph_outputs_dir) / "plots_by_type"

    ss_edge_index = as_np(ss_edge_index)
    ss_edge_attr  = as_np(ss_edge_attr)
    cs_edge_index = as_np(cs_edge_index)
    cs_edge_attr  = as_np(cs_edge_attr)
    cc_edge_index = as_np(cc_edge_index)
    cc_edge_attr  = as_np(cc_edge_attr)
    sample_labels = as_np(sample_labels)

    n_samples = int(n_train + n_eval)
    print("len(sample_labels):", sample_labels.shape if sample_labels is not None else None, flush=True)

    def degree_stats(edge_index: "np.ndarray | None", num_src: int, num_dst: int) -> Tuple[np.ndarray, np.ndarray]:
        if edge_index is None or edge_index.size == 0:
            return np.zeros(num_src, dtype=int), np.zeros(num_dst, dtype=int)
        src, dst = edge_index
        out_deg = np.bincount(src.astype(np.int64), minlength=num_src)
        in_deg  = np.bincount(dst.astype(np.int64), minlength=num_dst)
        return out_deg, in_deg

    def _load_plot_cache() -> Dict[str, Dict[str, Any]]:
        if cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception:
                return {}
        return {}

    def _save_plot_cache(cache: Dict[str, Dict[str, Any]]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)

    def _register_plot(plot_name: str, fig: plt.Figure) -> None:
        registry = getattr(self, "_graph_plot_registry", {})
        registry.setdefault(client_role, []).append((plot_name, fig))
        self._graph_plot_registry = registry
        print(f"[Viz][Debug] Registered plot {plot_name} for client {client_role}", flush=True)

    def _plot_path(plot_name: str) -> Path:
        dest = plots_by_type_root / Path(plot_name).stem
        dest.mkdir(parents=True, exist_ok=True)
        return dest / f"{client_role}.png"

    def _plot_grid(
        plot_name: str,
        values_by_split: Dict[str, list[tuple[str, np.ndarray]]],
        ncols: int,
        title_prefix: str,
        xlabel: str,
        ylabel: str,
        bins: int,
        top_text: str | None = None,
        row_order: list[str] | None = None,
    ) -> None:
        rows = row_order or ["val", "test"]
        fig, axs = plt.subplots(nrows=len(rows), ncols=ncols, figsize=(4 * ncols, 3.2 * len(rows)), constrained_layout=True)
        axes = np.atleast_2d(axs)
        if axes.shape[0] != len(rows):
            axes = axes.reshape(len(rows), ncols)
        empty_kw = dict(
            ha="center",
            va="center",
            fontsize=11,
            color="#C00000",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#C00000"),
        )
        for row_idx, row_name in enumerate(rows):
            row_values = values_by_split.get(row_name)
            if row_values is None or len(row_values) < ncols:
                row_values = [("all samples", np.array([])) for _ in range(ncols)]
            for col_idx in range(ncols):
                ax = axes[row_idx, col_idx]
                entry = row_values[col_idx]
                if isinstance(entry, dict):
                    title = entry.get("title", "")
                    vals = entry.get("values", np.array([]))
                else:
                    title, vals = entry
                if vals is not None and vals.size:
                    ax.hist(vals, bins=bins, color="#4C72B0", edgecolor="black")
                else:
                    ax.text(0.5, 0.5, "No edges", transform=ax.transAxes, **empty_kw)
                    ax.tick_params(left=False, labelleft=False)
                ax.set_title(f"{row_name.upper()} • {title}")
                if row_idx == len(rows) - 1:
                    ax.set_xlabel(xlabel)
                if col_idx == 0:
                    ax.set_ylabel(ylabel)
        fig.suptitle(title_prefix)
        if top_text:
            fig.text(0.5, 0.95, top_text, ha="center", fontsize=10)
        plot_path = _plot_path(plot_name)
        fig.savefig(plot_path)
        _register_plot(plot_name, fig)

    num_classes = int(getattr(args, "num_classes", sample_labels.max() + 1 if sample_labels is not None else 0))
    present_classes = list(range(num_classes)) if num_classes else []

    lines = []
    lines.append(f"Client: {client_role}")
    lines.append(f"Eval kind: {eval_kind}")
    lines.append(f"Num TRAIN sample nodes: {int(n_train)}")
    lines.append(f"Num EVAL  sample nodes: {int(n_eval)}")
    lines.append(f"Num classifier nodes:  {int(n_classifiers)}")

    # -------- SS: sample -> sample (TRAIN sources, TRAIN+EVAL destinations) --------
    ss_w_overall = np.array([])
    ss_w_eval = np.array([])
    ss_w_train = np.array([])
    if ss_edge_index is not None and ss_edge_index.size:
        ss_E = ss_edge_index.shape[1]
        ss_out, ss_in = degree_stats(ss_edge_index, n_samples, n_samples)
        dst = ss_edge_index[1]
        dst_train_mask = dst < n_train
        dst_eval_mask  = (dst >= n_train) & (dst < n_samples)

        # Weight splits
        if ss_edge_attr is not None and ss_edge_attr.size:
            ss_w_overall = ss_edge_attr
            ss_w_train = ss_edge_attr[dst_train_mask] if dst_train_mask.any() else np.array([])
            ss_w_eval  = ss_edge_attr[dst_eval_mask]  if dst_eval_mask.any()  else np.array([])
        else:
            ss_w_overall = np.array([]); ss_w_train = np.array([]); ss_w_eval = np.array([])

        lines.append(f"[ss] edges={ss_E}  avg_outdeg(all sources)={ss_out.mean():.3f}")
        lines.append(
            f"[ss] dest TRAIN avg_indeg={ss_in[:n_train].mean():.3f}  dest EVAL avg_indeg={ss_in[n_train:].mean() if n_eval>0 else 0.0:.3f}"
        )
    else:
        lines.append("[ss] edges=0")

    # -------- CS: classifier -> sample (TRAIN destinations only) --------
    cs_out = np.array([])
    if cs_edge_index is not None and cs_edge_index.size:
        cs_E = cs_edge_index.shape[1]
        cs_out, cs_in_full = degree_stats(cs_edge_index, n_classifiers, n_samples)
        cs_in_train = cs_in_full[:n_train]
        cs_in_eval  = cs_in_full[n_train:]
        lines.append(f"[cs] edges={cs_E}  avg_outdeg(classifier)={cs_out.mean():.3f}  avg_indeg TRAIN={cs_in_train.mean():.3f}  EVAL={cs_in_eval.mean() if n_eval>0 else 0.0:.3f}")
        lines.append(f"[cs] classifiers with no edges: {int((cs_out==0).sum())} / {int(len(cs_out))}")
        lines.append(f"[cs] samples with no edges (TRAIN): {int((cs_in_train==0).sum())} / {int(len(cs_in_train))}")
        if n_eval > 0:
            lines.append(f"[cs] samples with no edges (EVAL):  {int((cs_in_eval==0).sum())} / {int(len(cs_in_eval))}")
    else:
        lines.append("[cs] edges=0")

    # -------- CC: classifier <-> classifier --------
    if cc_edge_index is not None and cc_edge_index.size:
        cc_E = cc_edge_index.shape[1]
        cc_out, cc_in = degree_stats(cc_edge_index, n_classifiers, n_classifiers)
        cc_deg = ((cc_out + cc_in) / 2.0) if cc_out.size else np.zeros(n_classifiers, dtype=float)
        lines.append(f"[cc] edges={cc_E}  avg_deg(classifier)={cc_deg.mean():.3f}  isolated={(cc_deg==0).sum()} / {int(len(cc_deg))}")
        if cc_edge_attr is not None and cc_edge_attr.size and eval_kind == "val":
            plt.figure(); plt.hist(cc_edge_attr, bins=bins); plt.title("CC edge weights (1 - DF)"); plt.xlabel("weight"); plt.ylabel("count"); plt.tight_layout(); plt.savefig(plots_dir / "cc_edge_weights.png"); plt.close()
    else:
        lines.append("[cc] edges=0")

    # ---- write text + JSON ----
    with open(texts_dir / f"summary_{eval_kind}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    stats: Dict[str, Any] = {}
    if ss_edge_attr is not None and ss_edge_attr.size:
        stats["ss_edge_weights_all"] = {
            "min": float(ss_edge_attr.min()),
            "mean": float(ss_edge_attr.mean()),
            "max": float(ss_edge_attr.max()),
        }
    if "ss_w_eval" in locals() and ss_w_eval is not None and ss_w_eval.size:
        stats["ss_edge_weights_eval"] = {
            "min": float(ss_w_eval.min()),
            "mean": float(ss_w_eval.mean()),
            "max": float(ss_w_eval.max()),
        }
    if cs_edge_index is not None and cs_edge_index.size:
        stats["classifier_out_degree"] = {
            "min": int(cs_out.min()),
            "mean": float(float(cs_out.mean())),
            "max": int(cs_out.max()),
        }
        stats["classifiers_with_no_edges"] = int((cs_out == 0).sum())

    with open(texts_dir / f"summary_{eval_kind}.json", "w") as jf:
        json.dump({
            "client": client_role,
            "eval_kind": eval_kind,
            "n_train": int(n_train),
            "n_eval": int(n_eval),
            "n_classifiers": int(n_classifiers),
            "has_ss": bool(ss_edge_index is not None and ss_edge_index.size),
            "has_cs": bool(cs_edge_index is not None and cs_edge_index.size),
            "has_cc": bool(cc_edge_index is not None and cc_edge_index.size),
            "stats": stats,
        }, jf, indent=2)

    # Cache data for combined grids (val + test)
    cache = _load_plot_cache()
    cache[eval_kind] = {
        "sample_labels": sample_labels,
        "n_train": int(n_train),
        "n_eval": int(n_eval),
        "cs_edge_index": cs_edge_index,
        "cs_edge_attr": cs_edge_attr,
        "ss_edge_index": ss_edge_index,
        "ss_edge_attr": ss_edge_attr,
        "n_classifiers": int(n_classifiers),
    }
    _save_plot_cache(cache)

    # Build combined grids once both splits are available
    if {"val", "test"}.issubset(set(cache.keys())) and sample_labels is not None:
        max_label_plus1 = 0
        for split in ("val", "test"):
            labels = cache[split]["sample_labels"]
            if labels is None or not labels.size:
                continue
            max_label_plus1 = max(max_label_plus1, int(labels.max()) + 1)
        label_bins = max(num_classes, max_label_plus1)
        label_counts = np.zeros(label_bins, dtype=int)
        for split in ("val", "test"):
            labels = cache[split]["sample_labels"]
            if labels is None or not labels.size:
                continue
            counts = np.bincount(labels.astype(np.int64), minlength=label_bins)
            label_counts += counts
        grid_present_classes = [int(cls) for cls, cnt in enumerate(label_counts) if cnt > 0]
        grid_ncols = 1 + len(grid_present_classes)

        n_clf = cache["val"]["n_classifiers"]
        def _make_classifier_names() -> list[str]:
            if classifier_names:
                names = [str(name) for name in classifier_names]
            else:
                names = []
            defaults = [f"classifier_{i}" for i in range(n_clf)]
            if len(names) >= n_clf:
                return names[:n_clf]
            names.extend(defaults[len(names):])
            return names

        clf_names = _make_classifier_names()
        overall_cs_out = np.zeros(n_clf, dtype=int)
        if n_clf > 0:
            for split in ("val", "test"):
                data = cache[split]
                cs_idx = data["cs_edge_index"]
                if cs_idx is None or cs_idx.size == 0:
                    continue
                src = cs_idx[0].astype(np.int64)
                np.add.at(overall_cs_out, src, 1)

        top_k = min(3, n_clf)
        top_idx = np.argsort(overall_cs_out)[::-1][:top_k]
        top_names = [clf_names[int(i)] for i in top_idx] if n_clf else []
        top_text = f"Top CS classifiers: {', '.join(top_names)}" if top_names else None

        def _build_classwise_cs_support() -> None:
            if cs_edge_index is None:
                return
            dataset_name = getattr(args, "dataset", "")
            label_counts = load_client_label_counts(dataset_name)
            classifier_keys = classifier_names if classifier_names else []
            classifier_pairs: list[tuple[str, str]] = []
            for idx in range(n_clf):
                key = classifier_keys[idx] if idx < len(classifier_keys) else (f"classifier_{idx}", "")
                if isinstance(key, (list, tuple)) and len(key) >= 2:
                    classifier_pairs.append((str(key[0]), str(key[1])))
                else:
                    classifier_pairs.append((str(key), ""))

            src = cs_edge_index[0].astype(np.int64)
            dst = cs_edge_index[1].astype(np.int64)
            target_labels = sample_labels
            rows: list[Dict[str, Any]] = []
            top_k = 5
            for cls in grid_present_classes:
                mask = target_labels[dst] == cls
                if not mask.any():
                    continue
                counts = np.bincount(src[mask], minlength=n_clf)
                top_idxs = np.argsort(counts)[::-1]
                entries: list[Dict[str, Any]] = []
                unique_nodes = np.unique(dst[mask].astype(np.int64))
                target_class_count = int(unique_nodes.size)
                for idx in top_idxs:
                    if len(entries) >= top_k:
                        break
                    outdeg = int(counts[idx])
                    if outdeg == 0:
                        break
                    role, model = classifier_pairs[idx]
                    home_counts = label_counts.get(role, {})
                    home_total = sum(home_counts.values()) or 1
                    support = home_counts.get(int(cls), 0) / home_total
                    normalized = outdeg / target_class_count if target_class_count > 0 else 0.0
                    entries.append({
                        "home_client": role,
                        "model": model,
                        "outdegree": outdeg,
                        "home_support": support,
                        "home_counts": [{"label": int(k), "count": int(v)} for k, v in home_counts.items()],
                        "target_class_count": target_class_count,
                        "normalized_outdegree": normalized,
                    })
                if entries:
                    rows.append({
                        "target_class": int(cls),
                        "top_classifiers": entries,
                    })
            if not rows:
                return
            out_data = {
                "client": client_role,
                "eval_kind": eval_kind,
                "classwise_cs_support": rows,
            }
            with open(plots_dir / "cs_outdegree_classwise.json", "w") as jf:
                json.dump(out_data, jf, indent=2)

        _build_classwise_cs_support()

        def _format_top_k(counts: np.ndarray) -> str | None:
            if counts is None or counts.size == 0 or not clf_names:
                return "Top: none"
            top_idxs = np.argsort(counts)[::-1][:min(3, counts.size)]
            names = []
            for idx in top_idxs:
                idx = int(idx)
                if idx >= len(clf_names):
                    continue
                names.append(f"{clf_names[idx]} ({int(counts[idx])})")
            return "Top: " + ", ".join(names) if names else "Top: none"

        def _top_classifiers(counts: np.ndarray) -> list[dict[str, Any]]:
            if counts is None or not counts.size:
                return []
            top_idxs = np.argsort(counts)[::-1][:min(3, counts.size)]
            results = []
            for idx in top_idxs:
                idx = int(idx)
                if idx >= len(clf_names):
                    continue
                results.append({
                    "classifier": clf_names[idx],
                    "degree": int(counts[idx]),
                })
            return results

        def _extract_cs_out_degree(split: str) -> list[dict[str, Any]]:
            data = cache[split]
            labels = data["sample_labels"]
            if labels is None or data["cs_edge_index"] is None:
                return [{"title": "all samples", "values": np.array([]), "top_summary": []} for _ in range(grid_ncols)]
            src, dst = data["cs_edge_index"]
            n_clf_local = data["n_classifiers"]

            def counts(mask: np.ndarray) -> np.ndarray:
                if not mask.any():
                    return np.array([])
                masked_src = src[mask].astype(np.int64)
                return np.bincount(masked_src, minlength=n_clf_local)

            overall = counts(np.ones_like(dst, dtype=bool))
            results = [
                {
                    "title": "all samples",
                    "values": overall,
                    "top_summary": _top_classifiers(overall),
                }
            ]
            for cls in grid_present_classes:
                cls_mask = labels[dst.astype(np.int64)] == cls
                cls_counts = counts(cls_mask)
                results.append({
                    "title": f"class {cls}",
                    "values": cls_counts,
                    "top_summary": _top_classifiers(cls_counts),
                })
            return results

        def _extract_cs_weights(split: str) -> list[tuple[str, np.ndarray]]:
            data = cache[split]
            labels = data["sample_labels"]
            if labels is None or data["cs_edge_index"] is None or data["cs_edge_attr"] is None:
                return [(f"all samples", np.array([])) for _ in range(grid_ncols)]
            dst = data["cs_edge_index"][1].astype(np.int64)
            weights = data["cs_edge_attr"]
            result = [(f"all samples", weights)]
            for cls in grid_present_classes:
                mask = labels[dst] == cls
                result.append((f"class {cls}", weights[mask] if weights.size else np.array([])))
            return result

        def _extract_ss_eval_weights(split: str) -> list[tuple[str, np.ndarray]]:
            data = cache[split]
            labels = data["sample_labels"]
            if labels is None or data["ss_edge_index"] is None or data["ss_edge_attr"] is None:
                return [(f"eval (all)", np.array([])) for _ in range(grid_ncols)]
            dst = data["ss_edge_index"][1].astype(np.int64)
            eval_mask = dst >= data["n_train"]
            weights = data["ss_edge_attr"]
            base_vals = weights[eval_mask] if eval_mask.any() else np.array([])
            results = [(f"eval (all)", base_vals)]
            for cls in grid_present_classes:
                cls_mask = eval_mask & (labels[dst] == cls)
                cls_vals = weights[cls_mask] if weights.size else np.array([])
                results.append((f"eval class {cls}", cls_vals))
            return results

        cs_weights_rows = {split: _extract_cs_weights(split) for split in ["val", "test"]}
        _plot_grid(
            "cs_edge_weights.png",
            cs_weights_rows,
            grid_ncols,
            "CS edge weights",
            "weight",
            "count",
            bins,
            top_text=top_text,
        )

        cs_out_rows = {split: _extract_cs_out_degree(split) for split in ["val", "test"]}
        _plot_grid(
            "cs_out_degree.png",
            cs_out_rows,
            grid_ncols,
            "CS out-degree (classifier)",
            "out-degree",
            "count",
            bins,
            top_text=top_text,
        )

        cs_out_summary: Dict[str, Dict[str, list[dict[str, Any]]]] = {}
        for split, entries in cs_out_rows.items():
            summary: Dict[str, list[dict[str, Any]]] = {}
            for entry in entries:
                summary[entry["title"]] = entry.get("top_summary", [])
            cs_out_summary[split] = summary
        with open(plots_dir / "cs_out_degree_top.json", "w") as jf:
            json.dump(cs_out_summary, jf, indent=2)

        label_array = sample_labels if sample_labels is not None else np.array([])

        def _build_ss_row(label: str, mask: np.ndarray) -> list[tuple[str, np.ndarray]]:
            weights = ss_edge_attr if ss_edge_attr is not None else np.array([])
            row_values: list[tuple[str, np.ndarray]] = []
            base_vals = weights[mask] if mask.any() and weights.size else np.array([])
            row_values.append((f"{label} (all)", base_vals))
            for cls in grid_present_classes:
                cls_mask = mask & (label_array[dst] == cls) if label_array.size else np.zeros_like(mask, dtype=bool)
                cls_vals = weights[cls_mask] if weights.size and cls_mask.any() else np.array([])
                row_values.append((f"{label} class {cls}", cls_vals))
            return row_values

        def _ss_mask(data: Dict[str, Any], split_label: str) -> np.ndarray:
            ss_idx = data.get("ss_edge_index")
            if ss_idx is None or ss_idx.size == 0:
                return np.zeros(0, dtype=bool)
            dst = ss_idx[1].astype(np.int64)
            n_train = int(data.get("n_train", 0))
            n_eval = int(data.get("n_eval", 0))
            if split_label in {"val", "test"}:
                return (dst >= n_train) & (dst < n_train + n_eval)
            if split_label == "train":
                return dst < n_train
            return np.zeros_like(dst, dtype=bool)

        def _build_ss_row(split_label: str, data: Dict[str, Any]) -> list[tuple[str, np.ndarray]]:
            column_labels = [f"{split_label.upper()} • all"] + [
                f"{split_label.upper()} class {cls}" for cls in grid_present_classes
            ]
            if data is None:
                return [(label, np.array([])) for label in column_labels]
            ss_idx = data.get("ss_edge_index")
            ss_attr = data.get("ss_edge_attr")
            if ss_idx is None or ss_idx.size == 0 or ss_attr is None:
                weights = np.array([])
                dst = np.zeros(0, dtype=np.int64)
            else:
                weights = ss_attr
                dst = ss_idx[1].astype(np.int64)
            mask = _ss_mask(data, split_label)
            label_array = data.get("sample_labels")
            label_array = label_array if label_array is not None else np.array([])
            row_values: list[tuple[str, np.ndarray]] = []
            base_vals = weights[mask] if mask.size and weights.size else np.array([])
            row_values.append((column_labels[0], base_vals))
            for idx, cls in enumerate(grid_present_classes, start=1):
                if mask.size and weights.size and dst.size and label_array.size:
                    cls_mask = mask & (label_array[dst] == cls)
                else:
                    cls_mask = np.zeros_like(mask, dtype=bool)
                cls_vals = weights[cls_mask] if weights.size and cls_mask.size and cls_mask.any() else np.array([])
                row_values.append((column_labels[idx], cls_vals))
            return row_values

        if cache.get("val") or cache.get("test"):
            ss_rows: Dict[str, list[tuple[str, np.ndarray]]] = {}
            if cache.get("val"):
                ss_rows["val"] = _build_ss_row("val", cache["val"])
            if cache.get("test"):
                ss_rows["test"] = _build_ss_row("test", cache["test"])
            train_cache = cache.get("val") or cache.get("test")
            ss_rows["train"] = _build_ss_row("train", train_cache)
            _plot_grid(
                "ss_edge_weights.png",
                ss_rows,
                grid_ncols,
                "SS edge weights",
                "weight",
                "count",
                bins,
                row_order=["val", "test", "train"],
            )




def save_meta_histograms(
    self,
    meta_by_split: Dict[str, torch.Tensor],
    labels_by_split: Dict[str, torch.Tensor],
) -> None:
    split_names = ["train", "val", "test"]
    labels_np = {
        split: labels_by_split[split].detach().cpu().numpy()
        for split in split_names
    }
    meta_np = {
        split: meta_by_split[split].detach().cpu().numpy().astype(np.int32)
        for split in split_names
    }

    total_class_counts = np.zeros(self.args.num_classes, dtype=np.int64)
    for split in split_names:
        lbls = labels_np[split]
        if lbls.size:
            total_class_counts += np.bincount(lbls, minlength=self.args.num_classes)
    present_classes = [cls for cls, count in enumerate(total_class_counts) if count > 0]

    rows = 1 + len(present_classes)  # overall row + one row per present class
    cols = 3  # train / val / test
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.atleast_2d(axes)

    def _infer_classifier_count() -> int:
        for split in ("train", "val", "test"):
            meta = meta_by_split.get(split)
            if meta is not None and meta.numel() > 0:
                return meta.shape[1]
        return self.num_models

    num_classifiers = _infer_classifier_count()
    bins = np.arange(num_classifiers + 2, dtype=float) - 0.5
    empty_text_kw = dict(
        ha="center",
        va="center",
        fontsize=12,
        color="#C00000",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#C00000"),
    )

    for col, split in enumerate(split_names):
        meta = meta_np[split]
        labels = labels_np[split]
        correct_counts = meta.sum(axis=1)

        ax = axes[0, col]
        if correct_counts.size:
            ax.hist(correct_counts, bins=bins, color="#4C72B0", edgecolor="black")
        else:
            ax.text(0.5, 0.5, "No samples", transform=ax.transAxes, **empty_text_kw)
            ax.tick_params(left=False, labelleft=False)
            ax.tick_params(bottom=False, labelbottom=False)
        ax.set_title(f"{split.title()} • all classes (N={correct_counts.size})")
        ax.set_ylabel("Samples")
        ax.set_xlim(-0.5, num_classifiers + 0.5)

        for row_idx, cls in enumerate(present_classes, start=1):
            ax_cls = axes[row_idx, col]
            cls_mask = labels == cls
            if np.any(cls_mask):
                cls_counts = meta[cls_mask].sum(axis=1)
                ax_cls.hist(cls_counts, bins=bins, color="#55A868", edgecolor="black")
                n_cls = int(cls_mask.sum())
                ax_cls.set_title(f"{split.title()} • class {cls} (N={n_cls})")
            else:
                ax_cls.text(0.5, 0.5, "No samples", transform=ax_cls.transAxes, **empty_text_kw)
                ax_cls.tick_params(left=False, labelleft=False)
                ax_cls.tick_params(bottom=False, labelbottom=False)
                ax_cls.set_title(f"{split.title()} • class {cls} (N=0)")

            ax_cls.set_xlim(-0.5, num_classifiers + 0.5)

            if col == 0:
                ax_cls.set_ylabel(f"class {cls}")

    axes[-1, col].set_xlabel("# correct classifiers")

    fig.tight_layout()

    out_path = self.graph_outputs_dir / "meta_label_histograms.png"
    fig.savefig(out_path, dpi=150)

    plt.close(fig)

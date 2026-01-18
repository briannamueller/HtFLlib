#!/usr/bin/env python3
"""
make_phase_configs.py

Generate flat config files for FedDES/FedPAE phases and, if a checkpoint
root is provided, synchronize *_DONE.marker files based on actual filesystem
state.

Outputs in --out-dir:

- configs_base.txt   : one CLI line per base config
- configs_graph.txt  : one CLI line per graph config
- configs_gnn.txt    : one CLI line per gnn/meta config
- configs_pae.txt    : one CLI line per PAE config

- base_ids.txt       : fingerprint for each base config line
- graph_ids.txt      : fingerprint for each graph config line
- gnn_ids.txt        : fingerprint for each gnn config line
- pae_ids.txt        : fingerprint for each pae config line

If --ckpt-root is provided, also:

- base_clf/base[BASE_FP]_DONE.marker   (for complete base configs)
- graphs/base[BASE_FP]_graph[GRAPH_FP]_DONE.marker
- gnn/base[BASE_FP]_graph[GRAPH_FP]_gnn[GNN_FP]_DONE.marker
- pae/base[BASE_FP]_pae[PAE_FP]_DONE.marker

“Complete” is defined entirely via existing files:
- base: all expected model files for all clients are present
- graph: all expected graph + bundle files for all clients are present
- gnn: aggregated CSV exists
- pae: aggregated CSV exists

No changes to main.py are required.
"""

import argparse
import itertools
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import yaml


# ---------------------------------------------------------------------------
# Helpers: expand YAML sections into cartesian products of sweep params
# ---------------------------------------------------------------------------

def _expand_sweep(base_cfg: Dict[str, Any], sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not sweep:
        return [base_cfg]
    keys = list(sweep.keys())
    values_lists: List[List[Any]] = []
    for k in keys:
        v = sweep[k]
        values_lists.append(v if isinstance(v, list) else [v])
    configs: List[Dict[str, Any]] = []
    for combo in itertools.product(*values_lists):
        cfg = base_cfg.copy()
        for k, v in zip(keys, combo):
            cfg[k] = v
        configs.append(cfg)
    return configs


def _apply_conditional(
    cfg: Dict[str, Any],
    conditional: List[Dict[str, Any]],
    single_config: bool,
) -> List[Dict[str, Any]]:
    if not conditional:
        return [cfg]
    expanded: List[Dict[str, Any]] = []
    matched = False
    for rule in conditional:
        when = rule.get("when") or {}
        if when and not all(cfg.get(k) == v for k, v in when.items()):
            continue
        matched = True
        next_cfg = cfg.copy()
        for k, v in (rule.get("set") or {}).items():
            next_cfg[k] = v
        rule_sweep = rule.get("sweep") or {}
        if not rule_sweep:
            expanded.append(next_cfg)
            continue
        if single_config:
            picked = {
                k: (v[0] if isinstance(v, list) else v)
                for k, v in rule_sweep.items()
            }
            merged = next_cfg.copy()
            merged.update(picked)
            expanded.append(merged)
        else:
            expanded.extend(_expand_sweep(next_cfg, rule_sweep))
    if not matched:
        expanded.append(cfg)
    return expanded


def expand_section(section: Dict[str, Any], single_config: bool = False) -> List[Dict[str, Any]]:
    """
    Expand a section dict with optional 'sweep' into a list of concrete configs.
    Optionally apply 'conditional_sweep' rules.

    Supported keys:
      - sweep: dict of cartesian product values
      - conditional_sweep: list of rules:
          - when: {key: value, ...}
          - set: {key: value, ...}          # overrides
          - sweep: {key: [values], ...}     # conditional cartesian product
    """
    defaults = {
        k: v
        for k, v in section.items()
        if k not in {"sweep", "conditional_sweep"}
    }
    sweep = section.get("sweep") or {}
    conditional = section.get("conditional_sweep") or []

    if single_config:
        base_cfgs = [defaults]
    else:
        base_cfgs = _expand_sweep(defaults, sweep)

    configs: List[Dict[str, Any]] = []
    for cfg in base_cfgs:
        configs.extend(_apply_conditional(cfg, conditional, single_config))
    return configs


def defaults_only(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return a single config containing only the default keys from the section,
    applying conditional_sweep rules by choosing the first value in each sweep.
    """
    return expand_section(section, single_config=True)


def dict_to_cli(d: Dict[str, Any]) -> str:
    """
    Convert a config dict to a CLI string:

      {"base_clf_lr": 0.0005, "base_split_mode": "split_train"} ->
      "--base_clf_lr 0.0005 --base_split_mode split_train"
    """
    parts: List[str] = []
    for k, v in d.items():
        if k == "sweep":
            continue
        parts.append(f"--{k}")
        if isinstance(v, bool):
            parts.append("true" if v else "false")
        else:
            parts.append(str(v))
    return " ".join(parts)


def fingerprint(cfg: Dict[str, Any], prefix: str) -> str:
    """
    Compute a short MD5 fingerprint of the subset of cfg whose keys
    start with `prefix`. This mirrors the idea of derive_config_ids.
    """
    sub = {k: v for k, v in cfg.items() if isinstance(k, str) and k.startswith(prefix)}
    if not sub:
        return "none"
    blob = json.dumps(sub, sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Completeness checks based on num_clients & num_models
# ---------------------------------------------------------------------------

def base_complete(
    ckpt_root: Path,
    base_fp: str,
    num_clients: int,
    num_models: int,
) -> bool:
    """
    A base config (fingerprint = base_fp) is considered complete if, under:

      base_clf/base[base_fp]/,

    we have at least num_clients distinct client prefixes, and for each such
    client we see ALL expected model indices 0..num_models-1 as files:

      {client_prefix}_model_0.pt
      ...
      {client_prefix}_model_{num_models-1}.pt

    We do NOT assume anything about the exact client name pattern; we just
    derive prefixes from '*_model_*.pt' and check per-prefix indices.
    """
    base_dir = ckpt_root / "base_clf" / f"base[{base_fp}]"
    if not base_dir.exists():
        return False

    model_files = list(base_dir.glob("*_model_*.pt"))
    if not model_files:
        return False

    # Map: client_prefix -> set(model_indices)
    clients: Dict[str, set] = {}
    for p in model_files:
        name = p.name
        # Split on the last occurrence of "_model_"
        try:
            prefix, idx_str = name.rsplit("_model_", 1)
            if not idx_str.endswith(".pt"):
                continue
            idx = int(idx_str[:-3])  # strip ".pt"
        except Exception:
            continue
        clients.setdefault(prefix, set()).add(idx)

    if not clients:
        return False

    # Require at least num_clients clients
    if len(clients) < num_clients:
        return False

    expected_indices = set(range(num_models))
    for prefix, idxs in clients.items():
        # We require that this client has at least the indices 0..num_models-1.
        if not expected_indices.issubset(idxs):
            return False
        
    for client in clients:
        bundle_path = base_dir / f"{client}_graph_bundle.pt"
        if not bundle_path.exists():
            return False

    return True


def graph_complete(
    ckpt_root: Path,
    base_fp: str,
    graph_fp: str,
    num_clients: int,
) -> bool:
    """
    A graph config (base_fp, graph_fp) is considered complete if, under:

      graphs/base[base_fp]_graph[graph_fp]/
      base_clf/base[base_fp]/,

    for at least num_clients distinct client prefixes we have:

      graphs/.../{client}_graph_train_val.pt
      graphs/.../{client}_graph_train_test.pt
      base_clf/.../{client}_graph_bundle.pt
    """
    base_dir = ckpt_root / "base_clf" / f"base[{base_fp}]"
    graph_dir = ckpt_root / "graphs" / f"base[{base_fp}]_graph[{graph_fp}]"

    if not (base_dir.exists() and graph_dir.exists()):
        return False

    val_files = list(graph_dir.glob("*_graph_train_val.pt"))
    test_files = list(graph_dir.glob("*_graph_train_test.pt"))

    if not val_files or not test_files:
        return False

    def client_prefix_from_suffix(p: Path, suffix: str) -> str:
        name = p.name
        if name.endswith(suffix):
            return name[: -len(suffix)]
        return name

    val_clients = {client_prefix_from_suffix(p, "_graph_train_val.pt") for p in val_files}
    test_clients = {client_prefix_from_suffix(p, "_graph_train_test.pt") for p in test_files}
    all_clients = val_clients | test_clients

    if len(all_clients) < num_clients:
        return False

    # For every discovered client, require val, test, and bundle
    for client in all_clients:
        val_path = graph_dir / f"{client}_graph_train_val.pt"
        test_path = graph_dir / f"{client}_graph_train_test.pt"
        if not (val_path.exists() and test_path.exists()):
            return False

    return True


def gnn_complete(
    ckpt_root: Path,
    base_fp: str,
    graph_fp: str,
    gnn_fp: str,
) -> bool:
    """
    A GNN/meta config (base_fp, graph_fp, gnn_fp) is considered complete if the
    aggregated CSV from FedDES exists:

      gnn/base[base_fp]_graph[graph_fp]_gnn[gnn_fp].csv
    """
    csv = ckpt_root / "gnn" / f"base[{base_fp}]_graph[{graph_fp}]_gnn[{gnn_fp}].csv"
    return csv.exists()


def pae_complete(
    ckpt_root: Path,
    base_fp: str,
    pae_fp: str,
) -> bool:
    """
    A PAE config (base_fp, graph_fp, pae_fp) is considered complete if the
    aggregated CSV from FedPAE exists:

      pae/base[base_fp]_pae[pae_fp]/results.csv
    """
    csv = ckpt_root / "pae" / f"base[{base_fp}]_pae[{pae_fp}]" / "results.csv"
    return csv.exists()


def sync_markers(
    ckpt_root: Path,
    base_ids: List[str],
    graph_ids: List[str],
    gnn_ids: List[str],
    pae_ids: List[str],
    num_clients: int,
    num_models: int,
) -> None:
    """
    For every (base_fp), (base_fp, graph_fp), and (base_fp, graph_fp, gnn_fp)
    combination implied by the config grids, create or remove *_DONE.marker
    files so that markers always reflect the actual filesystem state.

    - Base markers:  base_clf/base[BASE_FP]_DONE.marker
    - Graph markers: graphs/base[BASE_FP]_graph[GRAPH_FP]_DONE.marker
    - GNN markers:   gnn/base[BASE_FP]_graph[GRAPH_FP]_gnn[GNN_FP]_DONE.marker
    - PAE markers:   pae/base[BASE_FP]_pae[PAE_FP]_DONE.marker
    """
    ckpt_root = ckpt_root.expanduser()

    # --- Base markers ---
    for b in base_ids:
        marker = ckpt_root / "base_clf" / f"base[{b}]_DONE.marker"
        if base_complete(ckpt_root, b, num_clients, num_models):
            if not marker.exists():
                marker.write_text(f"base_id={b}\nstatus=complete\n")
        else:
            if marker.exists():
                marker.unlink()

    # --- Graph markers ---
    for b in base_ids:
        for g in graph_ids:
            marker = ckpt_root / "graphs" / f"base[{b}]_graph[{g}]_DONE.marker"
            if graph_complete(ckpt_root, b, g, num_clients):
                if not marker.exists():
                    marker.write_text(f"base_id={b}\ngraph_id={g}\nstatus=complete\n")
            else:
                if marker.exists():
                    marker.unlink()

    # --- GNN markers ---
    for b in base_ids:
        for g in graph_ids:
            for n in gnn_ids:
                marker = ckpt_root / "gnn" / f"base[{b}]_graph[{g}]_gnn[{n}]_DONE.marker"
                if gnn_complete(ckpt_root, b, g, n):
                    if not marker.exists():
                        marker.write_text(
                            f"base_id={b}\ngraph_id={g}\ngnn_id={n}\nstatus=complete\n"
                        )
                else:
                    if marker.exists():
                        marker.unlink()

    # --- PAE markers ---
    for b in base_ids:
        for p in pae_ids:
            marker = ckpt_root / "pae" / f"base[{b}]_pae[{p}]_DONE.marker"
            if pae_complete(ckpt_root, b, p):
                if not marker.exists():
                    marker.write_text(
                        f"base_id={b}\npae_id={p}\nstatus=complete\n"
                    )
            else:
                if marker.exists():
                    marker.unlink()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs-yaml",
        type=Path,
        default=Path("system/conf/configs.yaml"),
        help="Path to configs (default: system/conf/configs.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("run_configs"),
        help="Output directory for configs_*.txt and *_ids.txt",
    )
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=None,
        help="Checkpoint root; if provided, *_DONE.marker files will be "
             "synchronized with actual filesystem state.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        required=True,
        help="Number of clients (nc) to expect artifacts for.",
    )
    parser.add_argument(
        "--num-models",
        type=int,
        required=True,
        help="Number of base models per client (for base completeness checks).",
    )
    parser.add_argument(
        "--single-config",
        action="store_true",
        help="Emit one config per section using only the defaults (ignore sweep).",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with args.configs_yaml.open("r") as f:
        root = yaml.safe_load(f)
    base_section = root["base_configs"]
    graph_section = root["graph_configs"]
    gnn_section = root["gnn_configs"]
    pae_section = root.get("pae_configs") or {}

    if args.single_config:
        base_configs = defaults_only(base_section)
        graph_configs = defaults_only(graph_section)
        gnn_configs = defaults_only(gnn_section)
        pae_configs = defaults_only(pae_section) if pae_section else [{}]
    else:
        base_configs = expand_section(base_section)
        graph_configs = expand_section(graph_section)
        gnn_configs = expand_section(gnn_section)
        pae_configs = expand_section(pae_section) if pae_section else [{}]

    # Write CLI configs
    base_path = out_dir / "configs_base.txt"
    graph_path = out_dir / "configs_graph.txt"
    gnn_path = out_dir / "configs_gnn.txt"
    pae_path = out_dir / "configs_pae.txt"

    base_path.write_text("\n".join(dict_to_cli(c) for c in base_configs) + "\n")
    graph_path.write_text("\n".join(dict_to_cli(c) for c in graph_configs) + "\n")
    gnn_path.write_text("\n".join(dict_to_cli(c) for c in gnn_configs) + "\n")
    pae_path.write_text("\n".join(dict_to_cli(c) for c in pae_configs) + "\n")

    print(f"Wrote {len(base_configs)} base configs to {base_path}")
    print(f"Wrote {len(graph_configs)} graph configs to {graph_path}")
    print(f"Wrote {len(gnn_configs)} gnn/meta configs to {gnn_path}")
    print(f"Wrote {len(pae_configs)} pae configs to {pae_path}")

    # Write fingerprints
    base_ids = [fingerprint(cfg, "base") for cfg in base_configs]
    graph_ids = [fingerprint(cfg, "graph") for cfg in graph_configs]
    gnn_ids = [fingerprint(cfg, "gnn") for cfg in gnn_configs]
    pae_ids = [fingerprint(cfg, "pae") for cfg in pae_configs]

    base_ids_path = out_dir / "base_ids.txt"
    graph_ids_path = out_dir / "graph_ids.txt"
    gnn_ids_path = out_dir / "gnn_ids.txt"
    pae_ids_path = out_dir / "pae_ids.txt"

    base_ids_path.write_text("\n".join(base_ids) + ("\n" if base_ids else ""))
    graph_ids_path.write_text("\n".join(graph_ids) + ("\n" if graph_ids else ""))
    gnn_ids_path.write_text("\n".join(gnn_ids) + ("\n" if gnn_ids else ""))
    pae_ids_path.write_text("\n".join(pae_ids) + ("\n" if pae_ids else ""))

    print(f"Wrote {len(base_ids)} base IDs to {base_ids_path}")
    print(f"Wrote {len(graph_ids)} graph IDs to {graph_ids_path}")
    print(f"Wrote {len(gnn_ids)} gnn IDs to {gnn_ids_path}")
    print(f"Wrote {len(pae_ids)} pae IDs to {pae_ids_path}")

    # Optional: sync *_DONE.marker files based on actual artifacts
    if args.ckpt_root is not None:
        print(f"Syncing *_DONE.marker files under ckpt_root={args.ckpt_root} ...")
        sync_markers(
            args.ckpt_root,
            base_ids,
            graph_ids,
            gnn_ids,
            pae_ids,
            num_clients=args.num_clients,
            num_models=args.num_models,
        )
        print("Marker sync complete.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List


import wandb
import matplotlib.pyplot as plt


PHASE3_META_PLOTS = [
    ("val_acc", "val_acc", "Validation accuracy"),
    ("val_bacc", "val_bacc", "Validation balanced accuracy"),
    ("ensemble_size", "val_ensemble_size", "Validation ensemble size"),
]


def filter_wandb_keys(kv: Dict[str, Any]) -> Dict[str, Any]:
    included_keys = {
        "algorithm", "dataset", "num_clients",
        "model_family", "models", "feature_dim", "batch_size",
    }
    prefixes = ("base_", "graph_", "gnn_", "pae_", "proto_")

    return {
        k: v
        for k, v in kv.items()
        if (k in included_keys or any(k.startswith(p) for p in prefixes))
    }


def _safe_wandb_init(**kwargs):
    if wandb is None:
        return None
    attempts = 3
    delay = 1.0
    for i in range(attempts):
        try:
            try:
                settings = wandb.Settings(
                    redirect_stdout=False,
                    redirect_stderr=False,
                    console="off",
                )
            except Exception:
                settings = wandb.Settings(console="off")
            return wandb.init(settings=settings, **kwargs)
        except Exception as exc:
            if i == attempts - 1:
                raise
            err = str(exc).lower()
            if "redirect_stdout" in err or "redirect_stderr" in err:
                try:
                    settings = wandb.Settings(console="off")
                    return wandb.init(settings=settings, **kwargs)
                except Exception:
                    pass
            if "409" not in err and "upsertbucket" not in err:
                raise
            time.sleep(delay)
            delay *= 2
    return wandb.init(**kwargs)


def setup_phase3_wandb(args: Any) -> bool:
    if wandb is None:
        return False

    manual_run = os.getenv("WANDB_MANUAL_RUN", "").lower() not in {"", "0", "false"}
    is_sweep = bool(os.getenv("WANDB_SWEEP_ID"))
    log_to_wandb = is_sweep or manual_run

    if not log_to_wandb:
        return False

    if wandb.run is None:
        cfg = filter_wandb_keys(vars(args))
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            job_type="train_meta_learner",
            config=cfg,
        )

    final_cfg = filter_wandb_keys(dict(wandb.config))
    for k, v in final_cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)

    lines = ["[FedDES][W&B] Config:"]
    for key in sorted(final_cfg):
        lines.append(f"  {key}: {final_cfg[key]}")
    print("\n".join(lines))

    return True


def setup_pae_wandb(args: Any) -> bool:
    if wandb is None:
        return False

    manual_run = os.getenv("WANDB_MANUAL_RUN", "").lower() not in {"", "0", "false"}
    is_sweep = bool(os.getenv("WANDB_SWEEP_ID"))
    log_to_wandb = is_sweep or manual_run

    if not log_to_wandb:
        return False

    if wandb.run is None:
        cfg = filter_wandb_keys(vars(args))
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            job_type="run_ensemble_selection",
            config=cfg,
        )

    final_cfg = filter_wandb_keys(dict(wandb.config))
    for k, v in final_cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)

    lines = ["[FedPAE][W&B] Config:"]
    for key in sorted(final_cfg):
        lines.append(f"  {key}: {final_cfg[key]}")
    print("\n".join(lines))

    return True


def log_phase3_results(
    csv_path: Path,
    gnn_alias: str,
    base_id: str,
    graph_id: str,
    gnn_id: str,
    aggregate: Dict[str, Any],
    meta_histories: Dict[str, List[Dict[str, Any]]] | None = None,
    clients: List[Any] | None = None,
) -> None:
    if wandb is None or wandb.run is None:
        return

    run = wandb.run
    metric_name = os.getenv("WANDB_SWEEP_METRIC", "bacc_beats_baselines")

    with csv_path.open("r", newline="") as f:
        reader = list(csv.reader(f))
    if len(reader) < 2:
        return

    table_cols, table_rows = reader[0], reader[1:]
    wb_table = wandb.Table(columns=table_cols, data=table_rows)
    art = wandb.Artifact(
        name="results_table",
        type="table",
        metadata={"gnn_id": gnn_id, "base_id": base_id, "graph_id": graph_id},
    )

    try:
        run.log({f"{gnn_alias}": wb_table})
        art.add_file(str(csv_path), name=csv_path.name)
        run.log_artifact(art, aliases=[gnn_alias])
        run.log({metric_name: aggregate.get(metric_name)})

        summary_keys = [
            "bacc_beats_baselines",
            "acc_beats_baselines",
            "mean_FedDES_acc",
            "mean_FedDES_bacc",
        ]
        summary_payload = {
            key: aggregate.get(key)
            for key in summary_keys
            if aggregate.get(key) is not None
        }
        if summary_payload:
            run.summary.update(summary_payload)

        extra_keys = [
            "mean_global_acc", "mean_global_bacc",
            "mean_local_acc", "mean_local_bacc",
            "acc_beats_local", "acc_beats_global",
            "bacc_beats_local", "bacc_beats_global",
        ]
        aggregate_json = {
            key: aggregate.get(key)
            for key in summary_keys + extra_keys
            if aggregate.get(key) is not None
        }
        if aggregate_json:
            json_path = csv_path.parent / "aggregate_metrics.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w") as jf:
                json.dump(aggregate_json, jf)
            art.add_file(str(json_path), name=json_path.name)
    except Exception:
        pass

    if meta_histories:
        _log_phase3_meta_plots(meta_histories, gnn_alias)

    if clients:
        _log_phase3_meta_selection(clients, gnn_alias)
        _log_phase3_saved_plots(clients, gnn_alias)


def log_pae_results(
    csv_path: Path,
    pae_alias: str,
    base_id: str,
    pae_id: str,
    aggregate: Dict[str, Any],
) -> None:
    if wandb is None or wandb.run is None:
        return

    run = wandb.run
    metric_name = os.getenv("WANDB_SWEEP_METRIC", "bacc_beats_baselines")

    with csv_path.open("r", newline="") as f:
        reader = list(csv.reader(f))
    if len(reader) < 2:
        return

    table_cols, table_rows = reader[0], reader[1:]
    wb_table = wandb.Table(columns=table_cols, data=table_rows)
    art = wandb.Artifact(
        name="results_table",
        type="table",
        metadata={"pae_id": pae_id, "base_id": base_id},
    )

    try:
        run.log({f"{pae_alias}": wb_table})
        art.add_file(str(csv_path), name=csv_path.name)
        run.log_artifact(art, aliases=[pae_alias])
        run.log({metric_name: aggregate.get(metric_name)})

        summary_keys = [
            "bacc_beats_baselines",
            "acc_beats_baselines",
            "mean_FedPAE_acc",
            "mean_FedPAE_bacc",
        ]
        summary_payload = {
            key: aggregate.get(key)
            for key in summary_keys
            if aggregate.get(key) is not None
        }
        if summary_payload:
            run.summary.update(summary_payload)

        extra_keys = [
            "mean_global_acc", "mean_global_bacc",
            "mean_local_acc", "mean_local_bacc",
            "acc_beats_local", "acc_beats_global",
            "bacc_beats_local", "bacc_beats_global",
        ]
        aggregate_json = {
            key: aggregate.get(key)
            for key in summary_keys + extra_keys
            if aggregate.get(key) is not None
        }
        if aggregate_json:
            json_path = csv_path.parent / "aggregate_metrics.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w") as jf:
                json.dump(aggregate_json, jf)
            art.add_file(str(json_path), name=json_path.name)
    except Exception:
        pass


def collect_graph_stats_rows(clients: List[Any]) -> List[Dict[str, Any]]:
    print('collect_graph_stats_rows called')
    def _read_stats(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open("r") as f:
                data = json.load(f)
        except Exception:
            return {}
        stats = data.get("stats")
        return stats if isinstance(stats, dict) else {}

    def _stat_field(stats: Dict[str, Any], key: str, field: str) -> float | None:
        value = stats.get(key)
        if isinstance(value, dict):
            return value.get(field)
        return None

    rows: List[Dict[str, Any]] = []
    for client in clients:
        val_stats = _read_stats(client.graph_outputs_dir / client.role / "texts" / "summary_val.json")
        test_stats = _read_stats(client.graph_outputs_dir / client.role / "texts" / "summary_test.json")
        base_stats = val_stats or test_stats or {}
        if not base_stats and not val_stats and not test_stats:
            print(f"wandb_utils: No stats found for client {client.role}")
            continue

        row: Dict[str, Any] = {"client": client.role}
        row["ss_edge_weights_all_min"] = _stat_field(base_stats, "ss_edge_weights_all", "min")
        row["ss_edge_weights_all_mean"] = _stat_field(base_stats, "ss_edge_weights_all", "mean")
        row["ss_edge_weights_all_max"] = _stat_field(base_stats, "ss_edge_weights_all", "max")

        row["ss_edge_weights_eval_val_min"] = _stat_field(val_stats, "ss_edge_weights_eval", "min")
        row["ss_edge_weights_eval_val_mean"] = _stat_field(val_stats, "ss_edge_weights_eval", "mean")
        row["ss_edge_weights_eval_val_max"] = _stat_field(val_stats, "ss_edge_weights_eval", "max")

        row["ss_edge_weights_eval_test_min"] = _stat_field(test_stats, "ss_edge_weights_eval", "min")
        row["ss_edge_weights_eval_test_mean"] = _stat_field(test_stats, "ss_edge_weights_eval", "mean")
        row["ss_edge_weights_eval_test_max"] = _stat_field(test_stats, "ss_edge_weights_eval", "max")

        row["classifier_out_degree_min"] = _stat_field(base_stats, "classifier_out_degree", "min")
        row["classifier_out_degree_mean"] = _stat_field(base_stats, "classifier_out_degree", "mean")
        row["classifier_out_degree_max"] = _stat_field(base_stats, "classifier_out_degree", "max")

        row["classifiers_with_no_edges_eval_val"] = val_stats.get("classifiers_with_no_edges") if val_stats else None
        row["classifiers_with_no_edges_eval_test"] = test_stats.get("classifiers_with_no_edges") if test_stats else None

        rows.append(row)
    return rows


def collect_classifier_perf_rows(clients: List[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for client in clients:
        path = Path(client.base_dir) / f"{client.role}_performance_baselines.json"
        if not path.exists():
            continue
        try:
            with path.open("r") as f:
                data = json.load(f)
        except Exception:
            continue
        for entry in data.get("individual_classifier_perf", []):
            classifier = entry.get("classifier")
            if classifier is None:
                continue
            rows.append({
                "client": client.role,
                "classifier": classifier,
                "home_client": entry.get("home_client"),
                "model": entry.get("model"),
                "acc": entry.get("acc"),
                "bacc": entry.get("bacc"),
            })
    return rows


def mean_summary(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if not rows:
        return summary
    keys = [k for k in rows[0].keys() if k != "client"]
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        summary[key] = sum(values) / len(values)
    return summary


def _read_classwise_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def _gather_cs_support_rows(clients: List[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for client in clients:
        json_path = Path(client.graph_outputs_dir) / client.role / "plots" / "cs_outdegree_classwise.json"
        data = _read_classwise_json(json_path)
        if not data:
            continue
        for entry in data.get("classwise_cs_support", []):
            target_class = entry.get("target_class")
            for clf in entry.get("top_classifiers", []):
                home_client = clf.get("home_client")
                model = clf.get("model")
                outdegree = clf.get("outdegree")
                support = clf.get("home_support")
                if target_class is None or home_client is None or model is None:
                    continue
                row = {
                    "target_client": client.role,
                    "target_class": int(target_class),
                    "home_client": home_client,
                    "model": model,
                    "outdegree": float(outdegree) if outdegree is not None else None,
                    "home_support": float(support) if support is not None else None,
                    "target_class_count": int(clf.get("target_class_count") or 0),
                    "class_fraction": float(clf.get("normalized_outdegree") or 0.0),
                }
                if isinstance(row["outdegree"], (int, float)) and isinstance(row["home_support"], (int, float)):
                    ratio = row["outdegree"] / (row["home_support"] + 1e-8) if row["home_support"] != 0 else None
                else:
                    ratio = None
                row["transfer_ratio"] = ratio
                rows.append(row)
    return rows


def _log_cs_support_series(clients: List[Any], run_name: str) -> None:
    if wandb is None or wandb.run is None:
        return
    rows = _gather_cs_support_rows(clients)
    if not rows:
        return

    columns = [
        "target_client",
        "target_class",
        "home_client",
        "model",
        "outdegree",
        "home_support",
        "transfer_ratio",
        "target_class_count",
        "class_fraction",
    ]
    table = wandb.Table(columns=columns, data=[[row.get(col) for col in columns] for row in rows])
    run = wandb.run
    try:
        scatter = wandb.plot.scatter(
            table,
            x="home_support",
            y="class_fraction",
            title=f"{run_name} CS support vs normalized outdegree",
        )
        run.log({f"{run_name}_cs_support_scatter": scatter})
    except Exception:
        pass


def log_graph_stats_to_wandb(
    clients: List[Any],
    args: Any,
    base_id: str,
    graph_id: str,
    run_name: str,
) -> None:
    if wandb is None:
        print("wandb_utils: wandb is None")
        return
    rows = collect_graph_stats_rows(clients)
    if not rows:
        return
    columns = [
        "client",
        "ss_edge_weights_all_min",
        "ss_edge_weights_all_mean",
        "ss_edge_weights_all_max",
        "ss_edge_weights_eval_val_min",
        "ss_edge_weights_eval_val_mean",
        "ss_edge_weights_eval_val_max",
        "ss_edge_weights_eval_test_min",
        "ss_edge_weights_eval_test_mean",
        "ss_edge_weights_eval_test_max",
        "classifier_out_degree_min",
        "classifier_out_degree_mean",
        "classifier_out_degree_max",
        "classifiers_with_no_edges_eval_val",
        "classifiers_with_no_edges_eval_test",
    ]

    table_data = [[row.get(col) for col in columns] for row in rows]
    table = wandb.Table(columns=columns, data=table_data)

    run = wandb.run
    created = False
    try:
        print(f"[W&B][Graph] Logging stats for {run_name}; existing run? {run is not None}", flush=True)
        if run is None:
            config = filter_wandb_keys(vars(args))
            config.update({
                "job_type": "graph_building",
                "batch_size": getattr(args, "batch_size", None),
                "feature_dim": getattr(args, "feature_dim", None),
            })
            print(f"[W&B][Graph] Initializing run for {run_name} with proj={os.getenv('WANDB_PROJECT')}", flush=True)
            run = _safe_wandb_init(
                project=os.getenv("WANDB_PROJECT"),
                entity=os.getenv("WANDB_ENTITY"),
                name=run_name,
                config=config,
                reinit=True,
            )
            if run is None:
                print(f"[W&B][Graph] _safe_wandb_init returned None for {run_name}", flush=True)
            created = True
        if run is None:
            print(f"[W&B][Graph] No run available for {run_name}", flush=True)
            return

        if run.name != run_name:
            try:
                run.name = run_name
            except Exception:
                pass

        run.log({f"{run_name}_graph_stats": table})
        print(f"[W&B][Graph] Logged graph stats table with {len(rows)} rows", flush=True)
        summary_payload = mean_summary(rows)
        if summary_payload:
            run.summary.update(summary_payload)
            print(f"[W&B][Graph] Summary payload for {run_name}: {summary_payload}", flush=True)
        run.summary.update({
            "graph_stats_rows": len(rows),
            "graph_stats_cols": len(columns),
        })
        stats_dir = Path(getattr(args, "outputs_root", Path("."))) / "graphs" / f"base[{base_id}]_graph[{graph_id}]" / "wandb"
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_csv = stats_dir / "graph_stats_table.csv"
        with stats_csv.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            for row in table_data:
                writer.writerow(row)
        aggregate_metrics = summary_payload.copy()
        aggregate_metrics.update(graph_stats_rows=len(rows), graph_stats_cols=len(columns))
        json_path = stats_dir / "aggregate_metrics.json"
        with json_path.open("w") as jf:
            json.dump(aggregate_metrics, jf, indent=2)
        stats_art = wandb.Artifact(
            name="graph_building_stats",
            type="table",
            metadata={"base_id": base_id, "graph_id": graph_id},
        )
        stats_art.add_file(str(stats_csv), name=stats_csv.name)
        stats_art.add_file(str(json_path), name=json_path.name)
        run.log_artifact(stats_art, aliases=[f"{run_name}_stats"])
        classifier_perf_rows = collect_classifier_perf_rows(clients)
        if classifier_perf_rows:
            perf_columns = [
                "client",
                "classifier",
                "home_client",
                "model",
                "acc",
                "bacc",
            ]
            perf_table = wandb.Table(
                columns=perf_columns,
                data=[[row.get(col) for col in perf_columns] for row in classifier_perf_rows],
            )
            run.log({f"{run_name}_classifier_perf": perf_table})
        _log_cs_support_series(clients, run_name)
        plot_groups: Dict[str, Dict[str, plt.Figure]] = {}
        for client in clients:
            plot_registry = getattr(client, "_graph_plot_registry", {})
            plot_entries = plot_registry.get(client.role, [])
            print(f"[W&B][Graph] Client {client.role} has {len(plot_entries)} registered plots", flush=True)
            if not plot_entries:
                continue
            for plot_name, fig in plot_entries:
                plot_groups.setdefault(plot_name, {})[client.role] = fig

        for plot_name, client_map in plot_groups.items():
            image_batch: Dict[str, wandb.Image] = {
                client_role: wandb.Image(fig)
                for client_role, fig in client_map.items()
            }
            plot_key = f"{plot_name}/"
            try:
                print(f"[W&B][Graph] Logging plots batch {plot_key}", flush=True)
                run.log({plot_key: image_batch})
            except Exception as exc:
                print(f"[W&B][Graph] Failed to log plots batch {plot_key}: {exc}", flush=True)
            finally:
                for fig in client_map.values():
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
    except Exception as exc:
        print(f"[W&B][Graph] Exception while logging {run_name}: {exc}", flush=True)
    finally:
        if created:
            try:
                wandb.finish()
            except Exception:
                pass


def _collect_meta_selection_rows(clients: List[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for client in clients:
        json_path = Path(client.graph_outputs_dir) / client.role / "plots" / "meta_selection.json"
        if not json_path.exists():
            continue
        try:
            with json_path.open("r") as jf:
                data = json.load(jf)
        except Exception:
            continue
        for row in data.get("rows", []):
            target_class = row.get("target_class")
            class_count = row.get("target_class_count")
            for entry in row.get("entries", []):
                selection_score = entry.get("selection_score")
                if selection_score is None:
                    continue
                homelen = entry.get("home_total") or 1
                rows.append({
                    "client": client.role,
                    "target_class": int(target_class) if target_class is not None else None,
                    "target_class_count": int(class_count) if class_count is not None else None,
                    "home_client": entry.get("home_client"),
                    "model": entry.get("model"),
                    "selection_score": float(selection_score),
                    "home_support_ratio": float(entry.get("home_support_ratio", 0.0)),
                    "home_support_count": int(entry.get("home_support_count", 0)),
                    "home_total": int(homelen),
                    "combination_mode": data.get("combination_mode"),
                })
    return rows


def _log_phase3_meta_selection(clients: List[Any], run_name: str) -> None:
    if wandb is None or wandb.run is None:
        return
    rows = _collect_meta_selection_rows(clients)
    if not rows:
        return

    columns = [
        "client",
        "target_class",
        "target_class_count",
        "home_client",
        "model",
        "selection_score",
        "home_support_ratio",
        "home_support_count",
        "home_total",
        "combination_mode",
    ]
    table = wandb.Table(columns=columns, data=[[row.get(col) for col in columns] for row in rows])
    run = wandb.run
    try:
        run.log({
            # f"{run_name}_meta_selection_table": table,
            f"{run_name}_meta_selection_scatter": wandb.plot.scatter(
                table,
                x="home_support_ratio",
                y="selection_score",
                title="meta selection vs home prevalence",
            ),
        })
    except Exception:
        pass


def _log_phase3_saved_plots(clients: List[Any], run_name: str) -> None:
    if wandb is None or wandb.run is None:
        return
    metrics = [
        # "train_loss",
        # "val_loss",
        # "val_acc",
        # "val_bacc",
        "support",
        "ensemble_size",
    ]
    run = wandb.run
    for metric in metrics:
        for client in clients:
            metric_root = Path(client.graph_outputs_dir) / client.role / "phase3_plots" / metric
            if not metric_root.exists():
                continue
            if metric == "support":
                for class_dir in sorted(metric_root.iterdir()):
                    if not class_dir.is_dir():
                        continue
                    img_path = class_dir / f"{client.role}.png"
                    if not img_path.exists():
                        continue
                    key = f"{metric}/{class_dir.name}/{client.role}"
                    try:
                        run.log({key: wandb.Image(str(img_path))})
                    except Exception:
                        pass
            else:
                img_path = metric_root / f"{client.role}.png"
                if not img_path.exists():
                    continue
                key = f"{metric}/{client.role}"
                try:
                    run.log({key: wandb.Image(str(img_path))})
                except Exception:
                    pass


def _log_phase3_meta_plots(
    meta_histories: Dict[str, List[Dict[str, Any]]],
    run_name: str,
) -> None:
    if wandb is None or wandb.run is None:
        return
    run = wandb.run

    for client_role, history in meta_histories.items():
        if not history:
            continue

        columns = sorted(
            {k for row in history for k in row.keys() if row.get(k) is not None}
        )
        table_data = [[row.get(col) for col in columns] for row in history]
        table = wandb.Table(columns=columns, data=table_data)

        # try:
        #     run.log({f"meta_history/{client_role}": table})
        # except Exception:
        #     pass
        plot_payload: Dict[str, Any] = {}
        for alias, key, title in PHASE3_META_PLOTS:
            if key not in columns:
                continue
            try:
                plot = wandb.plot.line(table, "epoch", key, title=f"{client_role} {title}")
            except Exception:
                continue
            plot_payload[f"{alias}/{client_role}"] = plot
        if plot_payload:
            try:
                run.log(plot_payload)
            except Exception:
                pass

        # Train vs val loss line-series per client.
        if "train_meta_loss" in columns and "val_loss" in columns:
            try:
                train_series = [row.get("train_meta_loss") for row in history]
                val_series = [row.get("val_loss") for row in history]
                if epochs and not any(v is None for v in train_series + val_series):
                    run.log({
                        f"loss/train_vs_val/{client_role}": wandb.plot.line_series(
                            xs=epochs,
                            ys=[train_series, val_series],
                            keys=["train", "val"],
                            title=f"{client_role} Train vs Val Loss",
                            xname="epoch",
                        )
                    })
            except Exception:
                pass

        # Log ESS line-series plots (train + val) as multi-line charts.
        epochs = [row.get("epoch") for row in history if row.get("epoch") is not None]
        if epochs:
            def _series(key_prefix: str):
                keys = ["q25", "mean", "q75"]
                kmap = {
                    "q25": f"ess_q25_{key_prefix}",
                    "mean": f"ess_mean_{key_prefix}",
                    "q75": f"ess_q75_{key_prefix}",
                }
                ys = []
                for k in keys:
                    series = [row.get(kmap[k]) for row in history]
                    if any(v is None for v in series):
                        return None
                    ys.append(series)
                return keys, ys

            for subset in ("train", "val"):
                series = _series(subset)
                if series is None:
                    continue
                keys, ys = series
                try:
                    run.log({
                        f"ess/{subset}/{client_role}": wandb.plot.line_series(
                            xs=epochs,
                            ys=ys,
                            keys=keys,
                            title=f"{client_role} ESS ({subset})",
                            xname="epoch",
                        )
                    })
                except Exception:
                    pass

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flcore.servers.serverbase import Server
from flcore.clients.clientdes_prototypes import clientDESPrototypes
from des.helpers import derive_config_ids, load_base_clf, run_stage
from des.wandb_utils import log_graph_stats_to_wandb, log_phase3_results, setup_phase3_wandb

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


class FedDES(Server):
    """
    FedDES server with 3 phases (base classifiers, graph construction, meta-learner).

    W&B integration (phase 3 only):
      - self.args is treated as the default hyperparameter config.
      - If W&B is enabled (sweep or manual run), we:
          1) call wandb.init(config=defaults_from_args)
          2) let W&B apply any overrides (sweeps, config files, UI, etc.)
          3) copy the final wandb.config values back into self.args
      - All later logic (IDs, logging, training) uses the *post-override* self.args.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        # Populate slow-client masks so set_clients builds all clients.
        self.set_slow_clients()
        self.set_clients(clientDESPrototypes)

        # Convenience names used for clf_key tuples.
        self.models = [f"model_{model_id}" for model_id in range(len(self.args.models))]

    # -------------------------
    # Main training workflow
    # -------------------------
    def _init_classifier_keys(self) -> None:
        """Initialize local and global classifier keys for all clients."""
        global_clf_keys: List[Tuple[str, str]] = []
        for client in self.clients:
            client.local_clf_keys = []
            for model_str in client.model_strs:
                clf_key = (client.role, model_str)
                global_clf_keys.append(clf_key)
                client.local_clf_keys.append(clf_key)

        self.global_clf_keys = global_clf_keys
        for client in self.clients:
            client.global_clf_keys = global_clf_keys

    def train(self):
        # Build global/local classifier keys once (used by clients and for classifier_pool).
        self._init_classifier_keys()

        phase = self.args.phase

        # -----------------------------
        # Phase 1: Base + graph bundle
        # -----------------------------
        if phase == 1:
            clients_needing_base = [c for c in self.clients if not c.base_classifiers_exist()]
            print("[FedDES][Server] Phase 1.1 starting: train_base_classifiers")
            run_stage(clients_needing_base, stage="train_base_classifiers")

            # Load all base classifiers into a shared pool for DS/meta-label projection.
            role_to_client = {c.role: c for c in self.clients}
            classifier_pool: Dict[Tuple[str, str], Any] = {}

            for client_role, model_str in self.global_clf_keys:
                client = role_to_client[client_role]
                classifier_pool[(client_role, model_str)] = load_base_clf(client, model_str)

            clients_needing_graph_prep = [c for c in self.clients if not c.graph_bundle_exists()]
            print("[FedDES][Server] Phase 1.2 starting: prepare_graph_data")
            run_stage(
                clients_needing_graph_prep,
                stage="prepare_graph_data",
                stage_inputs={"classifier_pool": classifier_pool},
            )
            return None

        # -----------------------------
        # Phase 2: Build graphs
        # -----------------------------
        if phase == 2:
            clients_missing_protos = [c for c in self.clients if c.missing_prototypes()]
            if clients_missing_protos:
                print("[FedDES][Server] Phase 2 precheck: prepare_graph_data (missing prototypes)")

                role_to_client = {c.role: c for c in self.clients}
                classifier_pool: Dict[Tuple[str, str], Any] = {}

                for client_role, model_str in self.global_clf_keys:
                    client = role_to_client[client_role]
                    classifier_pool[(client_role, model_str)] = load_base_clf(client, model_str)

                run_stage(
                    clients_missing_protos,
                    stage="prepare_graph_data",
                    stage_inputs={"classifier_pool": classifier_pool},
                )

            print("[FedDES][Server] Phase 2 starting: build_graph")
            clients_missing_graphs = [c for c in self.clients if not c.graphs_exist()]
            for client in clients_missing_graphs:
                client.build_graph()

            base_id, graph_id, _ = derive_config_ids(self.args)
            manual_wandb = os.getenv("WANDB_MANUAL_RUN", "").lower() not in {"", "0", "false"}
            is_sweep = (wandb is not None) and bool(os.getenv("WANDB_SWEEP_ID"))
            # if wandb is not None and (manual_wandb or is_sweep):
            if wandb is not None:
                print("serverdes: wandb is not None")
                log_graph_stats_to_wandb(
                    clients=self.clients,
                    args=self.args,
                    base_id=base_id,
                    graph_id=graph_id,
                    run_name=f"base[{base_id}]_graph[{graph_id}]",
                )
            else: print("serverdes: wandb is None")
            return None

        # -----------------------------
        # Phase 3: Train meta-learner
        # -----------------------------
        if phase == 3:
            # Handles:
            #   - deciding whether to log to W&B
            #   - wandb.init(config=defaults_from_args) if needed
            #   - syncing final wandb.config (with any overrides) back into self.args
            log_to_wandb = setup_phase3_wandb(self.args)

            # Derive IDs AFTER any W&B overrides, so gnn_id reflects the true params used.
            base_id, graph_id, gnn_id = derive_config_ids(self.args)
            gnn_alias = f"base[{base_id}]_graph[{graph_id}]_gnn[{gnn_id}]"
            csv_path = Path(self.args.ckpt_root) / "gnn" / gnn_alias / "results.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            skip_mode = getattr(self.args, "skip_meta_training", False)
            if not csv_path.exists() or not skip_mode:
                print("[FedDES][Server] Phase 3 starting: train_meta_learner")
                run_stage(self.clients, stage="train_meta_learner")
                self._log_results(csv_path)
            else:
                print(f"[FedDES][Server] Phase 3 skipped: results already exist at {csv_path}")

            meta_histories = {
                client.role: getattr(client, "meta_history", [])
                for client in self.clients
            }
            if log_to_wandb and csv_path.exists() and wandb.run is not None:
                wandb.run.name = gnn_alias
                log_phase3_results(
                    csv_path=csv_path,
                    gnn_alias=gnn_alias,
                    base_id=base_id,
                    graph_id=graph_id,
                    gnn_id=gnn_id,
                    aggregate=getattr(self, "perf_summary", {}),
                    meta_histories=meta_histories,
                    clients=self.clients,
                )

            return None

        raise ValueError(f"Unknown phase: {phase}")

    # -------------------------
    # CSV aggregation
    # -------------------------
    def _log_results(self, csv_path: Path) -> None:
        client_summaries = [(client.role, client.perf_summary) for client in self.clients]

        aggregate: Dict[str, Any] = {"n_clients": len(client_summaries)}
        mean_keys = ["local_acc", "local_bacc", "global_acc", "global_bacc", "FedDES_acc", "FedDES_bacc"]
        sum_keys = [
            "acc_beats_local", "bacc_beats_local",
            "acc_beats_global", "bacc_beats_global",
            "acc_beats_baselines", "bacc_beats_baselines",
        ]

        for key in mean_keys:
            vals = [s.get(key) for _, s in client_summaries]
            aggregate[f"mean_{key}"] = float(sum(vals) / len(vals))

        for key in sum_keys:
            vals = [s.get(key) for _, s in client_summaries]
            aggregate[f"{key}"] = int(sum(vals))

        excluded_table_cols = {"acc_beats_baselines", "bacc_beats_baselines"}
        all_metric_keys = {k for _, s in client_summaries for k in s}
        metric_cols = sorted(all_metric_keys - excluded_table_cols)
        table_cols = ["client"] + metric_cols

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(table_cols)
            # client rows only (aggregate stats handled separately)
            for role, summary in client_summaries:
                row = [role] + [summary.get(k) for k in metric_cols]
                writer.writerow(row)

        self.perf_summary = aggregate

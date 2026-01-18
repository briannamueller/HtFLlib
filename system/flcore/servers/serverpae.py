from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flcore.servers.serverbase import Server
from flcore.clients.clientpae import clientPAE
from des.helpers import derive_pae_config_ids, load_base_clf, run_stage
from des.wandb_utils import log_pae_results, setup_pae_wandb

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


class FedPAE(Server):


    def __init__(self, args, times):
        super().__init__(args, times)

        # Populate slow-client masks so set_clients builds all clients.
        self.set_slow_clients()
        self.set_clients(clientPAE)

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
        # Phase 1: Base + DS bundle
        # -----------------------------
        if phase == 1:
            clients_needing_base = [c for c in self.clients if not c.base_classifiers_exist()]
            if clients_needing_base:
                print("[FedPAE][Server] Phase 1.1 starting: train_base_classifiers")
                run_stage(clients_needing_base, stage="train_base_classifiers")

            # Load all base classifiers into a shared pool for DS/meta-label projection.
            role_to_client = {c.role: c for c in self.clients}
            classifier_pool: Dict[Tuple[str, str], Any] = {}

            for client_role, model_str in self.global_clf_keys:
                client = role_to_client[client_role]
                classifier_pool[(client_role, model_str)] = load_base_clf(client, model_str)

            clients_needing_graph_prep = [c for c in self.clients if not c.graph_bundle_exists()]
            if clients_needing_graph_prep:
                print("[FedPAE][Server] Phase 1.2 starting: prepare_graph_data")
                run_stage(
                    clients_needing_graph_prep,
                    stage="prepare_graph_data",
                    stage_inputs={"classifier_pool": classifier_pool},
                )
            else:
                print("[FedPAE][Server] All clients already have decision-space bundles.")
            return None

        # -----------------------------
        # Phase 2: NSGA-II ensemble search
        # -----------------------------
        elif phase == 2:
            log_to_wandb = setup_pae_wandb(self.args)

            base_id, pae_id = derive_pae_config_ids(self.args)
            pae_alias = f"base[{base_id}]_pae[{pae_id}]"
            csv_path = Path(self.args.ckpt_root) / "pae" / pae_alias / "results.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            skip_pae_training = getattr(self.args, "skip_pae_training", False)
            if skip_pae_training and csv_path.exists():
                print(f"[FedPAE][Server] Skipping PAE training because results exist at {csv_path}")
            else:
                print("[FedPAE][Server] Phase 2 starting: run_ensemble_selection")
                run_stage(self.clients, stage="run_ensemble_selection")
                self._log_results(csv_path)

            if log_to_wandb and csv_path.exists() and wandb is not None and wandb.run is not None:
                wandb.run.name = pae_alias
                log_pae_results(
                    csv_path=csv_path,
                    pae_alias=pae_alias,
                    base_id=base_id,
                    pae_id=pae_id,
                    aggregate=getattr(self, "perf_summary", {}),
                )
            return None

        else:
            print(f"[FedPAE][Server] Unknown phase {phase} specified. No action taken.")

    def _log_results(self, csv_path: Path) -> None:
        client_summaries = [(client.role, getattr(client, "perf_summary", {})) for client in self.clients]

        aggregate: Dict[str, Any] = {"n_clients": len(client_summaries)}
        mean_keys = ["local_acc", "local_bacc", "global_acc", "global_bacc", "FedPAE_acc", "FedPAE_bacc"]
        sum_keys = [
            "acc_beats_local",
            "bacc_beats_local",
            "acc_beats_global",
            "bacc_beats_global",
            "acc_beats_baselines",
            "bacc_beats_baselines",
        ]

        for key in mean_keys:
            vals = [s.get(key) for _, s in client_summaries]
            aggregate[f"mean_{key}"] = float(sum(vals) / len(vals)) if vals else float("nan")

        for key in sum_keys:
            vals = [s.get(key) for _, s in client_summaries]
            aggregate[f"{key}"] = int(sum(vals))

        excluded_table_cols = {"acc_beats_baselines", "bacc_beats_baselines"}
        all_metric_keys = {k for _, s in client_summaries for k in s}
        metric_cols = sorted(all_metric_keys - excluded_table_cols)
        table_cols = ["client"] + metric_cols

        with csv_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(table_cols)
            for role, summary in client_summaries:
                writer.writerow([role] + [summary.get(col) for col in metric_cols])

        self.perf_summary = aggregate

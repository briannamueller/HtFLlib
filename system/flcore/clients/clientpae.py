# from __future__ import annotations
# # system/flcore/clients/clientdes.py
# import json
# import random
# from types import SimpleNamespace

# import numpy as np
# import torch
# from sklearn.metrics import balanced_accuracy_score
# from pathlib import Path
# from types import SimpleNamespace
# from typing import Any, Dict, List, Optional
# import os
# import shutil
# import torch
# import torch.nn.functional as F
# import numpy as np
# import time
# import wandb
# import torch
# import torch.nn as nn
# import torch.nn.functional as F  # local import to support one-hot voting
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_score, f1_score
# import random
# from types import SimpleNamespace

# import numpy as np
# import torch
# from sklearn.metrics import balanced_accuracy_score
# # Shared training helpers

# # ---------- DES utilities ----------
# from des.graph_utils import build_train_eval_graph, project_to_DS
# from des.base_clf_utils import fit_clf
# from des.helpers import derive_config_ids, init_base_meta_loaders, get_performance_baselines

# from flcore.clients.clientbase import Client
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from des.graph_utils import project_to_DS
from des.base_clf_utils import fit_clf
from des.helpers import derive_pae_config_ids, init_base_meta_loaders
from flcore.clients.clientbase import Client

class clientPAE(Client):

    def __init__(self, args, id: int, train_samples: int, test_samples: int, **kwargs: Any) -> None:
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.args = args
        self.device = args.device
        total_models = len(args.models)
        if total_models == 0:
            raise ValueError("FedPAE expects at least one model in args.models.")
        self.base_single_model = getattr(args, "base_single_model", False)
        if self.base_single_model:
            model_ids = [self.id % total_models]
        else:
            model_ids = list(range(total_models))
        self.model_ids = model_ids
        self.model_strs = [f"model_{model_id}" for model_id in model_ids]
        self.num_models = len(model_ids)

        self.base_config_id, self.pae_config_id = derive_pae_config_ids(self.args)
        self.base_dir = args.ckpt_root / "base_clf" /  f"base[{self.base_config_id}]"
        self.base_outputs_dir = args.outputs_root / "base_clf" /  f"base[{self.base_config_id}]"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.base_outputs_dir.mkdir(parents=True, exist_ok=True)
        
        self._base_train_loader = None
        self._meta_train_loader = None
        self.meta_history: List[Dict[str, Any]] = []

    def base_classifiers_exist(self) -> bool:
        expected = [
            self.base_dir / f"{self.role}_{model_str}.pt"
            for model_str in self.model_strs
        ]
        statuses = {str(p): p.exists() for p in expected}
        ok = all(statuses.values()) if expected else True
        return ok
    
    def graph_bundle_exists(self) -> bool:
        bundle_path = Path(self.base_dir) / f"{self.role}_graph_bundle.pt"
        return bundle_path.exists()

    def train_base_classifiers(self, device=None):
        device = torch.device(device if device is not None else self.device)
        base_train_loader, _ = init_base_meta_loaders(self)
        val_loader = self.load_val_data()

        for model_id, model_str in zip(self.model_ids, self.model_strs):
            best_epoch, best_score, model= fit_clf(
                self,
                model_id, base_train_loader, val_loader, device,
                max_epochs=self.args.local_epochs,
                patience=self.args.base_es_patience,
                es_metric=self.args.base_es_metric,   # "val_loss" or "val_temp_scaled_loss"
                lr=self.args.base_clf_lr,
                min_delta=self.args.base_es_min_delta,
            )

            torch.save(model.cpu(), self.base_dir /f"{self.role}_{model_str}.pt")
            print(f"{self.role} {model_id} stopping training at epoch {best_epoch}, score = {best_score}")
    

    def prepare_graph_data(self, device=None, classifier_pool: Dict[Any, torch.nn.Module] = None) -> None:
        """
        Prepare decision-space and meta-data for graph building.

        Steps:
        1) Evaluate classifier pool → decision-space (DS), preds, labels,
            meta-labels, and sample features for train/val/test.
        2) Optionally load/save these tensors from/to a flat cache.
        """

        device = torch.device(device if device is not None else self.device)
        self.device = device

        _, meta_train_loader = init_base_meta_loaders(self)
        val_loader = self.load_val_data()
        test_loader = self.load_test_data()

        data_loaders = {"train": meta_train_loader, "val": val_loader, "test": test_loader}

        # Flat cache for graph inputs
        cache_path = (Path(self.base_dir)/ f"{self.role}_graph_bundle.pt")
        if cache_path.exists():
            print(f"[FedPAE][Client {self.role}] Loading cached artifacts from {cache_path}")
            graph_data = torch.load(cache_path, map_location="cpu")

        else:
            graph_data = {}
            for data_split in ["train", "val", "test"]:
                loader = data_loaders[data_split]
                ds, preds, y_true, meta_labels, feats = project_to_DS(
                    self, loader, classifier_pool
                )
                graph_data[data_split] = {
                    "ds": ds, "preds": preds, "y": y_true, "meta": meta_labels.float(), "feats": feats
                }
        
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(graph_data, cache_path)

    def _compute_ensemble_metrics(
        self,
        ds: torch.Tensor,
        y_true: torch.Tensor,
        selected_indices: list[int],
    ) -> dict[str, float]:
        """Compute accuracy and balanced accuracy for an ensemble.

        Args:
            ds: Decision-space predictions of shape [N, M * C], where
                N is #samples, M is #classifiers, C is #classes.
                Entries are per-class probabilities for each classifier.
            y_true: Ground-truth labels of shape [N].
            selected_indices: Indices of classifiers to include in the ensemble.

        Returns:
            A dict with keys "acc" and "bacc".
        """
        if len(selected_indices) == 0:
            return {"acc": 0.0, "bacc": 0.0}

        num_classes = self.args.num_classes
        N = ds.size(0)
        M = ds.size(1) // num_classes

        # [N, M, C]
        ds_reshaped = ds.view(N, M, num_classes)

        # Select classifiers and average their probabilities
        selected_ds = ds_reshaped[:, selected_indices, :]
        avg_probs = selected_ds.mean(dim=1)

        preds = avg_probs.argmax(dim=1)

        preds_np = preds.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()

        acc = (preds == y_true).float().mean().item()

        try:
            bacc = balanced_accuracy_score(y_true_np, preds_np)
        except ValueError:
            # e.g., only one class present in y_true
            bacc = float("nan")

        return {"acc": acc, "bacc": bacc}
    

    def run_ensemble_selection(self, device=None) -> None:
        """Run the FedPAE ensemble selection stage on this client.

        Steps (per FedPAE paper, client-side):
          1. Load cached decision-space predictions for val/test.
          2. Treat the global model bench as the candidate set.
          3. Use NSGA-II to optimize:
               - strength: average individual val accuracy of selected models,
               - diversity: average pairwise independence of their probability
                 outputs (1 - cosine similarity).
          4. From the Pareto set, choose the ensemble with highest ensemble
             val accuracy (using probability averaging for the ensemble).
          5. Evaluate this ensemble on test and compare to:
               - local-only ensemble,
               - global bench ensemble.
        """
        # Device
        device = torch.device(device if device is not None else self.device)
        self.device = device

        # Load DS bundle built by prepare_graph_data
        bundle_path = self.base_dir / f"{self.role}_graph_bundle.pt"
        if not bundle_path.exists():
            print(f"[FedPAE][Client {self.role}] No decision-space bundle at {bundle_path}; skipping ensemble selection.")
            return

        graph_data = torch.load(bundle_path, map_location="cpu")

        # Only val and test needed
        val = SimpleNamespace(**graph_data["val"])
        test = SimpleNamespace(**graph_data["test"])

        val.ds = val.ds.to(device)
        val.y = val.y.to(device)
        test.ds = test.ds.to(device)
        test.y = test.y.to(device)

        # If no validation data, fallback to baselines only
        if val.ds.size(0) == 0:
            print(f"[FedPAE][Client {self.role}] Empty validation set; computing baselines only.")
            num_classes = self.args.num_classes
            M = test.ds.size(1) // num_classes
            global_indices = list(range(M))

            M_global = len(getattr(self, "global_clf_keys", []))
            if M_global == M:
                local_indices = [idx for idx, (role, _) in enumerate(self.global_clf_keys) if role == self.role]
            else:
                local_indices = global_indices

            local_metrics = self._compute_ensemble_metrics(test.ds, test.y, local_indices)
            global_metrics = self._compute_ensemble_metrics(test.ds, test.y, global_indices)

            FedPAE_acc = 0.0
            FedPAE_bacc = 0.0

            self.perf_summary = {
                "local_acc": local_metrics["acc"],
                "local_bacc": local_metrics["bacc"],
                "global_acc": global_metrics["acc"],
                "global_bacc": global_metrics["bacc"],
                "FedPAE_acc": FedPAE_acc,
                "FedPAE_bacc": FedPAE_bacc,
                "acc_beats_local": 0,
                "bacc_beats_local": 0,
                "acc_beats_global": 0,
                "bacc_beats_global": 0,
                "acc_beats_baselines": 0,
                "bacc_beats_baselines": 0,
            }
            return

        # -------------------------
        # Precompute per-model stats on val set
        # -------------------------
        num_classes = self.args.num_classes
        N_val = val.ds.size(0)
        M = val.ds.size(1) // num_classes

        # Identify local vs global indices using global_clf_keys
        M_global = len(getattr(self, "global_clf_keys", []))
        if M_global == M:
            local_indices = [idx for idx, (role, _) in enumerate(self.global_clf_keys) if role == self.role]
        else:
            local_indices = list(range(M))

        val_ds_reshaped = val.ds.view(N_val, M, num_classes)
        val_probs = val_ds_reshaped.detach().cpu().numpy()  # [N_val, M, C]
        y_val_np = val.y.detach().cpu().numpy()

        # Individual accuracies (strength objective)
        indiv_acc = np.zeros(M, dtype=np.float32)
        for m in range(M):
            preds_m = val_probs[:, m, :].argmax(axis=1)
            indiv_acc[m] = (preds_m == y_val_np).mean()

        # Flattened, normalized probabilities for diversity (independence)
        flat_probs = []
        for m in range(M):
            vec = val_probs[:, m, :].reshape(-1).astype(np.float32)
            vec = vec - vec.mean()
            norm = np.linalg.norm(vec) + 1e-12
            vec = vec / norm
            flat_probs.append(vec)

        # Pairwise independence-based diversity: 1 - cosine similarity
        div_matrix = np.zeros((M, M), dtype=np.float32)
        for i in range(M):
            for j in range(i + 1, M):
                corr = float(np.dot(flat_probs[i], flat_probs[j]))
                independence = 1.0 - corr
                div_matrix[i, j] = independence
                div_matrix[j, i] = independence

        def evaluate_individual(mask: np.ndarray) -> tuple[float, float]:
            """Return (strength, diversity) for a chromosome mask."""
            selected = np.where(mask > 0)[0]
            if selected.size == 0:
                return 0.0, 0.0

            # Strength: average individual accuracy of selected models
            strength = float(indiv_acc[selected].mean())

            # Diversity: average pairwise independence among selected
            if selected.size < 2:
                diversity = 0.0
            else:
                sub = div_matrix[np.ix_(selected, selected)]
                iu, ju = np.triu_indices(sub.shape[0], k=1)
                if iu.size > 0:
                    diversity = float(sub[iu, ju].mean())
                else:
                    diversity = 0.0

            return strength, diversity

        # -------------------------
        # NSGA-II
        # -------------------------
        pop_size = int(getattr(self.args, "pae_pop_size", 40))
        num_generations = int(getattr(self.args, "pae_num_generations", 40))
        mutation_prob = float(getattr(self.args, "pae_mutation_prob", 0.05))
        crossover_prob = float(getattr(self.args, "pae_crossover_prob", 0.9))
        min_ens_size = int(getattr(self.args, "pae_min_ensemble_size", 1))
        raw_max_ens_size = int(getattr(self.args, "pae_max_ensemble_size", 0))
        max_ens_size = M if raw_max_ens_size <= 0 else raw_max_ens_size

        min_ens_size = max(1, min(min_ens_size, M))
        max_ens_size = max(min_ens_size, min(max_ens_size, M))

        def init_individual() -> np.ndarray:
            size = random.randint(min_ens_size, max_ens_size)
            mask = np.zeros(M, dtype=np.int8)
            chosen = random.sample(range(M), size)
            mask[chosen] = 1
            return mask

        def repair(mask: np.ndarray) -> np.ndarray:
            idx = np.where(mask > 0)[0]
            if idx.size < min_ens_size:
                available = [i for i in range(M) if mask[i] == 0]
                needed = min_ens_size - idx.size
                if available:
                    chosen = random.sample(available, min(needed, len(available)))
                    mask[chosen] = 1
            elif idx.size > max_ens_size:
                to_turn_off = random.sample(idx.tolist(), idx.size - max_ens_size)
                mask[to_turn_off] = 0
            return mask

        def dominates(f1: tuple[float, float], f2: tuple[float, float]) -> bool:
            return (f1[0] >= f2[0] and f1[1] >= f2[1]) and (f1[0] > f2[0] or f1[1] > f2[1])

        def non_dominated_sort(pop: list[np.ndarray], fitnesses: list[tuple[float, float]]) -> list[list[int]]:
            fronts: list[list[int]] = []
            S = [[] for _ in range(len(pop))]
            n = [0 for _ in range(len(pop))]

            for p in range(len(pop)):
                for q in range(len(pop)):
                    if dominates(fitnesses[p], fitnesses[q]):
                        S[p].append(q)
                    elif dominates(fitnesses[q], fitnesses[p]):
                        n[p] += 1
                if n[p] == 0:
                    if not fronts:
                        fronts.append([])
                    fronts[0].append(p)

            i = 0
            while i < len(fronts) and fronts[i]:
                next_front: list[int] = []
                for p in fronts[i]:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(front_indices: list[int], fitnesses: list[tuple[float, float]]) -> dict[int, float]:
            distance: dict[int, float] = {idx: 0.0 for idx in front_indices}
            if len(front_indices) == 0:
                return distance

            for obj in range(2):  # strength, diversity
                front_sorted = sorted(front_indices, key=lambda idx: fitnesses[idx][obj])
                f_min = fitnesses[front_sorted[0]][obj]
                f_max = fitnesses[front_sorted[-1]][obj]
                if f_max == f_min:
                    continue
                distance[front_sorted[0]] = float("inf")
                distance[front_sorted[-1]] = float("inf")
                for i in range(1, len(front_sorted) - 1):
                    prev_f = fitnesses[front_sorted[i - 1]][obj]
                    next_f = fitnesses[front_sorted[i + 1]][obj]
                    distance[front_sorted[i]] += (next_f - prev_f) / (f_max - f_min)

            return distance

        def tournament_select(indices: list[int], ranks: dict[int, int], crowd: dict[int, float]) -> int:
            i, j = random.sample(indices, 2)
            if ranks[i] < ranks[j]:
                return i
            if ranks[j] < ranks[i]:
                return j
            return i if crowd.get(i, 0.0) >= crowd.get(j, 0.0) else j

        def crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if random.random() > crossover_prob:
                return parent1.copy(), parent2.copy()
            child1 = parent1.copy()
            child2 = parent2.copy()
            for idx in range(M):
                if random.random() < 0.5:
                    child1[idx], child2[idx] = child2[idx], child1[idx]
            return child1, child2

        def mutate(mask: np.ndarray) -> np.ndarray:
            for idx in range(M):
                if random.random() < mutation_prob:
                    mask[idx] = 1 - mask[idx]
            return repair(mask)

        # Initialize population
        pop: list[np.ndarray] = [init_individual() for _ in range(pop_size)]

        # Seed with purely local ensemble (negative transfer protection)
        if local_indices:
            local_mask = np.zeros(M, dtype=np.int8)
            for idx in local_indices:
                local_mask[idx] = 1
            pop[0] = repair(local_mask)

        for _ in range(num_generations):
            fitnesses = [evaluate_individual(ind) for ind in pop]
            fronts = non_dominated_sort(pop, fitnesses)

            ranks: dict[int, int] = {}
            crowd: dict[int, float] = {}
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
                cd = crowding_distance(front, fitnesses)
                crowd.update(cd)

            all_indices = list(range(len(pop)))
            offspring: list[np.ndarray] = []
            while len(offspring) < pop_size:
                p1_idx = tournament_select(all_indices, ranks, crowd)
                p2_idx = tournament_select(all_indices, ranks, crowd)
                c1, c2 = crossover(pop[p1_idx], pop[p2_idx])
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring.append(c1)
                if len(offspring) < pop_size:
                    offspring.append(c2)

            pop = offspring

        # Final Pareto set and selection by ensemble val accuracy
        final_fitnesses = [evaluate_individual(ind) for ind in pop]
        final_fronts = non_dominated_sort(pop, final_fitnesses)

        if not final_fronts or not final_fronts[0]:
            print(f"[FedPAE][Client {self.role}] Empty Pareto front; fallback to local baseline.")
            selected_indices = local_indices if local_indices else list(range(M))
        else:
            pareto_indices = final_fronts[0]

            best_mask = None
            best_acc = -1.0
            best_size = None
            best_div = -1.0

            for idx in pareto_indices:
                mask = pop[idx]
                selected = np.where(mask > 0)[0].tolist()
                if not selected:
                    continue

                metrics = self._compute_ensemble_metrics(val.ds, val.y, selected)
                ens_acc = metrics["acc"]
                _, diversity = final_fitnesses[idx]
                size = len(selected)

                if (
                    ens_acc > best_acc
                    or (ens_acc == best_acc and (best_size is None or size < best_size))
                    or (ens_acc == best_acc and size == best_size and diversity > best_div)
                ):
                    best_acc = ens_acc
                    best_size = size
                    best_div = diversity
                    best_mask = mask.copy()

            if best_mask is None:
                selected_indices = local_indices if local_indices else list(range(M))
            else:
                selected_indices = np.where(best_mask > 0)[0].tolist()

        print(f"[FedPAE][Client {self.role}] Selected ensemble with {len(selected_indices)} models: {selected_indices}")

        # -------------------------
        # Evaluate on test + baselines
        # -------------------------
        FedPAE_metrics = self._compute_ensemble_metrics(test.ds, test.y, selected_indices)
        FedPAE_acc = FedPAE_metrics["acc"]
        FedPAE_bacc = FedPAE_metrics["bacc"]

        global_indices = list(range(M))
        local_metrics = self._compute_ensemble_metrics(test.ds, test.y, local_indices)
        global_metrics = self._compute_ensemble_metrics(test.ds, test.y, global_indices)

        self.perf_summary = {
            "local_acc": local_metrics["acc"],
            "local_bacc": local_metrics["bacc"],
            "global_acc": global_metrics["acc"],
            "global_bacc": global_metrics["bacc"],
            "FedPAE_acc": FedPAE_acc,
            "FedPAE_bacc": FedPAE_bacc,
            "acc_beats_local": int(FedPAE_acc > local_metrics["acc"]),
            "bacc_beats_local": int(FedPAE_bacc > local_metrics["bacc"]),
            "acc_beats_global": int(FedPAE_acc > global_metrics["acc"]),
            "bacc_beats_global": int(FedPAE_bacc > global_metrics["bacc"]),
            "acc_beats_baselines": int(FedPAE_acc > max(local_metrics["acc"], global_metrics["acc"])),
            "bacc_beats_baselines": int(FedPAE_bacc > max(local_metrics["bacc"], global_metrics["bacc"])),
        }

        print(
            f"[FedPAE][Client {self.role}] Test metrics -- "
            f"FedPAE_acc={FedPAE_acc:.4f}, FedPAE_bacc={FedPAE_bacc:.4f}, "
            f"local_acc={local_metrics['acc']:.4f}, global_acc={global_metrics['acc']:.4f}"
        )

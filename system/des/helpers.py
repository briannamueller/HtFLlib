
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import KFold
from utils.data_utils import read_client_data
from typing import Iterable

def available_devices() -> List[str]:
    if torch.cuda.is_available():
        try:
            cuda_devices = torch.cuda.device_count()
            # Touch each device to ensure it is actually usable; otherwise fall back to CPU.
            for idx in range(cuda_devices):
                _ = torch.cuda.get_device_properties(idx)
            if cuda_devices:
                return [f"cuda:{idx}" for idx in range(cuda_devices)]
        except Exception as exc:
            print(f"[FedDES][Server] CUDA unavailable for this run ({exc}); using CPU.")
    return ["cpu"]

def run_stage(clients: List[Any], stage: str, stage_inputs: Dict[str, Any] | Callable[[str], Dict[str, Any]] | None = None) -> None:
    """
    Run a named method across clients, optionally in parallel if GPUs are available.
    """
    devices = available_devices()
    n_workers = len(devices) if devices and devices[0] != "cpu" else 1
    print(f"[FedDES][run_stage] Stage={stage} devices={devices} n_workers={n_workers} n_clients={len(clients)}")

    def _move_input(value: Any, device: str, stage: str) -> Any:
        if isinstance(value, torch.nn.Module):
            cloned = copy.deepcopy(value)
            cloned.eval()
            if stage == "prepare_graph_data":
                return cloned.to("cpu")
            return cloned.to(device)
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: _move_input(v, device, stage) for k, v in value.items()}
        if isinstance(value, list):
            return [_move_input(v, device, stage) for v in value]
        if isinstance(value, tuple):
            return tuple(_move_input(v, device, stage) for v in value)
        return value

    def _prepare_stage_kwargs(stage_inputs: Any | None, device: str, stage: str) -> Dict[str, Any]:
        if stage_inputs is None:
            return {}
        inputs = stage_inputs(device) if callable(stage_inputs) else stage_inputs
        if not isinstance(inputs, dict):
            raise TypeError("stage_inputs must be a dict or callable returning a dict")
        return {k: _move_input(v, device, stage) for k, v in inputs.items()}

    def run(client, device):
        fn = getattr(client, stage)
        kwargs = _prepare_stage_kwargs(stage_inputs, device, stage)
        print(f"[FedDES][run_stage] Client={client.role} Stage={stage} Device={device} Inputs={list(kwargs.keys())}")
        return fn(device, **kwargs)

    if n_workers == 1:
        device = devices[0]
        for client in clients:
            run(client, device)
        return

    print(f"[FedDES][Server] Running {stage} with {n_workers} workers.")
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(run, client, devices[idx % len(devices)]): client
            for idx, client in enumerate(clients)
        }
        for fut, client in futures.items():
            try:
                fut.result()  # surface client exceptions
            except Exception as exc:
                import traceback
                print(f"[FedDES][run_stage][ERROR] Client={client.role} Stage={stage} Exception={exc}")
                print(traceback.format_exc())
                raise

def derive_config_ids(args: Any) -> Tuple[str, str, str]:
    data = args if isinstance(args, Mapping) else vars(args)

    def fp(prefix: str) -> str:
        cfg = {k: v for k, v in data.items() if isinstance(k, str) and k.startswith(prefix)}
        if not cfg:
            return "none"
        blob = json.dumps(cfg, sort_keys=True).encode("utf-8")
        return hashlib.md5(blob).hexdigest()[:8]

    return fp("base"), fp("graph"), fp("gnn")


def derive_pae_config_ids(args: Any) -> Tuple[str, str]:
    data = args if isinstance(args, Mapping) else vars(args)

    def fp(prefix: str) -> str:
        cfg = {k: v for k, v in data.items() if isinstance(k, str) and k.startswith(prefix)}
        if not cfg:
            return "none"
        blob = json.dumps(cfg, sort_keys=True).encode("utf-8")
        return hashlib.md5(blob).hexdigest()[:8]

    return fp("base"), fp("pae")

def load_base_clf(client: Any, model_str: str) -> torch.nn.Module:
    """
    Load a base classifier model for the given client and model string.
    """
    model_path = Path(client.base_dir) / f"{client.role}_{model_str}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Base classifier model not found at {model_path}")

    model = torch.load(model_path, map_location=client.device)
    # model.eval()
    return model



def init_base_meta_loaders(client: Any) -> Tuple[DataLoader, DataLoader]:
    """
    Split the client's train dataset into base/meta loaders (50/50).
    Caches loaders on the client to avoid rebuilding.
    """
    if getattr(client, "_base_train_loader", None) is not None and getattr(client, "_meta_train_loader", None) is not None:
        return client._base_train_loader, client._meta_train_loader

    # client.load_train_data only accepts batch_size; shuffling handled via random_split below.
    full_loader = client.load_train_data(batch_size=client.batch_size)
    dataset = full_loader.dataset
    n_total = len(dataset)
    g = torch.Generator().manual_seed(getattr(client.args, "base_split_seed", 0))
    n_base = n_total // 2
    n_meta = n_total - n_base
    base_ds, meta_ds = random_split(dataset, [n_base, n_meta], generator=g)

    seed = int(getattr(client.args, "seed", getattr(client.args, "base_split_seed", 0)))
    worker_init = lambda wid: np.random.seed(seed + wid)
    gen = torch.Generator().manual_seed(seed)

    client._base_train_loader = DataLoader(
        base_ds, batch_size=client.batch_size, shuffle=True, drop_last=True,
        worker_init_fn=worker_init, generator=gen,
    )
    client._meta_train_loader = DataLoader(
        meta_ds, batch_size=client.batch_size, shuffle=False, drop_last=False,
        worker_init_fn=worker_init, generator=gen,
    )
    return client._base_train_loader, client._meta_train_loader


def get_kfold_loaders(
    client: Any,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[DataLoader, DataLoader, np.ndarray]]:
    """
    Build deterministic K-fold (train_loader, val_loader, val_idx) tuples
    over the full local training set. Val loaders do NOT shuffle.
    """
    full_train_loader = client.load_train_data(batch_size=client.batch_size)
    dataset = full_train_loader.dataset

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(dataset))

    loaders = []
    for train_idx, val_idx in kf.split(indices):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        worker_init = lambda wid: np.random.seed(seed + wid)
        gen = torch.Generator().manual_seed(seed)

        train_loader = DataLoader(
            train_subset,
            batch_size=client.batch_size,
            shuffle=True,
            drop_last=True,
            worker_init_fn=worker_init,
            generator=gen,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=client.batch_size,
            shuffle=False,
            drop_last=False,
            worker_init_fn=worker_init,
            generator=gen,
        )
        loaders.append((train_loader, val_loader, val_idx))

    return loaders


def build_dataset_partition_id(
    partition_type: str,
    alpha: float,
    C: int,
    min_size: int,
    train_ratio: float,
    seed: int,
    num_clients: int,
) -> str:
    def _fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    components = []
    if partition_type == "pat":
        components.append(f"C={_fmt(C)}")
    elif partition_type == "dir":
        components.append(f"alpha={_fmt(alpha)}")
    elif partition_type == "exdir":
        ex_components = [
            f"alpha={_fmt(alpha)}",
            f"C={_fmt(C)}",
        ]
        if min_size is not None:
            ex_components.append(f"min_size={_fmt(min_size)}")
        components.extend(ex_components)
    detail = ",".join(components)
    suffix = (
        f"_nc[{num_clients}]"
        f"_tr[{_fmt(train_ratio)}]"
        f"_s[{seed}]"
    )
    return f"{partition_type}[{detail}]{suffix}"


def build_eicu_partition_id(
    task: str,
    min_size: int,
    seed: int,
    train_ratio: float,
    num_clients: int,
) -> str:
    def _fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    detail = f"task={task},min_size={min_size}"
    suffix = (
        f"_nc[{num_clients}]"
        f"_tr[{_fmt(train_ratio)}]"
        f"_s[{seed}]"
    )
    return f"eicu[{detail}]{suffix}"

def get_performance_baselines(client: Any, test_bundle: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """
    Compute baseline global/local ensemble metrics (soft and hard) from cached test preds/y.
    Saves a nested dict keyed by "soft" and "hard".
    """
    test_preds = test_bundle["preds"]  # [N, M]
    y_test = test_bundle["y"]          # [N]
    ds = test_bundle["ds"]             # [N, M*C] flattened probs

    C = client.args.num_classes
    probs = ds.view(test_preds.size(0), test_preds.size(1), C).float()  # [N, M, C]

    def combine_soft(prob_tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return prob_tensor.mean(dim=1)
        weights = mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (prob_tensor * weights.unsqueeze(-1)).sum(dim=1) / denom

    def combine_hard(pred_tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        one_hot = F.one_hot(pred_tensor, num_classes=C).float()  # [N, M, C]
        if mask is not None:
            one_hot = one_hot * mask.unsqueeze(-1)
        return one_hot.sum(dim=1).argmax(dim=1)

    key_to_idx = {k: i for i, k in enumerate(client.global_clf_keys)}
    local_indices = [key_to_idx[k] for k in client.local_clf_keys if k in key_to_idx]
    local_mask = torch.zeros_like(test_preds, dtype=torch.float)
    local_mask[:, local_indices] = 1.0

    metrics: Dict[str, Dict[str, float]] = {}

    # Soft
    global_probs = combine_soft(probs, None)
    local_probs = combine_soft(probs, local_mask if local_indices else None)
    global_preds = global_probs.argmax(dim=1)
    local_preds = local_probs.argmax(dim=1)
    metrics["soft"] = {
        "global_acc": (global_preds == y_test).float().mean().item(),
        "global_bacc": client.balanced_accuracy(global_preds, y_test),
        "local_acc": (local_preds == y_test).float().mean().item(),
        "local_bacc": client.balanced_accuracy(local_preds, y_test),
    }

    # Hard
    global_preds_h = combine_hard(test_preds, None)
    local_preds_h = combine_hard(test_preds, local_mask)
    metrics["hard"] = {
        "global_acc": (global_preds_h == y_test).float().mean().item(),
        "global_bacc": client.balanced_accuracy(global_preds_h, y_test),
        "local_acc": (local_preds_h == y_test).float().mean().item(),
        "local_bacc": client.balanced_accuracy(local_preds_h, y_test),
    }

    # Voting baseline treated same as hard voting (unweighted subset)
    metrics["voting"] = metrics["hard"].copy()

    individual_classifier_perf: List[Dict[str, Any]] = []
    for idx, clf_key in enumerate(client.global_clf_keys):
        clf_preds = test_preds[:, idx]
        acc = (clf_preds == y_test).float().mean().item()
        bacc = client.balanced_accuracy(clf_preds, y_test)
        individual_classifier_perf.append({
            "classifier": f"{clf_key[0]}:{clf_key[1]}",
            "home_client": clf_key[0],
            "model": clf_key[1],
            "acc": acc,
            "bacc": bacc,
        })
    metrics["individual_classifier_perf"] = individual_classifier_perf

    out_path = Path(client.base_dir) / f"{client.role}_performance_baselines.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

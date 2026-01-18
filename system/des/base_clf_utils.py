import csv
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from flcore.trainmodel.models import BaseHeadSplit
from utils.data_utils import read_client_data
from des.wandb_utils import _safe_wandb_init

from probmetrics.metrics import Metrics
metrics = Metrics.from_names(['refinement_logloss_ts-mix_all'])

try:
    import wandb
except Exception:
    wandb = None


def process_batch(batch, device: torch.device):
    """
    Normalize a batch to (x, y) tensors on the target device.
    Handles cases where x/y are wrapped in list/tuple.
    """
    x, y = batch
    x = x[0] if isinstance(x, (list, tuple)) else x
    y = y[0] if isinstance(y, (list, tuple)) else y
    return x.to(device), y.to(device)


# -------------------------------
# Core epoch helpers
# -------------------------------

def train_one_epoch(model, loader, device, optimizer, loss_fn, scheduler=None):
    """Single training epoch: cross-entropy loss + accuracy."""
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for batch in loader:
        x, y = process_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            total_correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    return {"loss": avg_loss, "acc": acc}


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, return_logits=False):
    """
    Evaluate loss + accuracy + balanced accuracy.

    If return_logits=True, also return (logits_cat, labels_cat) for temp scaling.
    """
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_logits = [] if return_logits else None
    all_labels = [] if return_logits else None
    y_true, y_pred = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * y.size(0)
        total_correct += (preds == y).sum().item()
        total += y.size(0)

        y_true.append(y.detach().cpu())
        y_pred.append(preds.detach().cpu())

        if return_logits:
            all_logits.append(logits.detach())
            all_labels.append(y.detach())

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)

    if y_true:
        y_true_cat = torch.cat(y_true).numpy()
        y_pred_cat = torch.cat(y_pred).numpy()
        if len(set(y_true_cat)) > 1:
            bal_acc = balanced_accuracy_score(y_true_cat, y_pred_cat)
        else:
            bal_acc = 0.0
    else:
        bal_acc = 0.0

    stats = {"loss": avg_loss, "acc": acc, "bal_acc": bal_acc}

    if return_logits:
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        return stats, logits_cat, labels_cat

    return stats


def fit_calibrator(logits, labels):
    """
    Fit a temperature T by minimizing *unweighted* CE on val logits.
    Uses LBFGS on log_T. Returns a scalar float temperature.
    """
    device = logits.device
    x = logits.detach().float().to(device)
    y = labels.detach().long().to(device)

    log_T = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T)
        loss = F.cross_entropy(x / (T + 1e-6), y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except RuntimeError:
        print("Temp scaling: LBFGS failed, using T=1")
        return 1.0

    if not torch.isfinite(log_T).all():
        print("Temp scaling: non-finite log_T, using T=1")
        return 1.0

    T = torch.exp(log_T).clamp(0.05, 20.0).item()
    return float(T)


# -------------------------------
# Loss builder
# -------------------------------

class BalancedBCEWithLogits(nn.Module):
    """Binary cross-entropy with logits using a positive-class weight."""
    def __init__(self, pos_weight: float):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() > 1 and logits.size(1) == 2:
            logits = logits[:, 1]           # keep only the logit for class 1
        targets = targets.float()
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.to(logits.device),
        )


def build_loss_fn(client, weighted: bool, device: torch.device):
    """Binary: BCEWithLogits (optional pos_weight); Multi-class: CE (optional weights)."""
    num_classes = client.args.num_classes

    # Binary case
    if num_classes == 2:
        if not weighted:
            return BalancedBCEWithLogits(pos_weight=1.0)

        counts = torch.zeros(num_classes, dtype=torch.float)
        train_ds = read_client_data(
            client.args.dataset,
            client.id,
            is_train=True,
            few_shot=client.args.few_shot,
        )
        for _, y in train_ds:
            lbl = int(y.item()) if torch.is_tensor(y) else int(y)
            if 0 <= lbl < counts.numel():
                counts[lbl] += 1.0
        pos_weight = (counts[0] / counts[1].clamp(min=1.0)).clamp(max=50.0)
        # return nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(client.device))
        return BalancedBCEWithLogits(pos_weight=pos_weight.to(device))

    # Multi-class case
    if not weighted:
        return nn.CrossEntropyLoss()

    counts = torch.zeros(num_classes, dtype=torch.float)
    train_ds = read_client_data(
        client.args.dataset,
        client.id,
        is_train=True,
        few_shot=client.args.few_shot,
    )
    for _, y in train_ds:
        lbl = int(y.item()) if torch.is_tensor(y) else int(y)
        if 0 <= lbl < counts.numel():
            counts[lbl] += 1.0

    if counts.sum() == 0:
        return nn.CrossEntropyLoss()

    safe = counts.clamp(min=1.0)
    weights = (safe.sum() / safe.numel()) / safe
    return nn.CrossEntropyLoss(weight=weights.to(device))


# -------------------------------
# Main training function
# -------------------------------

def fit_clf(
    self,
    model_id,
    train_loader,
    val_loader,
    device,
    max_epochs=100,
    patience=10,
    min_delta=0.0,
    es_metric="val_loss",   # "val_loss" or "val_temp_scaled_loss"
    lr=1e-3,
    warmup_epochs=10,
):
    model = BaseHeadSplit(self.args, model_id).to(device)
    loss_fn = build_loss_fn(self, getattr(self.args, "base_weighted_loss", True), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warmup_scheduler = None
    if warmup_epochs > 0 and len(train_loader) > 0:
        warmup_steps = warmup_epochs * len(train_loader)
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, float(step + 1) / warmup_steps),
        )

    best_state = None
    best_metric = -float("inf")
    best_epoch, stale_epochs = 0, 0

    # For logging
    history = []

    for epoch in range(1, max_epochs + 1):
        # ---- train ----
        train_stats = train_one_epoch(model, train_loader, device, optimizer, loss_fn, scheduler=warmup_scheduler)

        # ---- validate (with or without temp scaling) ----
        if es_metric == "val_temp_scaled_loss":
            val_stats, val_logits, val_labels = evaluate(
                model, val_loader, device, loss_fn, return_logits=True
            )
            ts_results = metrics.compute_all_from_labels_logits(val_labels, val_logits)
            val_stats["ts_ref_loss"] = ts_results['refinement_logloss_ts-mix_all'].item()
            # T = fit_calibrator(val_logits, val_labels)
            # calib_logits = val_logits / T
            # ts_ref_loss = F.cross_entropy(calib_logits, val_labels).item()
            # val_stats["ts_ref_loss"] = ts_ref_loss
        else:
            val_stats = evaluate(model, val_loader, device, loss_fn)

        # scalar ES metric (larger = better)
        if es_metric == "val_temp_scaled_loss":
            metric_val = -val_stats["ts_ref_loss"]  # lower TS loss is better
        else:
            metric_val = -val_stats["loss"]         # lower val_loss is better

        # ---- log row for this epoch ----
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "val_bacc": val_stats["bal_acc"],
        }
        if es_metric == "val_temp_scaled_loss" and "ts_ref_loss" in val_stats:
            row["val_ts_ref_loss"] = val_stats["ts_ref_loss"]
        row["es_metric"] = val_stats["ts_ref_loss"] if es_metric == "val_temp_scaled_loss" and "ts_ref_loss" in val_stats else val_stats["loss"]
        history.append(row)

        # ---- checkpoint selection (always by metric_val) ----
        improved = metric_val > best_metric + float(min_delta)
        if improved:
            best_metric = metric_val
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        # ---- stopping rule ----
        if stale_epochs >= patience:
            print(f"[BaseClf][EarlyStop] epoch={epoch}, best_metric={best_metric:.6f}")
            break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------------------------
    # Save training logs to CSV
    # ---------------------------
    logs_dir = self.base_outputs_dir / f"{self.role}_training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"model_{model_id}.csv"

    fieldnames: List[str] = []
    if history:
        # Collect all keys used across rows (handles optional ts_ref_loss)
        fieldnames = sorted({k for row in history for k in row.keys()})
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)

    print(f"[BaseClf] Saved training logs to {log_path}")

    def _should_log_wandb() -> bool:
        if wandb is None:
            return False
        flag = os.getenv("WANDB_BASE_LOG", "")
        return flag == "" or flag.lower() not in {"0", "false"}

    if history and fieldnames and _should_log_wandb():
        run = None
        try:
            run_config = {
                "batch_size": getattr(self.args, "batch_size", None),
                "feature_dim": getattr(self.args, "feature_dim", None),
            }
            for key, value in vars(self.args).items():
                if key.startswith("base_"):
                    run_config[key] = value

            run = _safe_wandb_init(
                project=os.getenv("WANDB_PROJECT"),
                entity=os.getenv("WANDB_ENTITY"),
                name=f"{self.role}_model_{model_id}",
                reinit=True,
                config=run_config,
                job_type="base_clf_training",
            )
        except Exception:
            run = None
        if run is None:
            return best_epoch, best_metric, model

        table_data = [[row.get(col) for col in fieldnames] for row in history]
        history_table = wandb.Table(columns=fieldnames, data=table_data)
        train_plot = wandb.plot.line(
            history_table,
            "epoch",
            "train_loss",
            title=f"Base model {model_id} train loss",
        )
        val_plot = wandb.plot.line(
            history_table,
            "epoch",
            "val_loss",
            title=f"Base model {model_id} val loss",
        )
        log_payload = {
            f"base_model_{model_id}_history": history_table,
            f"base_model_{model_id}_train_loss": train_plot,
            f"base_model_{model_id}_val_loss": val_plot,
        }
        if es_metric != "val_loss" and "es_metric" in fieldnames:
            es_plot = wandb.plot.line(
                history_table,
                "epoch",
                "es_metric",
                title=f"Base model {model_id} {es_metric}",
            )
            log_payload[f"base_model_{model_id}_es_metric"] = es_plot

        try:
            run.log(log_payload)
        except Exception:
            pass
        finally:
            if hasattr(run, "summary") and run.summary is not None:
                for key in log_payload:
                    try:
                        del run.summary[key]
                    except Exception:
                        pass
                try:
                    run.summary.update({
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        "early_stopping_metric": "val_ts_ref_loss" if es_metric == "val_temp_scaled_loss" else "val_loss",
                    })
                except Exception:
                    pass
            try:
                wandb.finish()
            except Exception:
                pass

    return best_epoch, best_metric, model

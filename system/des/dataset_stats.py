from __future__ import annotations

from pathlib import Path
from typing import Dict
import json


def load_client_label_counts(dataset_name: str) -> Dict[str, Dict[int, int]]:
    dataset_root = Path(__file__).resolve().parents[2] / "dataset"
    config_path = dataset_root / dataset_name / "config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    counts_map: Dict[str, Dict[int, int]] = {}
    entries = data.get("client_label_counts") or data.get("Size of samples for labels in clients", [])
    for idx, entry in enumerate(entries):
        cls_counts: Dict[int, int] = {}
        for cls, cnt in entry:
            cls_counts[int(cls)] = int(cnt)
        counts_map[f"Client_{idx}"] = cls_counts
    return counts_map

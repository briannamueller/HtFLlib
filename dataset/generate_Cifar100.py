#!/usr/bin/env python
"""Create CIFAR-100 client splits and store each configuration in a descriptive directory."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from utils.dataset_utils import (
    DEFAULT_ALPHA,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_MIN_SIZE,
    save_file,
    separate_data,
    split_data,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build CIFAR-100 client splits with descriptive folders.")
    parser.add_argument("--dataset-name", type=str, default="Cifar100")
    parser.add_argument("--output-root", type=str, default="dataset")
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--partition-id", type=str, default="")
    parser.add_argument(
        "--partition",
        type=str,
        choices=["pat", "dir", "exdir"],
        default="dir",
        help="Partition strategy for client splits.",
    )
    parser.add_argument(
        "--niid",
        dest="niid",
        action="store_true",
        help="Generate non-IID splits (default).",
    )
    parser.add_argument(
        "--iid",
        dest="niid",
        action="store_false",
        help="Generate IID splits (overrides --partition to 'pat').",
    )
    parser.set_defaults(niid=True)
    parser.add_argument(
        "--balance",
        dest="balance",
        action="store_true",
        help="Try to balance samples per client in 'pat' splits.",
    )
    parser.add_argument(
        "--no-balance",
        dest="balance",
        action="store_false",
        help="Allow imbalance across clients.",
    )
    parser.set_defaults(balance=False)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--C", type=int, default=2,
                        help="Number of classes assigned to each client (pat/exdir).")
    parser.add_argument("--min-require-size-per-label", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    dataset_root = output_root / args.dataset_name
    raw_dir = dataset_root / "rawdata"
    dataset_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    random.seed(args.seed)
    np.random.seed(args.seed)

    trainset = torchvision.datasets.CIFAR100(
        root=str(raw_dir), train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root=str(raw_dir), train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False
    )

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    partition = args.partition
    if not args.niid:
        partition = "pat"

    min_require_per_label = args.min_require_size_per_label
    if partition == "exdir" and min_require_per_label is None:
        min_require_per_label = max(
            args.C * args.num_clients // num_classes // 2, 1
        )

    split_config = {
        "dataset_name": args.dataset_name,
        "num_clients_requested": args.num_clients,
        "num_clients_actual": args.num_clients,
        "niid": args.niid,
        "balance": args.balance,
        "partition": partition,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "train_ratio": args.train_ratio,
        "C": args.C,
        "min_require_size_per_label": min_require_per_label,
        "min_size": args.min_size,
        "seed": args.seed,
        "num_classes": num_classes,
    }

    sys.path.insert(0, str(repo_root))
    from system.des.helpers import build_dataset_partition_id

    partition_id = args.partition_id or build_dataset_partition_id(
        partition,
        args.alpha,
        args.C,
        args.min_size,
        args.train_ratio,
        args.seed,
        args.num_clients,
    )
    split_config["partition_id"] = partition_id
    dataset_dir = dataset_root / partition_id
    train_path = dataset_dir / "train"
    test_path = dataset_dir / "test"
    config_path = dataset_dir / "config.json"

    if config_path.exists():
        print(
            f"\nDataset already generated at {dataset_dir} (partition_id={partition_id}).\n"
            f"Use --dataset {args.dataset_name} --data-partition {partition_id} when running the experiment."
        )
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        args.num_clients,
        num_classes,
        niid=args.niid,
        balance=args.balance,
        partition=partition,
        class_per_client=args.C,
        alpha=args.alpha,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        labels_per_client=args.C,
        min_require_size_per_label=min_require_per_label,
        min_size=args.min_size,
    )
    train_data, test_data = split_data(X, y, train_ratio=args.train_ratio)

    save_file(
        str(config_path),
        str(train_path),
        str(test_path),
        train_data,
        test_data,
        args.num_clients,
        num_classes,
        statistic,
        split_config=split_config,
    )

    try:
        relative_path = dataset_dir.relative_to(repo_root)
    except ValueError:
        relative_path = dataset_dir

    print(f"\nGenerated {args.dataset_name} split with partition_id {partition_id} at {dataset_dir}.")
    print(f"Relative path: {relative_path}")
    print(f"Point your experiment at --dataset {args.dataset_name} --data-partition {partition_id}.")


if __name__ == "__main__":
    main()

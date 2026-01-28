import os
import ujson
import numpy as np
import gc
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_BATCH_SIZE = 10
DEFAULT_TRAIN_RATIO = 0.75  # merge original training set and test set, then split it manually.
DEFAULT_ALPHA = 10
DEFAULT_LABELS_PER_CLIENT = 10
DEFAULT_MIN_SIZE = 10

def separate_data(
    data,
    num_clients,
    num_classes,
    niid=False,
    balance=False,
    partition=None,
    class_per_client=None,
    *,
    alpha=DEFAULT_ALPHA,
    batch_size=DEFAULT_BATCH_SIZE,
    train_ratio=DEFAULT_TRAIN_RATIO,
    labels_per_client=DEFAULT_LABELS_PER_CLIENT,
    min_require_size_per_label=None,
    min_size=DEFAULT_MIN_SIZE,
):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    train_gap = max(1.0 - train_ratio, 1e-6)
    least_samples = int(
        min(batch_size / train_gap, len(dataset_label) / num_clients / 2)
    )

    dataidx_map = {}

    if not niid:
        partition = "pat"
        class_per_client = num_classes

    if partition == "pat":
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = [
                client for client in range(num_clients) if class_num_per_client[client] > 0
            ]
            if len(selected_clients) == 0:
                break
            selected_clients = selected_clients[
                : int(np.ceil((num_clients / num_classes) * class_per_client))
            ]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(
                    max(num_per / 10, least_samples / num_classes),
                    num_per,
                    num_selected_clients - 1,
                ).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map:
                    dataidx_map[client] = idx_for_each_class[i][idx : idx + num_sample]
                else:
                    dataidx_map[client] = np.append(
                        dataidx_map[client],
                        idx_for_each_class[i][idx : idx + num_sample],
                        axis=0,
                    )
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(
                    f"Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time."
                )

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / num_clients)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]

    elif partition == "exdir":
        r"""This strategy comes from https://arxiv.org/abs/2311.03154
        See details in https://github.com/TsingZ0/PFLlib/issues/139

        This version in PFLlib is slightly different from the original version 
        Some changes are as follows:
        n_nets -> num_clients, n_class -> num_classes
        """
        debug = os.environ.get("EXDIR_DEBUG", "0") == "1"
        debug_every = int(os.environ.get("EXDIR_DEBUG_EVERY", "100"))
        C = labels_per_client
        if C < 1:
            raise ValueError("labels_per_client must be >= 1")
        min_size_per_label = 0
        actual_min_require = (
            min_require_size_per_label
            if min_require_size_per_label is not None
            else max(C * num_clients // num_classes // 2, 1)
        )
        if actual_min_require < 1:
            raise ValueError
        clientidx_map = {}
        assign_attempts = 0
        while min_size_per_label < actual_min_require:
            assign_attempts += 1
            for k in range(num_classes):
                clientidx_map[k] = []
            for i in range(num_clients):
                labelidx = np.random.choice(range(num_classes), C, replace=False)
                for k in labelidx:
                    clientidx_map[k].append(i)
            min_size_per_label = min([len(clientidx_map[k]) for k in range(num_classes)])
            if debug and assign_attempts % debug_every == 0:
                per_label = [len(clientidx_map[k]) for k in range(num_classes)]
                print(
                    f"[exdir][assign] attempt={assign_attempts} "
                    f"min_size_per_label={min_size_per_label} "
                    f"actual_min_require={actual_min_require} "
                    f"min={min(per_label)} max={max(per_label)}"
                )

        dataidx_map = {}
        y_train = dataset_label
        min_size_threshold = min_size
        K = num_classes
        N = len(y_train)
        print("\n*****clientidx_map*****")
        print(clientidx_map)
        print("\n*****Number of clients per label*****")
        print([len(clientidx_map[i]) for i in range(len(clientidx_map))])

        current_min_size = 0
        alloc_attempts = 0
        while current_min_size < min_size_threshold:
            alloc_attempts += 1
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array(
                    [
                        p
                        * (
                            len(idx_j) < N / num_clients
                            and j in clientidx_map[k]
                        )
                        for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                if proportions[-1] != len(idx_k):
                    for w in range(clientidx_map[k][-1], num_clients - 1):
                        proportions[w] = len(idx_k)
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                current_min_size = min([len(idx_j) for idx_j in idx_batch])
            if debug and alloc_attempts % debug_every == 0:
                sizes = [len(idx_j) for idx_j in idx_batch]
                print(
                    f"[exdir][alloc] attempt={alloc_attempts} "
                    f"current_min_size={current_min_size} "
                    f"min_size_threshold={min_size_threshold} "
                    f"min={min(sizes)} max={max(sizes)}"
                )

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            dataidx_map[j] = idx_batch[j]

    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
            

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y, train_ratio=DEFAULT_TRAIN_RATIO):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data

def format_client_counts(statistic):
    formatted = []
    for client in statistic:
        formatted.append([[int(cls), int(cnt)] for cls, cnt in client])
    return formatted


def plot_client_distribution(statistic, plot_path):
    if not statistic:
        return
    xs = []
    ys = []
    sizes = []
    for client_idx, client in enumerate(statistic):
        for cls, cnt in client:
            xs.append(client_idx)
            ys.append(cls)
            sizes.append(max(10, cnt / 5))
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys, s=sizes, c="red", alpha=0.65, edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Client IDs")
    ax.set_ylabel("Class IDs")
    ax.set_title("Client label distribution")
    ax.set_xlim(-0.5, max(xs) + 0.5)
    ax.set_ylim(-0.5, max(ys) + 0.5)
    ax.set_xticks(range(max(xs) + 1))
    ax.set_yticks(range(max(ys) + 1))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

def build_class_prevalence(counts, num_classes):
    prevalence = {}
    for client_idx, client_counts in enumerate(counts):
        totals = [0] * num_classes
        total = 0
        for cls, cnt in client_counts:
            if 0 <= cls < num_classes:
                totals[cls] = cnt
                total += cnt
        if total > 0:
            prevalence[client_idx] = [cnt / total for cnt in totals]
        else:
            prevalence[client_idx] = [0.0] * num_classes
    return prevalence

def save_file(
    config_path,
    train_path,
    test_path,
    train_data,
    test_data,
    num_clients,
    num_classes,
    statistic,
    *,
    split_config=None,
):
    config_path = Path(config_path)
    train_dir = Path(train_path)
    test_dir = Path(test_path)
    config = dict(split_config or {})
    if 'num_clients' not in config:
        config['num_clients'] = num_clients
    if 'num_classes' not in config:
        config['num_classes'] = num_classes
    counts = format_client_counts(statistic)
    if 'client_label_counts' not in config:
        config['client_label_counts'] = counts
    if 'class_prevalence' not in config:
        config['class_prevalence'] = build_class_prevalence(counts, num_classes)

    # gc.collect()
    print("Saving to disk.\n")

    train_dir = Path(train_path)
    test_dir = Path(test_path)
    for idx, train_dict in enumerate(train_data):
        target = train_dir / f"{idx}.npz"
        with open(target, 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        target = test_dir / f"{idx}.npz"
        with open(target, 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    placeholder = "__CLIENT_COUNT_PLACEHOLDER__"
    counts_data = config.get('client_label_counts', [])
    config['client_label_counts'] = placeholder
    text = json.dumps(config, indent=2)
    if counts_data:
        lines = [f"    {json.dumps(client)}" for client in counts_data]
        counts_block = "[\n" + ",\n".join(lines) + "\n  ]"
    else:
        counts_block = "[]"
    text = text.replace(f'"client_label_counts": "{placeholder}"', f'"client_label_counts": {counts_block}')
    with open(config_path, 'w') as f:
        f.write(text + "\n")

    plot_client_distribution(statistic, config_path.parent / "distribution.png")

    print("Finish generating dataset.\n")


class ImageDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing file names
            image_folder (str): Path to the folder containing the images
            transform (callable, optional): Optional transform to be applied to the image
        """
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the file name from the DataFrame
        img_name = self.dataframe.iloc[idx]['file_name']
        img_label = self.dataframe.iloc[idx]['class']
        img_path = os.path.join(self.image_folder, img_name)
        
        # Load the image using PIL
        image = Image.open(img_path).convert('RGB')  # Ensure RGB if not grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_label

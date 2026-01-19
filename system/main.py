#!/usr/bin/env python
import torch
import argparse
import os
import time
import warnings
import numpy as np
import logging

from flcore.servers.serverlocal import Local
from flcore.servers.serverproto import FedProto
from flcore.servers.servergen import FedGen
from flcore.servers.serverfd import FD
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.serverfml import FML
from flcore.servers.serverkd import FedKD
from flcore.servers.servergh import FedGH
from flcore.servers.servertgp import FedTGP
# from flcore.servers.serverktl_stylegan_xl import FedKTL as FedKTL_stylegan_xl
# from flcore.servers.serverktl_stylegan_3 import FedKTL as FedKTL_stylegan_3
from flcore.servers.serverktl_stable_diffusion import FedKTL as FedKTL_stable_diffusion
from flcore.servers.servermrl import FedMRL
from flcore.servers.serverdes import FedDES
from flcore.servers.serverpae import FedPAE
from typing import Any, Callable, Dict, List, Mapping, Tuple

from utils.result_utils import average_data
from utils.mem_utils import MemReporter
# from system.des.helpers import build_dataset_partition_id, build_eicu_partition_id




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


def extract_eicu_task_from_dataset(dataset_name: str) -> str:
    marker = "task=["
    idx = dataset_name.find(marker)
    if idx == -1:
        return dataset_name
    idx += len(marker)
    end = dataset_name.find("]", idx)
    if end == -1:
        end = len(dataset_name)
    return dataset_name[idx:end]

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")

def run(args):

    time_list = []
    reporter = MemReporter()

    print(f"[FedDES][Main] Starting run with phase={getattr(args, 'phase', None)}, algorithm={args.algorithm}, dataset={args.dataset}")

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.models
        if args.model_family == "HtFE-img-2":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)', # for 32x32 img
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtFE-img-3":
            args.models = [
                'resnet10(num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtFE-img-4":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)', # for 32x32 img
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)'
            ]

        elif args.model_family == "HtFE-img-5":
            args.models = [
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtFE-img-8":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)', # for 32x32 img
                # 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816)', # for 64x64 img
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
            ]

        elif args.model_family == "HtFE-img-9":
            args.models = [
                'resnet4(num_classes=args.num_classes)', 
                'resnet6(num_classes=args.num_classes)', 
                'resnet8(num_classes=args.num_classes)', 
                'resnet10(num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtFE-img-8-HtC-img-4":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)', # for 32x32 img
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
            ]
            args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)'# for 32x32 img
            args.heads = [
                'Head(hidden_dims=[512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 256], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 128], num_classes=args.num_classes)', 
            ]

        elif args.model_family == "Res34-HtC-img-4":
            args.models = [
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
            ]
            args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)'# for 32x32 img
            args.heads = [
                'Head(hidden_dims=[512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 256], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 128], num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtM-img-10":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)', # for 32x32 img
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.vit_b_16(image_size=32, num_classes=args.num_classes)', 
                'torchvision.models.vit_b_32(image_size=32, num_classes=args.num_classes)'
            ]

        elif args.model_family == "HtFE-txt-2":
            args.models = [
                'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)'
            ]

        elif args.model_family == "HtFE-txt-4":
            args.models = [
                'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, output_size=args.num_classes, num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, embedding_length=args.feature_dim)'
            ]

        elif args.model_family == "HtFE-txt-5-1":
            args.models = [
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
            ]

        elif args.model_family == "HtFE-txt-5-2":
            args.models = [
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            ]

        elif args.model_family == "HtFE-txt-5-3":
            args.models = [
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
            ]
        
        elif args.model_family == "HtFE-txt-6":
            args.models = [
                'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)', 
                'LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)', 
                'BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, output_size=args.num_classes, num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, embedding_length=args.feature_dim)', 
                'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
                'TextLogisticRegression(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                'GRUNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)'
            ]

        elif args.model_family == "MLPs":
            args.models = [
                'AmazonMLP(feature_dim=[])', 
                'AmazonMLP(feature_dim=[500])', 
                'AmazonMLP(feature_dim=[1000, 500])', 
                'AmazonMLP(feature_dim=[1000, 500, 200])', 
            ]

        elif args.model_family == "MLP_1layer":
            args.models = [
                'AmazonMLP(feature_dim=[200])', 
                'AmazonMLP(feature_dim=[500])', 
            ]

        elif args.model_family == "MLP_layers":
            args.models = [
                'AmazonMLP(feature_dim=[500])', 
                'AmazonMLP(feature_dim=[1000, 500])', 
                'AmazonMLP(feature_dim=[1000, 500, 200])', 
            ]

        elif args.model_family == "HtFE-sen-2":
            args.models = [
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
            ]

        elif args.model_family == "HtFE-sen-3":
            args.models = [
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
            ]

        elif args.model_family == "HtFE-sen-5":
            args.models = [
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
                'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=1)',
                'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=1)',
            ]

        elif args.model_family == "HtFE-sen-8":
            args.models = [
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=1)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=2)',
                'HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, stride=3)',
                'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=1)',
                'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=2)',
                'HARCNN1(9, dim_hidden=832, num_classes=args.num_classes, stride=3)',
                'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=1)',
                'HARCNN3(9, dim_hidden=3328, num_classes=args.num_classes, stride=2)',
            ]

        elif args.model_family == "eICU":
            args.models = [
                'CNN_V3(in_channels=args.eicu_in_channels, L_in=args.eicu_seq_len, output_size=args.num_classes, depth=1, n_filters=32, n_neurons=32, dropout=0.3, activation="relu")',
                'TCN(in_channels=args.eicu_in_channels, n_classes=args.num_classes, n_filters=32, kernel_size=3, num_levels=2, dropout=0.3)',
                'ResNet1D(in_channels=args.eicu_in_channels, base_filters=32, n_blocks=3, downsample_gap=3, increasefilter_gap=4, n_classes=args.num_classes, use_bn=True, use_do=True, dropout=0.3)',
            ]

        else:
            raise NotImplementedError
            
        for model in args.models:
            print(model)

        if hasattr(args, 'global_model'):
            print('global_model:', args.global_model)

        if hasattr(args, 'heads'):
            for head in args.heads:
                print('head:', head)

        # select algorithm
        if args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedProto":
            server = FedProto(args, i)

        elif args.algorithm == "FedGen":
            server = FedGen(args, i)

        elif args.algorithm == "FD":
            server = FD(args, i)

        elif args.algorithm == "LG-FedAvg":
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            server = FedKD(args, i)

        elif args.algorithm == "FedGH":
            server = FedGH(args, i)

        elif args.algorithm == "FedTGP":
            server = FedTGP(args, i)
            
        # elif args.algorithm == "FedKTL-stylegan-xl":
        #     server = FedKTL_stylegan_xl(args, i)

        # elif args.algorithm == "FedKTL-stylegan-3":
        #     server = FedKTL_stylegan_3(args, i)

        elif args.algorithm == "FedKTL-stable-diffusion":
            server = FedKTL_stable_diffusion(args, i)

        elif args.algorithm == "FedMRL":
            server = FedMRL(args, i)

        elif args.algorithm == "FedDES":
            print("[FedDES][Main] Initializing FedDES server...")
            server = FedDES(args, i)

        elif args.algorithm == "FedPAE":
            print("[FedPAE][Main] Initializing FedPAE server...")
            server = FedPAE(args, i)
            
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    # average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    # reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default=None)
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('--partition-type', type=str, default="exdir",
                        choices=["pat", "dir", "exdir"],
                        help="Data partition strategy for default dataset splits.")
    parser.add_argument('--partition-alpha', type=float, default=1.0,
                        help="Dirichlet alpha for default data splits.")
    parser.add_argument('--partition-C', type=int, default=5,
                        help="Number of labels assigned to each client.")
    parser.add_argument('--partition-min-size', type=int, default=10,
                        help="Minimum per-client sample size for exdir splits.")
    parser.add_argument('--partition-train-ratio', type=float, default=0.75,
                        help="Train ratio slug incorporated into partition ID.")
    parser.add_argument('--partition-seed', type=int, default=1,
                        help="Seed used when building default partition IDs.")
    parser.add_argument('--data-partition', type=str, default="",
                        help="Optional descriptive ID for a pre-generated partition.")
    parser.add_argument('--dataset-hash', type=str, default="",
                        help="Optional hash for pre-generated dataset splits.")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model_family", type=str, default="HtM10")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=str2bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=300)
    parser.add_argument('-tc', "--top_cnt", type=int, default=50, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=str2bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=2,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp_des')
    parser.add_argument('-ab', "--auto_break", type=str2bool, default=True)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-mfn', "--models_folder_name", type=str, default='',
                        help="The folder of pre-trained models")
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=str2bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    parser.add_argument('--base_single_model', type=str2bool, default=False,
                        help="each FedDES client trains only one base classifier")
    # FedProto
    parser.add_argument('-lam', "--lamda", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=100)
    # FML
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    parser.add_argument('-bt', "--beta", type=float, default=1.0)
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.01)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=0.01)
    # FedTGP
    parser.add_argument('-mart', "--margin_threthold", type=float, default=100.0)
    # FedKTL
    parser.add_argument('-GPath', "--generator_path", type=str, default='stylegan/stylegan-xl-models/imagenet64.pkl')
    parser.add_argument('-prompt', "--stable_diffusion_prompt", type=str, default='a cat')
    parser.add_argument('-sbs', "--server_batch_size", type=int, default=100)
    parser.add_argument('-gbs', "--gen_batch_size", type=int, default=4,
                        help="Not related to the performance. A small value saves GPU memory.")
    parser.add_argument('-mu', "--mu", type=float, default=50.0)
    # FedMRL
    parser.add_argument('-sfd', "--sub_feature_dim", type=int, default=128)
    # Optional validation / balanced accuracy
    parser.add_argument('--use_val', type=str2bool, default=True)
    parser.add_argument('--val_ratio', type=float, default=0.25)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--use_bacc_metric', type=str2bool, default=False)
    parser.add_argument('--ckpt_root', type=str, default="ckpts",
                        help="Root directory for checkpoints/artifacts used by FedDES.")
    parser.add_argument('--outputs_root', type=str, default="../outputs",
                        help="Optional root for outputs/logs (pass-through for launcher compatibility).")
    parser.add_argument('--results_dir', type=str, default="../results",
                        help="Directory for saving .h5 results.")
    parser.add_argument('--phase', type=int, default=3,
                        help="FedDES phase: 1=base, 2=graph, 3=meta.")
    parser.add_argument('--use_sweeps', action="store_true",
                        help="Hint for sweep-mode runs (passed through from controller).")
    # FedDES-specific overrides (base/graph/gnn)
    parser.add_argument('--base_split_mode', type=str, default="split_train")
    parser.add_argument('--base_split_seed', type=int, default=123)
    parser.add_argument('--base_es_metric', type=str, default="val_loss")
    parser.add_argument('--base_es_patience', type=int, default=20)
    parser.add_argument('--base_weighted_by_class', type=str2bool, default=True)
    parser.add_argument('--base_es_min_delta', type=float, default=0.0001)
    parser.add_argument('--base_clf_lr', type=float, default=0.0005)
    parser.add_argument('--pool_calib_method', type=str, default="ts-mix")
    parser.add_argument('--graph_k_per_class', type=int, default=5)
    parser.add_argument('--proto_use', type=str2bool, default=False)
    parser.add_argument('--proto_min_samples', type=int, default=5)
    parser.add_argument('--proto_max_k', type=int, default=5)
    parser.add_argument('--graph_cs_topk', type=int, default=3)
    parser.add_argument('--graph_cc_topk', type=int, default=3)
    parser.add_argument('--graph_cs_mode', type=str, default="balanced_acc:logloss")
    parser.add_argument('--graph_meta_min_pos', type=int, default=3)
    parser.add_argument('--graph_sample_node_feats', type=str, default="embedding_mean")
    parser.add_argument('--graph_drop_disconnected_cls', type=str2bool, default=False)
    parser.add_argument('--gnn_arch', type=str, default="gat")
    parser.add_argument('--gnn_heterogenous', type=str2bool, default=False)
    parser.add_argument('--gnn_hidden_dim', type=int, default=128)
    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--gnn_heads', type=int, default=4)
    parser.add_argument('--gnn_dropout', type=float, default=0.2)
    parser.add_argument('--gnn_lr', type=float, default=0.0005)
    parser.add_argument('--gnn_weight_decay', type=float, default=0.0001)
    parser.add_argument('--gnn_use_sample_residual', type=str2bool, default=False)
    parser.add_argument('--gnn_use_edge_attr', type=str2bool, default=False)
    parser.add_argument('--gnn_drop_edge_rate', type=float, default=0.0)
    parser.add_argument(
        '--gnn_sampler',
        type=str,
        default="none",
        help="Neighbor sampling mode: none | unweighted | weighted",
    )
    parser.add_argument(
        '--gnn_sample_feat_norm',
        type=str,
        default="none",
        help="Sample feature normalization: none | layernorm | l2",
    )
    parser.add_argument('--gnn_debug_feat_stats', type=str2bool, default=False)
    parser.add_argument('--gnn_debug_degrees', type=str2bool, default=False)
    parser.add_argument('--gnn_pair_decoder', type=str, default=None)
    parser.add_argument('--gnn_bidirectionality', type=str2bool, default=False)
    parser.add_argument('--gnn_es_metric', type=str, default="val_loss")
    parser.add_argument('--gnn_patience', type=int, default=20)
    parser.add_argument('--gnn_loss', type=str, default="meta_labels_BCE")
    parser.add_argument('--gnn_sample_weight_mode', type=str, default="class_prevalence")
    parser.add_argument('--gnn_ens_combination_mode', type=str, default="soft",
                        help="Ensemble combination: soft (weights * probs) or hard (weights * one-hot preds).")
    parser.add_argument('--gnn_learned_cc_attrs', type=str2bool, default=False)
    parser.add_argument('--gnn_update_classifier_nodes', type=str2bool, default=False)
    parser.add_argument('--gnn_learned_cc_attr_dim', type=int, default=32)
    parser.add_argument('--gnn_drop_cc_edges', type=str2bool, default=False,
                        help="If true, ignore classifier-classifier edges when building the meta-learner.")
    parser.add_argument('--gnn_diversity_regularization', type=str2bool, default=False)
    parser.add_argument('--gnn_diversity_lambda', type=float, default=0.1)
    parser.add_argument('--gnn_diversity_eps', type=float, default=1e-6)
    parser.add_argument('--gnn_diversity_binary_neighbor_k_cap', type=int, default=9)
    parser.add_argument('--gnn_rank_lambda', type=float, default=0.0)
    parser.add_argument('--gnn_rank_margin', type=float, default=0.5)
    parser.add_argument('--gnn_rank_max_pairs', type=int, default=256)
    parser.add_argument('--gnn_rank_sample_n', type=int, default=256)
    parser.add_argument('--gnn_weight_rank_loss', type=str2bool, default=False)
    parser.add_argument('--gnn_use_pos_weight', type=str2bool, default=False)
    parser.add_argument('--eicu_seq_len', type=int, default=None,
                        help="Sequence length for eICU time-series models.")
    parser.add_argument('--eicu_in_channels', type=int, default=None,
                        help="Input channel count for eICU time-series models.")
    parser.add_argument('--skip_meta_training', type=str2bool, default=False,
                        help="If true, skip Phase 3 meta training when results already exist.")
    # FedPAE-specific overrides (ensemble selection)
    parser.add_argument('--pae_pop_size', type=int, default=40)
    parser.add_argument('--pae_num_generations', type=int, default=40)
    parser.add_argument('--pae_mutation_prob', type=float, default=0.05)
    parser.add_argument('--pae_crossover_prob', type=float, default=0.9)
    parser.add_argument('--pae_min_ensemble_size', type=int, default=1)
    parser.add_argument('--pae_max_ensemble_size', type=int, default=0,
                        help="0 means use all available models for max ensemble size.")
    parser.add_argument('--skip_pae_training', type=str2bool, default=False,
                        help="If true, skip Phase 2 ensemble selection when results already exist.")

    args = parser.parse_args()
    env_data_partition = os.environ.get("DATASET_PARTITION", "").strip()
    if not args.data_partition and env_data_partition:
        args.data_partition = env_data_partition

    if args.device_id not in (None, "", "-1"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    # Normalize ckpt_root to Path for downstream code
    try:
        from pathlib import Path
        args.ckpt_root = Path(args.ckpt_root).expanduser()
        if getattr(args, "outputs_root", None):
            args.outputs_root = Path(args.outputs_root).expanduser()
        if getattr(args, "results_dir", None):
            args.results_dir = Path(args.results_dir).expanduser()
    except Exception:
        pass

    dataset_arg = args.dataset.replace("\\", "/")
    partition_arg = args.data_partition.strip()
    dataset_name = dataset_arg.split("/", 1)[0]
    inline_partition = ""
    if "/" in dataset_arg:
        inline_partition = dataset_arg.split("/", 1)[1]
    partition_id = partition_arg or inline_partition
    if not partition_id:
        if dataset_name.lower().startswith("eicu"):
            task_name = extract_eicu_task_from_dataset(dataset_name)
            partition_id = build_eicu_partition_id(
                task=task_name,
                min_size=args.partition_min_size,
                seed=args.partition_seed,
                train_ratio=args.partition_train_ratio,
                num_clients=args.num_clients,
            )
        else:
            partition_id = build_dataset_partition_id(
                args.partition_type,
                args.partition_alpha,
                args.partition_C,
                args.partition_min_size,
                args.partition_train_ratio,
                args.partition_seed,
                args.num_clients,
            )
    print("partition_id:", partition_id)
    args.dataset_name = dataset_name
    args.data_partition = partition_id
    dataset_path = (
        f"{dataset_name}/{partition_id}" if partition_id else dataset_name
    )
    print("dataset_path:", dataset_path)
    args.dataset = dataset_path
    hash_tag = partition_id.replace("/", "-") if partition_id else ""
    args.dataset_tag = dataset_name if not hash_tag else f"{dataset_name}-{hash_tag}"

    if args.algorithm not in ["FedDES", "FedPAE"]:
        print("=" * 50)
        for arg in vars(args):
            print(arg, '=',getattr(args, arg))
        print("=" * 50)


    # if args.dataset == "mnist" or args.dataset == "fmnist":
    #     generate_mnist('../dataset/mnist/', args.num_clients, 10, args.niid)
    # elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
    #     generate_cifar10('../dataset/Cifar10/', args.num_clients, 10, args.niid)
    # else:
    #     generate_synthetic('../dataset/synthetic/', args.num_clients, 10, args.niid)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)

    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")

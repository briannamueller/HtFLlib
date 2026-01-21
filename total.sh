#!/usr/bin/env bash
#$ -S /bin/bash
#$ -N total
#$ -cwd
#$ -q UI-GPU,MANSCI-GPU,COB-GPU

#$ -pe smp 40
#$ -l ngpus=3

##$ -o total.log
##$ -e total.err

# Activate the Conda environment
eval "$(${HOME}/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate pfllib-des-cu117
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"


# ---- Make GPU visibility explicit (after env activation) ----
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# If the scheduler already exposes 0,1,2 this keeps it; otherwise set it.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2
fi


cd system

# python -u main.py -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 10 -data Cifar10 -m HtFE-img-3 -fd 256 -did 0 -algo Local > total-Cifar10-HtFE-img-3-fd=256-Local.out 2>&1 &
# python -u main.py -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 10 -data Cifar10 -m HtFE-img-3 -fd 256 -did 1 -algo FedProto -lam 10 > total-Cifar10-HtFE-img-3-fd=256-FedProto.out 2>&1 &
# python -u main.py -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 10 -data Cifar10 -m HtFE-img-3 -fd 256 -did 1 -algo FedGen -nd 32 -glr 0.1 -hd 512 -se 100 > total-Cifar10-HtFE-img-3-fd=256-FedGen.out 2>&1 &
# python -u main.py -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 10 -data Cifar10 -m HtFE-img-3 -fd 256 -did 2 -algo FD -lam 1 > total-Cifar10-HtFE-img-3-fd=256-FD.out 2>&1 &
# python -u main.py -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 10 -data Cifar10 -m HtFE-img-3 -fd 256 -did 2 -algo FML -al 0.5 -bt 0.5 > total-Cifar10-HtFE-img-3-fd=256-FML.out 2>&1 &
# python -u main.py -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 10 -data Cifar10 -m HtFE-img-3 -fd 256 -did 3 -algo FedKD -mlr 0.01 -Ts 0.95 -Te 0.98 > total-Cifar10-HtFE-img-3-fd=256-FedKD.out 2>&1 &
# python -u main.py -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 10 -data Cifar10 -m HtFE-img-3 -fd 256 -did 3 -algo LG-FedAvg > total-Cifar10-HtFE-img-3-fd=256-LG-FedAvg.out 2>&1 &
# python -u main.py -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 10 -data Cifar10 -m HtFE-img-3 -fd 256 -did 3 -algo FedGH -slr 0.01 -se 1 > total-Cifar10-HtFE-img-3-fd=256-FedGH.out 2>&1 &
# python -u main.py -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 10 -data Cifar10 -m HtFE-img-3 -fd 256 -did 4 -algo FedTGP -lam 10 -se 100 -mart 100 > total-Cifar10-HtFE-img-3-fd=256-FedTGP.out 2>&1 &
# python -u main.py -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 10 -data Cifar10 -m HtFE-img-3 -fd 256 -did 5 -algo FedKTL-stylegan-xl -slr 0.01 -sbs 100 -se 100 -lam 1 -mu 50 > total-Cifar10-HtFE-img-3-fd=256-FedKTL-stylegan-xl.out 2>&1 &
# python -u main.py -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 10 -data Cifar10 -m HtFE-img-3 -fd 256 -did 2 -algo FedKTL-stylegan-3 -slr 0.01 -sbs 100 -se 100 -lam 0.1 -mu 50 -GPath stylegan/stylegan-3-models/Benches-512.pkl > total-Cifar10-HtFE-img-3-fd=256-FedKTL-stylegan-3.out 2>&1 &
# python -u main.py -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 10 -data Cifar10 -m HtFE-img-3 -fd 256 -did 1 -algo FedKTL-stable-diffusion -slr 0.1 -sbs 100 -se 100 -lam 0.01 -mu 100 -GPath stable-diffusion/v1.5 > total-Cifar10-HtFE-img-3-fd=256-FedKTL-stable-diffusion.out 2>&1 &
# python -u main.py -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 10 -data Cifar10 -m HtFE-img-3 -fd 256 -did 4 -algo FedMRL -sfd 128 > total-Cifar10-HtFE-img-3-fd=256-FedMRL.out 2>&1 &

metric=acc
data=Cifar10
models=HtFE-img-5
fd=256
alpha=10
type=exdir
C=5
min_size=10
train_ratio=0.8
seed=1
nc=20
use_val=true

declare -A method_opts=(
    [Local]="-algo Local"
    [FedProto]="-algo FedProto -lam 10"
    [FedGen]="-algo FedGen -nd 32 -glr 0.1 -hd 512 -se 100"
    [FD]="-algo FD -lam 1"
    [FML]="-algo FML -al 0.5 -bt 0.5"
    [FedKD]="-algo FedKD -mlr 0.01 -Ts 0.95 -Te 0.98"
    [LG-FedAvg]="-algo LG-FedAvg"
    [FedGH]="-algo FedGH -slr 0.01 -se 1"
    [FedTGP]="-algo FedTGP -lam 10 -se 100 -mart 100"
    [FedKTL-stylegan-xl]="-algo FedKTL-stylegan-xl -slr 0.01 -sbs 100 -se 100 -lam 1 -mu 50"
    [FedKTL-stylegan-3]="-algo FedKTL-stylegan-3 -slr 0.01 -sbs 100 -se 100 -lam 0.1 -mu 50 -GPath stylegan/stylegan-3-models/Benches-512.pkl"
    [FedKTL-stable-diffusion]="-algo FedKTL-stable-diffusion -slr 0.1 -sbs 100 -se 100 -lam 0.01 -mu 100 -GPath stable-diffusion/v1.5"
    [FedMRL]="-algo FedMRL -sfd 128"
)

if [[ "$metric" == "bacc" ]]; then
    use_bacc_args="--use_bacc_metric true"
else
    use_bacc_args=""
fi

partition_id="${type}[alpha=${alpha},C=${C},min_size=${min_size}]_nc[${nc}]_tr[${train_ratio}]_s[${seed}]"
dataset_tag="${data}-${partition_id}"
run_root="PFL/${metric}"
log_root="${run_root}/logs/${dataset_tag}"
results_dir="${run_root}/results"

mkdir -p "${log_root}"

max_gpus=3
methods=(FedTGP LG-FedAvg FedGH)
sfn="${run_root}"

idx=0
for method in "${methods[@]}"; do
    did=$(( idx % max_gpus ))
    extra="${method_opts[$method]}"
    out="${log_root}/${method}_${models}-fd=${fd}.out"
    python -u main.py -ab 1 -lr 0.01 -jr 1 -lbs 16 -ls 1 -nc $nc -ncl 10 -gr 300 \
      -data "$data" -m "$models" -fd "$fd" -did "$did" \
      ${extra} -sfn "$sfn" --results_dir "$results_dir" ${use_bacc_args} \
      --partition-alpha "$alpha" --partition-type "$type" --partition-C "$C" --partition-min-size "$min_size" \
      --partition-train-ratio "$train_ratio" --partition-seed "$seed" --use_val "$use_val" \
      > "${out}" 2>&1 &
    idx=$((idx + 1))
done
wait

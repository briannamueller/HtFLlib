#!/usr/bin/env bash
#$ -S /bin/bash
#$ -N eICU_bacc
#$ -cwd
#$ -q UI-GPU,MANSCI-GPU,COB-GPU

#$ -pe smp 50
#$ -l ngpus=3

##$ -o /dev/null
##$ -e /dev/null

# ---- Environment ----
eval "$(${HOME}/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate pfllib-des-cu117
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=1
# export WANDB_CONSOLE=wrap

# ---- User config ----
metric="bacc"

algo="FedDES"
task="mortality_48h"
data="eICU_task=[${task}]"
models="eICU"
ncl=2
nc=3
phase=1


# ---- Paths ----
log_root="${PWD}/logs"
log_dir="${log_root}/${data}"
cache_dir="/Shared/lss_brimueller"
# Allow override from controller args; fallback to legacy path if not provided.
ckpt_root="/Shared/lss_brimueller/ckpts/${data}"
args=("$@")
for ((idx=0; idx<${#args[@]}; idx++)); do
  token="${args[$idx]}"
  case "$token" in
    --ckpt_root=*)
      ckpt_root="${token#--ckpt_root=}"
      ;;
    --ckpt_root)
      next_idx=$((idx+1))
      if [[ $next_idx -lt ${#args[@]} ]]; then
        ckpt_root="${args[$next_idx]}"
      fi
      ;;
  esac
done
outputs_root="${PWD}/outputs/${data}"
mkdir -p "${log_dir}"

# ---- Derived ----
project="${data}_[${metric}]_nc[${nc}]"
job_id="${SLURM_JOB_ID:-${PBS_JOBID:-${LSB_JOBID:-${JOB_ID:-$$}}}}"
log_file="${log_dir}/${data}_[${metric}]_job[${job_id}].log"

# ---- Launch ----
cd system
python -u main.py \
  -data "${data}" -nc "${nc}" -algo "${algo}" -m "${models}" -ncl "${ncl}" \
  -gr 1 -ls 30 -lbs 16 -jr 1.0 -eg 1 \
  -sfn "${data}_${metric}/temp_nc[${nc}]" \
  --ckpt_root "${ckpt_root}" --outputs_root "${outputs_root}" \
  --use_val true --val_ratio 0.2 --split_seed 123 \
  "$@" >"${log_file}" 2>&1

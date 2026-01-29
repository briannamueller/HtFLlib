#!/usr/bin/env bash
#
# Minimal SGE directives so the script can be submitted directly but does not
# redirect output to /dev/null; logs are therefore available for debugging.

#$ -S /bin/bash
#$ -N data_gen
#$ -cwd
#$ -q UI-GPU,MANSCI-GPU,COB-GPU
###$ -l gpu.cuda.0.mem_free=50

#$ -pe smp 40
#$ -l ngpus=1

eval "$(${HOME}/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate pfllib-des-cu117
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=1

export EXDIR_DEBUG=1
export EXDIR_DEBUG_EVERY=10


set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$(pwd)}}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/dataset}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/system/conf/configs.yaml}"

DATASET_TYPE="${1:-eicu}"
shift || true
declare -a EXTRA_ARGS=("$@")

case "${DATASET_TYPE,,}" in
  cifar10|cifar)
    FAMILY="Cifar10"
    GENERATOR="generate_Cifar10.py"
    ;;
  cifar100)
    FAMILY="Cifar100"
    GENERATOR="generate_Cifar100.py"
    ;;
  eicu*)
    FAMILY="eicu"
    GENERATOR="generate_eICU.py"
    ;;
  *)
    echo "Unsupported dataset type '${DATASET_TYPE}'. Use 'Cifar10' or 'eicu'."
    exit 1
    ;;
esac

export CONFIG_FILE FAMILY

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Dataset directory '${DATASET_DIR}' not found. Run from the repository root or set REPO_ROOT." >&2
  exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Configuration file '${CONFIG_FILE}' not found." >&2
  exit 1
fi

cd "${DATASET_DIR}"

PYTHON_BINARY="${PYTHON_BINARY:-python3}"

set +o noglob
mapfile -t DEFAULT_CLI < <("${PYTHON_BINARY}" - <<'PY'
import os
from pathlib import Path

import yaml

config_path = Path(os.environ["CONFIG_FILE"])
cfg = yaml.safe_load(config_path.read_text())
pipeline = cfg.get("pipeline", {})
family = os.environ["FAMILY"]
family_args = pipeline.get("family_args", {})
family_cfg = family_args.get(family, {})
family_data = family_cfg.get("data_partition_args", {})
eicu_defaults = family_args.get("eicu_defaults", {}).get("data_partition_args", {})
eicu_allowed = {
    "task",
    "data_dir",
    "output_root",
    "dataset_name",
    "partition_id",
    "train_ratio",
    "min_size",
    "seed",
    "num_clients",
    "prefer_positive",
    "client_sort_mode",
}
data = pipeline.get("data_partition_args", {}).copy()
if family == "eicu" and eicu_defaults:
    data.update(eicu_defaults)
if family_data:
    data.update(family_data)

skip_keys = {"skip_partition_cli", "skip_partition_generation"}
args = []

def add_bool_flag(key: str, value: bool) -> None:
    if key == "niid":
        args.append("--niid" if value else "--iid")
        return
    if key == "balance":
        args.append("--balance" if value else "--no-balance")
        return
    if value and key not in {"niid", "balance"}:
        args.append(f"--{key.replace('_', '-')}")

for key in sorted(data):
    if key in skip_keys:
        continue
    if family == "eicu" and key not in eicu_allowed:
        continue
    value = data[key]
    if isinstance(value, bool):
        add_bool_flag(key, value)
        continue
    if value is None:
        continue
    args.append(f"--{key.replace('_', '-')}")
    args.append(str(value))

print("\n".join(args))
PY
)
set -o noglob

set +u
filtered_cli=()
for arg in "${DEFAULT_CLI[@]}"; do
  [[ -n "${arg}" ]] && filtered_cli+=("${arg}")
done
DEFAULT_CLI=("${filtered_cli[@]}")
set -u

PYTHONPATH="${PYTHONPATH:-}"
PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}:${REPO_ROOT}/system"
export PYTHONPATH

cmd=("${PYTHON_BINARY}" -u "${GENERATOR}")
if [[ ${#DEFAULT_CLI[@]} -gt 0 ]]; then
  cmd+=("${DEFAULT_CLI[@]}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

"${cmd[@]}"

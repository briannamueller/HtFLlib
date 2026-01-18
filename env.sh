#!/usr/bin/env bash

# Environment setup for FedDES jobs
# Avoid nounset issues inside activation scripts by relaxing -u temporarily.
set +u
eval "$(${HOME}/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate pfllib-des-cu117
set -u

# Compute repo root relative to this file to avoid CWD surprises
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT

# Ensure MKL vars are defined to satisfy activate.d scripts
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-GNU}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-INTEL}

# Ensure project root is on PYTHONPATH so local modules resolve inside batch jobs
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

# Wandb defaults (can be overridden upstream)
export WANDB_ENTITY="${WANDB_ENTITY:-brianna-mueller-university-of-iowa}"
export WANDB_PROJECT="${WANDB_PROJECT:-FedDES}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=1

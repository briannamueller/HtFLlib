#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# worker_prototypes.sh (2-file version)
#
# MODE=task:
#   PHASE=base|graph|meta_grid|pae
#   Uses SGE_TASK_ID as base_idx
#   Reads configs_*.txt / *_ids.txt from RUN_DIR
#   Marker-skips if *_DONE.marker exists
#   Runs: cd ${REPO_ROOT}/system && python main_prototypes.py ... --phase {1|2|3} --ckpt_root ...
#
# MODE=sweep_agent (Option B, ARRAY per g):
#   Receives RUN_DIR, G_IDX, ckpt_root, SWEEP_IDS_FILE
#   Uses SGE_TASK_ID as B_IDX to:
#     - look up sweep id from SWEEP_IDS_FILE (line b)
#     - check the corresponding graph marker once
#     - run: wandb agent --count SWEEP_MAX_RUNS $SWEEP_ID
#
# Env activation:
#   - micromamba activate pfllib-des-cu117
#   - export REPO_ROOT / PYTHONPATH / MKL vars / WANDB defaults / LD_LIBRARY_PATH
# -----------------------------------------------------------------------------

MODE="${MODE:?MODE missing}"          # "task" or "sweep_agent"
REPO_ROOT="${REPO_ROOT:?REPO_ROOT missing}"
export REPO_ROOT

# -------------------------
# Inline env activation
# -------------------------
set +u
module purge
eval "$(${HOME}/micromamba/bin/micromamba shell hook --shell bash)"
CONDA_ENV="${CONDA_ENV:-pygtest-cu117}"
micromamba activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
set -u

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-INTEL}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Let wandb stop hijacking stdout/stderr so qsub logs show our prints.
export WANDB_REDIRECT_STDOUT="${WANDB_REDIRECT_STDOUT:-false}"
export WANDB_REDIRECT_STDERR="${WANDB_REDIRECT_STDERR:-false}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

# Store all wandb artifacts/logs under the shared filesystem.
export WANDB_DIR="${WANDB_DIR:-/Shared/lss_brimueller/wandb}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-/Shared/lss_brimueller/wandb/artifacts}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/Shared/lss_brimueller/wandb/data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/Shared/lss_brimueller/wandb/cache}"

# Ensure the shared directories exist and are writable.
for wandb_path in "${WANDB_DIR}" "${WANDB_ARTIFACT_DIR}" "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}"; do
  if [[ -n "${wandb_path}" && ! -d "${wandb_path}" ]]; then
    mkdir -p "${wandb_path}" || true
  fi
done

# Minimal PYTHONPATH: project root only; main_prototypes.py will handle anything else.
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

export WANDB_ENTITY="${WANDB_ENTITY:-brianna-mueller-university-of-iowa}"
export WANDB_PROJECT="${WANDB_PROJECT:-FedDES}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=1

# Optional: print PyG/torch versions once and exit.
if [[ "${DEBUG_PYG_VERSIONS:-0}" == "1" ]]; then
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
  python -c "import torch_geometric; print('pyg', torch_geometric.__version__)"
  python -c "import pyg_lib, torch_scatter, torch_sparse, torch_cluster; print('pyg_lib', pyg_lib.__version__); print('torch_scatter', torch_scatter.__version__); print('torch_sparse', torch_sparse.__version__); print('torch_cluster', torch_cluster.__version__)"
  exit 0
fi



# -------------------------
# MODE=sweep_agent (Option B, ARRAY JOB per g)
# -------------------------
if [[ "${MODE}" == "sweep_agent" ]]; then
  WANDB_BIN="${WANDB_BIN:-wandb}"
  SWEEP_MAX_RUNS="${SWEEP_MAX_RUNS:-300}"
  RUN_DIR="${RUN_DIR:?RUN_DIR missing}"
  ALLOW_NONPROT_RUN_DIR="${ALLOW_NONPROT_RUN_DIR:-0}"
  if [[ "${ALLOW_NONPROT_RUN_DIR}" -ne 1 && "${RUN_DIR}" != *"_prototypes_"* ]]; then
    echo "[worker][sweep_agent] ERROR: RUN_DIR must include '_prototypes_' for prototype runs."
    echo "[worker][sweep_agent] RUN_DIR=${RUN_DIR}"
    exit 2
  fi
  G_IDX="${G_IDX:?G_IDX missing}"
  ckpt_root="${ckpt_root:?ckpt_root missing}"
  SWEEP_IDS_FILE="${SWEEP_IDS_FILE:?SWEEP_IDS_FILE missing}"

  BASE_IDS="${RUN_DIR}/base_ids.txt"
  GRAPH_IDS="${RUN_DIR}/graph_ids.txt"

  # Use SGE_TASK_ID as B_IDX (1..n_base)
  B_IDX="${SGE_TASK_ID:-0}"
  if [[ "${B_IDX}" -le 0 ]]; then
    echo "[worker][sweep_agent] ERROR: SGE_TASK_ID missing or invalid"
    exit 2
  fi

  # Look up sweep ID for this (g, b) from the file written by run_pipeline.sh
  SWEEP_ID="$(sed -n "${B_IDX}p" "${SWEEP_IDS_FILE}" || true)"
  if [[ -z "${SWEEP_ID}" ]]; then
    echo "[worker][sweep_agent] ERROR: No sweep id found for g=${G_IDX}, b=${B_IDX} in ${SWEEP_IDS_FILE}"
    exit 2
  fi

  base_fp="$(sed -n "${B_IDX}p" "${BASE_IDS}")"
  graph_fp="$(sed -n "${G_IDX}p" "${GRAPH_IDS}")"
  marker="${ckpt_root}/graphs/base[${base_fp}]_graph[${graph_fp}]_DONE.marker"

  echo "[worker][sweep_agent] WANDB_PROJECT=${WANDB_PROJECT}"
  echo "[worker][sweep_agent] WANDB_ENTITY=${WANDB_ENTITY}"
  echo "[worker][sweep_agent] SWEEP_ID=${SWEEP_ID}"
  echo "[worker][sweep_agent] SWEEP_MAX_RUNS=${SWEEP_MAX_RUNS}"
  echo "[worker][sweep_agent] G_IDX=${G_IDX} B_IDX=${B_IDX}"
  echo "[worker][sweep_agent] Expecting marker: ${marker}"

  # With -hold_jid_ad on the graph array, this marker SHOULD exist already.

  echo "[worker][sweep_agent] Starting wandb agent."
  "${WANDB_BIN}" agent --count "${SWEEP_MAX_RUNS}" "${SWEEP_ID}"
  exit 0
fi

# -------------------------
# MODE=task
# -------------------------
if [[ "${MODE}" != "task" ]]; then
  echo "[worker] ERROR: Unknown MODE='${MODE}'"
  exit 2
fi

PHASE="${PHASE:?PHASE missing}"       # base|graph|meta_grid|pae
RUN_DIR="${RUN_DIR:?RUN_DIR missing}"
ALLOW_NONPROT_RUN_DIR="${ALLOW_NONPROT_RUN_DIR:-0}"
if [[ "${ALLOW_NONPROT_RUN_DIR}" -ne 1 && "${RUN_DIR}" != *"_prototypes_"* ]]; then
  echo "[worker][task] ERROR: RUN_DIR must include '_prototypes_' for prototype runs."
  echo "[worker][task] RUN_DIR=${RUN_DIR}"
  exit 2
fi
if [[ -z "${COMMON_ARGS:-}" ]]; then
  if [[ -n "${COMMON_ARGS_B64:-}" ]]; then
    COMMON_ARGS="$(printf '%s' "${COMMON_ARGS_B64}" | base64 --decode)"
  fi
fi
COMMON_ARGS="${COMMON_ARGS:?COMMON_ARGS missing}"
ckpt_root="${ckpt_root:?ckpt_root missing}"
outputs_root="${outputs_root:-${ckpt_root}}"

BASE_CFG="${RUN_DIR}/configs_base.txt"
GRAPH_CFG="${RUN_DIR}/configs_graph.txt"
GNN_CFG="${RUN_DIR}/configs_gnn.txt"
PAE_CFG="${RUN_DIR}/configs_pae.txt"

BASE_IDS="${RUN_DIR}/base_ids.txt"
GRAPH_IDS="${RUN_DIR}/graph_ids.txt"
GNN_IDS="${RUN_DIR}/gnn_ids.txt"
PAE_IDS="${RUN_DIR}/pae_ids.txt"

base_idx="${SGE_TASK_ID:-0}"
[[ "${base_idx}" -gt 0 ]] || { echo "[worker][task] ERROR: SGE_TASK_ID missing"; exit 2; }

base_args="$(sed -n "${base_idx}p" "${BASE_CFG}")"
base_fp="$(sed -n "${base_idx}p" "${BASE_IDS}")"

case "${PHASE}" in
  base)
    marker="${ckpt_root}/base_clf/base[${base_fp}]_DONE.marker"
    if [[ -f "${marker}" ]]; then
      echo "[worker][base] DONE: ${marker}"
      exit 0
    fi

    # outputs_root="${outputs_root}/base[${base_fp}]"
    mkdir -p "${outputs_root}"
    echo "[worker][base] base_idx=${base_idx} base_fp=${base_fp}"
    cd "${REPO_ROOT}/system"
    python -u main_prototypes.py ${COMMON_ARGS} ${base_args} \
      --phase 1 \
      --ckpt_root "${ckpt_root}" \
      --outputs_root "${outputs_root}"

    ;;

  graph)
    GRAPH_IDX="${GRAPH_IDX:?GRAPH_IDX missing}"
    graph_args="$(sed -n "${GRAPH_IDX}p" "${GRAPH_CFG}")"
    graph_fp="$(sed -n "${GRAPH_IDX}p" "${GRAPH_IDS}")"

    marker="${ckpt_root}/graphs/base[${base_fp}]_graph[${graph_fp}]_DONE.marker"
    if [[ -f "${marker}" ]]; then
      echo "[worker][graph] DONE: ${marker}"
      exit 0
    fi

    echo "[worker][graph] base_idx=${base_idx} base_fp=${base_fp} graph_idx=${GRAPH_IDX} graph_fp=${graph_fp}"
    # outputs_root="${outputs_root}/graph_b[${base_fp}]_g[${graph_fp}]"
    mkdir -p "${outputs_root}"
    cd "${REPO_ROOT}/system"
    python -u main_prototypes.py ${COMMON_ARGS} ${base_args} ${graph_args} \
      --phase 2 \
      --ckpt_root "${ckpt_root}" \
      --outputs_root "${outputs_root}"

    ;;

  meta_grid)
    GRAPH_IDX="${GRAPH_IDX:?GRAPH_IDX missing}"
    GNN_IDX="${GNN_IDX:?GNN_IDX missing}"

    graph_args="$(sed -n "${GRAPH_IDX}p" "${GRAPH_CFG}")"
    gnn_args="$(sed -n "${GNN_IDX}p" "${GNN_CFG}")"
    graph_fp="$(sed -n "${GRAPH_IDX}p" "${GRAPH_IDS}")"
    gnn_fp="$(sed -n "${GNN_IDX}p" "${GNN_IDS}")"

    marker="${ckpt_root}/gnn/base[${base_fp}]_graph[${graph_fp}]_gnn[${gnn_fp}]_DONE.marker"
    if [[ -f "${marker}" ]]; then
      echo "[worker][meta_grid] DONE: ${marker}"
      exit 0
    fi

    echo "[worker][meta_grid] base_idx=${base_idx} base_fp=${base_fp} graph_idx=${GRAPH_IDX} gnn_idx=${GNN_IDX}"
    cd "${REPO_ROOT}/system"
    export PYTHONFAULTHANDLER=1
    # outputs_root="${outputs_root}/gnn_b[${base_fp}]_g[${graph_fp}]_k[${gnn_fp}]"
    mkdir -p "${outputs_root}"
    python -X faulthandler -u main_prototypes.py ${COMMON_ARGS} ${base_args} ${graph_args} ${gnn_args} \
      --phase 3 \
      --ckpt_root "${ckpt_root}" \
      --outputs_root "${outputs_root}"
    ;;

  pae)
    PAE_IDX="${PAE_IDX:?PAE_IDX missing}"

    pae_args="$(sed -n "${PAE_IDX}p" "${PAE_CFG}")"
    pae_fp="$(sed -n "${PAE_IDX}p" "${PAE_IDS}")"

    marker="${ckpt_root}/pae/base[${base_fp}]_pae[${pae_fp}]_DONE.marker"
    if [[ -f "${marker}" ]]; then
      echo "[worker][pae] DONE: ${marker}"
      exit 0
    fi

    echo "[worker][pae] base_idx=${base_idx} base_fp=${base_fp} pae_idx=${PAE_IDX}"
    mkdir -p "${outputs_root}"
    cd "${REPO_ROOT}/system"
    python -u main_prototypes.py ${COMMON_ARGS} ${base_args} ${pae_args} \
      --phase 2 \
      --ckpt_root "${ckpt_root}" \
      --outputs_root "${outputs_root}"
    ;;

  *)
    echo "[worker][task] ERROR: Unknown PHASE='${PHASE}'"
    exit 2
    ;;
esac

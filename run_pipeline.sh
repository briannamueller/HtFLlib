#!/usr/bin/env bash
set -euo pipefail

# Readability/organization TODOs:
# - Consolidate env/config setup into helpers (activate_env, load_defaults, ensure_dirs)
# - Centralize logging/error helpers (log, warn, die) and replace raw echo statements
# - Wrap repeated qsub calls in submit_array()/submit_sweep_agent() helpers to de-duplicate flags
# - Keep marker/pending-list builders together; isolate sweep creation into create_sweep()
# - Add a clear main() flow with early returns for ACTION=cancel vs ACTION=run
# - Co-locate constants by domain (cluster, meta, wandb); lift inline literals into vars
# - Separate meta sweep vs meta grid paths into dedicated functions gated by flags
# - Prefer local variables + function parameters over globals where practical

# -----------------------------------------------------------------------------
# run_pipeline.sh  (2-file version, Option B for sweeps + run-scoped kill lists)
#
# Main modes:
#   ACTION=run    (default): plan + submit jobs
#   ACTION=cancel           : bulk qdel jobs for RUN_DIR + cancel wandb sweeps
#
# Usage:
#   # normal
#   bash run_pipeline.sh
#   EXP_FAMILY=Cifar10 use_meta_grid=1 bash run_pipeline.sh
#   use_meta_sweeps=1 use_meta_grid=0 bash run_pipeline.sh
#
#   # cancel ONLY this run (no other experiments)
#   ACTION=cancel RUN_DIR=/path/to/run_configs/eicu_YYYYMMDDThhmmss bash run_pipeline.sh
# -----------------------------------------------------------------------------

ACTION="${ACTION:-run}"
set +u
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-INTEL}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
module purge
eval "$(${HOME}/micromamba/bin/micromamba shell hook --shell bash)"
CONDA_ENV="${CONDA_ENV:-pygtest-cu117}"
micromamba activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Fail fast with a clear error if PyYAML is missing (required for configs.yaml parsing).
python - <<'PY'
try:
    import yaml  # noqa: F401
except Exception as e:
    print("[ERROR] Missing Python dependency: PyYAML (import yaml) failed.")
    print("        Activate the same env and install it with ONE of:")
    print("          micromamba install -c conda-forge pyyaml")
    print("          pip install pyyaml")
    raise
PY
set -u
# -----------------------
# Predictable overrides
# -----------------------
EXP_FAMILY="${EXP_FAMILY:-${exp_family:-Cifar10}}"

# Paths
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}:${REPO_ROOT}/system"
export PYTHONPATH
CONFIGS_YAML="${CONFIGS_YAML:-${REPO_ROOT}/system/conf/configs.yaml}"
MAKE_PHASE_CONFIGS="${MAKE_PHASE_CONFIGS:-${REPO_ROOT}/make_phase_configs.py}"
WORKER_SH="${WORKER_SH:-${REPO_ROOT}/worker.sh}"

RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/run_configs}"
STAMP="${STAMP:-$(date +%Y%m%dT%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RUNS_ROOT}/${EXP_FAMILY}_${STAMP}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"

JOBS_TSV="${RUN_DIR}/sge_jobs.tsv"
SWEEPS_TSV="${RUN_DIR}/wandb_sweeps.tsv"

# Cluster defaults (override-able)
queue="${queue:-UI-GPU,MANSCI-GPU,COB-GPU}"
gpu_req="${gpu_req:-ngpus=1}"
base_gpu_req="${base_gpu_req:-ngpus=3}"
meta_gpu_req="${meta_gpu_req:-ngpus=1}"
# meta_gpu_req="${meta_gpu_req:-ngpus=1,gpu=true,gpu_k80=false,gpu_p100=false,gpu_1080ti=false,gpu_2080ti=false}"

hrt="${hrt:-48:00:00}"
mem="${mem:-32G}"
base_smp="${base_smp:-20}"
smp="${smp:-1}"

base_tc="${base_tc:-100}"
graph_tc="${graph_tc:-20}"
meta_tc="${meta_tc:-1}"
MAX_META_JOBS="${MAX_META_JOBS:-12}"           # 0 => unlimited
META_CHECK_INTERVAL="${META_CHECK_INTERVAL:-60}"  # seconds between qstat poll

# Meta modes
use_meta_grid="${use_meta_grid:-1}"      # PRIMARY
use_meta_sweeps="${use_meta_sweeps:-0}"  # OPTIONAL
single_default_configs="${single_default_configs:-0}"  # if 1, run only defaults from configs.yaml

# W&B sweep settings
WANDB_BIN="${WANDB_BIN:-wandb}"
SWEEP_METHOD="${SWEEP_METHOD:-grid}"
SWEEP_METRIC_NAME="${SWEEP_METRIC_NAME:-bacc_beats_baselines}"
SWEEP_METRIC_GOAL="${SWEEP_METRIC_GOAL:-maximize}"
SWEEP_MAX_RUNS="${SWEEP_MAX_RUNS:-500}"

PHASES="${PHASES:-${phases:-}}"
phase_enabled() {
  local target="$1"
  local phases=",${PHASES// /},"
  [[ "${phases}" == *",${target},"* ]]
}

# -----------------------------------------------------------------------------
# Cancel mode: qdel all jobs listed for this RUN_DIR and cancel all sweeps listed
# -----------------------------------------------------------------------------
if [[ "${ACTION}" == "cancel" ]]; then
  RUN_DIR="${RUN_DIR:?RUN_DIR is required for ACTION=cancel}"
  JOBS_TSV="${RUN_DIR}/sge_jobs.tsv"
  SWEEPS_TSV="${RUN_DIR}/wandb_sweeps.tsv"

  echo "[cancel] RUN_DIR=${RUN_DIR}"

  if [[ -f "${JOBS_TSV}" ]]; then
    echo "[cancel] Killing SGE jobs listed in ${JOBS_TSV}"
    # kill in reverse submission order to reduce dangling dependents
    awk -F'\t' 'NF>=2{print $2}' "${JOBS_TSV}" | tac | awk 'NF' | xargs -r qdel
  else
    echo "[cancel] No ${JOBS_TSV} found; skipping qdel."
  fi

  if [[ -f "${SWEEPS_TSV}" ]]; then
    echo "[cancel] Cancelling W&B sweeps listed in ${SWEEPS_TSV}"
    # Each line: g \t b \t entity/project/sweepid \t sweep_file
    while IFS=$'\t' read -r g b sweep_id sweep_file; do
      [[ -z "${sweep_id}" ]] && continue
      echo "[cancel] wandb sweep --cancel ${sweep_id}   (g=${g} b=${b})"
      "${WANDB_BIN}" sweep --cancel "${sweep_id}" || true
    done < "${SWEEPS_TSV}"
  else
    echo "[cancel] No ${SWEEPS_TSV} found; skipping wandb cancel."
  fi

  echo "[cancel] Done."
  exit 0
fi

load_pipeline_defaults() {
  local family="$1"
  local tmp
  tmp="$(mktemp)"
  python - "$CONFIGS_YAML" "$family" "${REPO_ROOT}" > "${tmp}" <<'PY'
import os
import sys
import yaml
from pathlib import Path

repo_root = Path(sys.argv[3])
sys.path.insert(0, str(repo_root))
from system.des.helpers import build_dataset_partition_id, build_eicu_partition_id

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
pipeline = cfg.get("pipeline", {})
family = sys.argv[2]
common = pipeline.get("common_args", {}).copy()
family_args = pipeline.get("family_args", {}).get(family, {})
combined = {**common, **family_args}
special = {"NC", "NUM_MODELS"}
ignored_pipeline_keys = {
  "partition_builder",
  "data_partition_args",
  "skip_partition_cli",
  "skip_partition_generation",
}
ignored_data_keys = {
  "partition",
  "alpha",
  "C",
  "min_size",
  "min_samples",
  "train_ratio",
  "seed",
  "niid",
  "balance",
}
out = {
  k: v
  for k, v in combined.items()
  if k not in special and k not in ignored_pipeline_keys and k not in ignored_data_keys
}
flag_map = {
  "algorithm": "-algo",
  "num_clients": "-nc",
  "global_rounds": "-gr",
  "local_epochs": "-ls",
  "use_val": "--use_val",
  "skip_meta_training": "--skip_meta_training",
  "data": "-data",
  "model_family": "-m",
  "num_classes": "-ncl",
  "feature_dim": "-fd"
}
parts = []
for key, value in out.items():
  flag = flag_map.get(
    key,
    f"--{'dataset' if key == 'data' else key.replace('_', '-')}"
  )
  val = str(value).lower() if isinstance(value, bool) else str(value)
  parts.append(f"{flag} {val}")
data_pipeline = pipeline.get("data_partition_args", {}).copy()
family_data = family_args.get("data_partition_args", {}) or {}
num_clients = family_args.get("num_clients", common.get("num_clients"))
if num_clients is None or num_clients == "":
  num_clients = combined.get("num_clients", 0)
builder = family_args.get("partition_builder")
if builder == "eicu":
  min_size = (
    family_data.get("min_size")
    or family_data.get("min_samples")
    or data_pipeline.get("min_size")
    or data_pipeline.get("min_samples")
    or 0
  )
  data_partition_id = build_eicu_partition_id(
    task=family_data.get("task", family),
    min_size=min_size,
    seed=family_data.get("seed", data_pipeline.get("seed", 0)),
    train_ratio=family_data.get("train_ratio", data_pipeline.get("train_ratio", 0.75)),
    num_clients=num_clients,
  )
else:
  merged_data = {**data_pipeline, **family_data}
  data_partition_id = build_dataset_partition_id(
    merged_data.get("partition", "dir"),
    merged_data.get("alpha", 1.0),
    merged_data.get("C", 5),
    merged_data.get("min_size"),
    merged_data.get("train_ratio", 0.75),
    merged_data.get("seed", 0),
    num_clients,
  )
parts.append(f"--data-partition {data_partition_id}")
print(f"common_args#{' '.join(parts)}")
print(f"NC#{family_args.get('NC', combined.get('num_clients', ''))}")
print(f"NUM_MODELS#{family_args.get('NUM_MODELS', '')}")
print(f"DATA_PARTITION#{data_partition_id}")
PY

  while IFS='#' read -r key value; do
    case "${key}" in
      common_args) pipeline_defaults_common="${value}" ;;
      NC) pipeline_defaults_nc="${value}" ;;
      NUM_MODELS) pipeline_defaults_num_models="${value}" ;;
      DATA_PARTITION) pipeline_defaults_data_partition="${value}" ;;
    esac
  done < "${tmp}"
  rm -f "${tmp}"

  WANDB_ENTITY="${WANDB_ENTITY:-brianna-mueller-university-of-iowa}"
  export WANDB_ENTITY
}

detect_pipeline_algo() {
  local algo="FedDES"
  if [[ "${COMMON_ARGS}" == *"-algo FedPAE"* || "${COMMON_ARGS}" == *"--algorithm FedPAE"* ]]; then
    algo="FedPAE"
  fi
  PIPELINE_ALGO="${algo}"
}

apply_algo_override() {
  local override="${algo:-${ALGO:-}}"
  if [[ -z "${override}" ]]; then
    return
  fi
  if [[ "${COMMON_ARGS}" == *"-algo "* || "${COMMON_ARGS}" == *"--algorithm "* ]]; then
    COMMON_ARGS="$(printf '%s' "${COMMON_ARGS}" | sed -E "s/(^|[[:space:]])-algo[[:space:]]+[^[:space:]]+/\1-algo ${override}/; s/(^|[[:space:]])--algorithm[[:space:]]+[^[:space:]]+/\1--algorithm ${override}/")"
  else
    COMMON_ARGS="${COMMON_ARGS} -algo ${override}"
  fi
}

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
: > "${JOBS_TSV}"
: > "${SWEEPS_TSV}"

pipeline_defaults_common=""
pipeline_defaults_nc=""
pipeline_defaults_num_models=""
pipeline_defaults_data_partition=""
load_pipeline_defaults "${EXP_FAMILY}"

pipeline_defaults_data_partition="${pipeline_defaults_data_partition:-}"
if [[ -z "${DATASET_PARTITION:-}" && -n "${pipeline_defaults_data_partition}" ]]; then
  DATASET_PARTITION="${pipeline_defaults_data_partition}"
fi
export DATASET_PARTITION

if [[ -z "${COMMON_ARGS:-}" ]]; then
  COMMON_ARGS="${pipeline_defaults_common}"
fi
apply_algo_override
COMMON_ARGS_B64="$(printf '%s' "${COMMON_ARGS}" | base64 | tr -d '\n')"
detect_pipeline_algo
if [[ -z "${PHASES}" ]]; then
  if [[ "${PIPELINE_ALGO}" == "FedPAE" ]]; then
    PHASES="1,2"
  else
    PHASES="2,3"
  fi
fi
NC="${NC:-${pipeline_defaults_nc}}"
NUM_MODELS="${NUM_MODELS:-${pipeline_defaults_num_models}}"
ckpt_partition="${DATASET_PARTITION:+/${DATASET_PARTITION}}"
ckpt_root="${ckpt_root:-/Shared/lss_brimueller/${EXP_FAMILY}${ckpt_partition}}"
export ckpt_root
outputs_root="${outputs_root:-${ckpt_root}}"
export outputs_root

if [[ -z "${WANDB_PROJECT:-}" ]]; then
  if [[ -n "${DATASET_PARTITION:-}" ]]; then
    WANDB_PROJECT="${EXP_FAMILY}_${DATASET_PARTITION}"
  else
    WANDB_PROJECT="${EXP_FAMILY}"
  fi
  WANDB_PROJECT="${WANDB_PROJECT}_${PIPELINE_ALGO}"
fi
export WANDB_PROJECT

make_phase_extra_args=""
if [[ "${single_default_configs}" -eq 1 ]]; then
  use_meta_sweeps=0
  use_meta_grid=1
  make_phase_extra_args="--single-config"
fi

echo "[submit_all] EXP_FAMILY=${EXP_FAMILY}"
echo "[submit_all] RUN_DIR=${RUN_DIR}"
echo "[submit_all] LOG_DIR=${LOG_DIR}"
echo "[submit_all] COMMON_ARGS=${COMMON_ARGS}"
echo "[submit_all] COMMON_ARGS_B64=${COMMON_ARGS_B64}"
echo "[submit_all] PIPELINE_ALGO=${PIPELINE_ALGO}"
echo "[submit_all] ckpt_root=${ckpt_root}"
echo "[submit_all] outputs_root=${outputs_root}"
echo "[submit_all] NC=${NC} NUM_MODELS=${NUM_MODELS}"
echo "[submit_all] use_meta_sweeps=${use_meta_sweeps} use_meta_grid=${use_meta_grid}"
echo "[submit_all] single_default_configs=${single_default_configs}"
echo "[submit_all] WANDB_PROJECT=${WANDB_PROJECT} WANDB_ENTITY=${WANDB_ENTITY}"

# -----------------------------------------------------------------------------
# 2) Plan configs + sync markers (your make_phase_configs.py interface)
# -----------------------------------------------------------------------------

python "${MAKE_PHASE_CONFIGS}" \
  --configs-yaml "${CONFIGS_YAML}" \
  --out-dir "${RUN_DIR}" \
  --ckpt-root "${ckpt_root}" \
  --num-clients "${NC}" \
  --num-models "${NUM_MODELS}" \
  ${make_phase_extra_args:+${make_phase_extra_args}}

BASE_CFG="${RUN_DIR}/configs_base.txt"
GRAPH_CFG="${RUN_DIR}/configs_graph.txt"
GNN_CFG="${RUN_DIR}/configs_gnn.txt"
PAE_CFG="${RUN_DIR}/configs_pae.txt"

BASE_IDS="${RUN_DIR}/base_ids.txt"
GRAPH_IDS="${RUN_DIR}/graph_ids.txt"
GNN_IDS="${RUN_DIR}/gnn_ids.txt"
PAE_IDS="${RUN_DIR}/pae_ids.txt"

n_base="$(wc -l < "${BASE_CFG}" | tr -d ' ')"
n_graph="$(wc -l < "${GRAPH_CFG}" | tr -d ' ')"
n_gnn="$(wc -l < "${GNN_CFG}" | tr -d ' ')"
n_pae="$(wc -l < "${PAE_CFG}" | tr -d ' ')"

echo "[submit_all] n_base=${n_base} n_graph=${n_graph} n_gnn=${n_gnn} n_pae=${n_pae}"

# -----------------------------------------------------------------------------
# Helpers: build lists of base_idx that are NOT complete (by markers)
# -----------------------------------------------------------------------------
build_tlist_base() {
  local tlist="" b b_fp marker
  for b in $(seq 1 "${n_base}"); do
    b_fp="$(sed -n "${b}p" "${BASE_IDS}")"
    marker="${ckpt_root}/base_clf/base[${b_fp}]_DONE.marker"
    [[ -f "${marker}" ]] && continue
    tlist="${tlist:+${tlist},}${b}"
  done
  echo "${tlist}"
}

build_tlist_graph_for_g() {
  local g="$1"
  local tlist="" b b_fp g_fp marker
  g_fp="$(sed -n "${g}p" "${GRAPH_IDS}")"
  for b in $(seq 1 "${n_base}"); do
    b_fp="$(sed -n "${b}p" "${BASE_IDS}")"
    marker="${ckpt_root}/graphs/base[${b_fp}]_graph[${g_fp}]_DONE.marker"
    [[ -f "${marker}" ]] && continue
    tlist="${tlist:+${tlist},}${b}"
  done
  echo "${tlist}"
}

build_tlist_meta_grid_for_gk() {
  local g="$1" k="$2"
  local tlist="" b b_fp g_fp k_fp marker
  g_fp="$(sed -n "${g}p" "${GRAPH_IDS}")"
  k_fp="$(sed -n "${k}p" "${GNN_IDS}")"
  for b in $(seq 1 "${n_base}"); do
    b_fp="$(sed -n "${b}p" "${BASE_IDS}")"
    marker="${ckpt_root}/gnn/base[${b_fp}]_graph[${g_fp}]_gnn[${k_fp}]_DONE.marker"
    [[ -f "${marker}" ]] && continue
    tlist="${tlist:+${tlist},}${b}"
  done
  echo "${tlist}"
}

build_tlist_pae_for_p() {
  local p="$1"
  local tlist="" b b_fp p_fp marker
  p_fp="$(sed -n "${p}p" "${PAE_IDS}")"
  for b in $(seq 1 "${n_base}"); do
    b_fp="$(sed -n "${b}p" "${BASE_IDS}")"
    marker="${ckpt_root}/pae/base[${b_fp}]_pae[${p_fp}]_DONE.marker"
    [[ -f "${marker}" ]] && continue
    tlist="${tlist:+${tlist},}${b}"
  done
  echo "${tlist}"
}

# -----------------------------------------------------------------------------
# 3) Submit Phase 1: BASE (one array), task_id == base_idx
#    We ALWAYS submit 1-n_base so -hold_jid_ad works for graph,
#    but worker skips tasks with *_DONE.marker present.
# -----------------------------------------------------------------------------
base_tlist_pending="$(build_tlist_base)"
base_jobid=""

if phase_enabled 1; then
  if [[ -n "${base_tlist_pending}" ]]; then
    echo "[submit_all] BASE tasks pending (by marker): ${base_tlist_pending}"
    base_range="1-${n_base}"
    echo "[submit_all] Submitting BASE array over full range: ${base_range}"
    base_jobid="$(
      qsub -terse -N "des_base_${EXP_FAMILY}" -cwd -j y \
        -q "${queue}" -pe smp "${base_smp}" -l ${base_gpu_req},h_rt=${hrt} \
        -o "${LOG_DIR}/base.\$JOB_ID.\$TASK_ID.log" -tc "${base_tc}" \
        -t "${base_range}" \
        -v MODE="task",PHASE="base",RUN_DIR="${RUN_DIR}",REPO_ROOT="${REPO_ROOT}",COMMON_ARGS_B64="${COMMON_ARGS_B64}",WANDB_PROJECT="${WANDB_PROJECT}",WANDB_ENTITY="${WANDB_ENTITY}",ckpt_root,outputs_root \
        "${WORKER_SH}"
    )"
    base_jobid="${base_jobid%%.*}"
    echo -e "base\t${base_jobid}\tNA\tNA" >> "${JOBS_TSV}"
    echo "[submit_all] base_jobid=${base_jobid}"
  else
    echo "[submit_all] All base complete; skipping BASE."
  fi
else
  echo "[submit_all] Skipping Phase 1 (BASE) by request. PHASES=${PHASES}"
fi

# -----------------------------------------------------------------------------
# 4) Submit Phase 2: GRAPH (one array per g), task_id == base_idx
#    Dependency: graph(g).task(b) waits on base.task(b) via -hold_jid_ad
# -----------------------------------------------------------------------------
declare -A graph_jobid_by_g
if phase_enabled 2 && [[ "${PIPELINE_ALGO}" == "FedDES" ]]; then
  for g in $(seq 1 "${n_graph}"); do
    graph_tlist_pending="$(build_tlist_graph_for_g "${g}")"
    if [[ -z "${graph_tlist_pending}" ]]; then
      echo "[submit_all] Graphs complete for g=${g}; skipping."
      continue
    fi

    hold_opt=()
    if [[ -n "${base_jobid}" ]]; then
      # task-aligned dependency: both arrays are 1-n_base
      hold_opt=(-hold_jid_ad "${base_jobid}")
    fi

    graph_range="1-${n_base}"
    echo "[submit_all] GRAPH g=${g} tasks pending (by marker): ${graph_tlist_pending}"
    echo "[submit_all] Submitting GRAPH g=${g} array over full range: ${graph_range}"
    gjob="$(
      qsub -terse -N "des_graph_${EXP_FAMILY}_g${g}" -cwd -j y \
        -q "${queue}" -pe smp "${smp}" -l ${gpu_req},h_rt=${hrt} \
        -o "${LOG_DIR}/graph_g${g}.\$JOB_ID.\$TASK_ID.log" -tc "${graph_tc}" \
        ${hold_opt[@]+"${hold_opt[@]}"} \
        -t "${graph_range}" \
        -v MODE="task",PHASE="graph",GRAPH_IDX="${g}",RUN_DIR="${RUN_DIR}",REPO_ROOT="${REPO_ROOT}",COMMON_ARGS_B64="${COMMON_ARGS_B64}",WANDB_PROJECT="${WANDB_PROJECT}",WANDB_ENTITY="${WANDB_ENTITY}",ckpt_root,outputs_root \
        "${WORKER_SH}"
    )"
    gjob="${gjob%%.*}"
    graph_jobid_by_g["${g}"]="${gjob}"
    echo -e "graph\t${gjob}\t${g}\tNA" >> "${JOBS_TSV}"
    echo "[submit_all] graph g=${g} jobid=${gjob}"
  done
else
  if phase_enabled 2 && [[ "${PIPELINE_ALGO}" != "FedDES" ]]; then
    echo "[submit_all] Skipping Phase 2 (GRAPH) because PIPELINE_ALGO=${PIPELINE_ALGO}"
  else
    echo "[submit_all] Skipping Phase 2 (GRAPH) by request. PHASES=${PHASES}"
  fi
fi

# -----------------------------------------------------------------------------
# 4b) Submit Phase 2 (PAE): arrays over p, task_id == base_idx
#     Dependency: pae(p).task(b) waits on base.task(b) via -hold_jid_ad
# -----------------------------------------------------------------------------
if phase_enabled 2 && [[ "${PIPELINE_ALGO}" == "FedPAE" ]]; then
  echo "[submit_all] Submitting PAE grid (FedPAE)."
  for p in $(seq 1 "${n_pae}"); do
    pae_tlist_pending="$(build_tlist_pae_for_p "${p}")"
    [[ -z "${pae_tlist_pending}" ]] && continue

    hold_opt=()
    if [[ -n "${base_jobid}" ]]; then
      hold_opt=(-hold_jid_ad "${base_jobid}")
    fi

    pae_range="1-${n_base}"
    echo "[submit_all] PAE p=${p} tasks pending (by marker): ${pae_tlist_pending}"
    echo "[submit_all] Submitting PAE p=${p} array over full range: ${pae_range}"
    pjob="$(
      qsub -terse -N "pae_${EXP_FAMILY}_p${p}" -cwd -j y \
        -q "${queue}" -pe smp "${smp}" -l ${gpu_req},h_rt=${hrt} \
        -o "${LOG_DIR}/pae_p${p}.\$JOB_ID.\$TASK_ID.log" -tc "${meta_tc}" \
        ${hold_opt[@]+"${hold_opt[@]}"} \
        -t "${pae_range}" \
        -v MODE="task",PHASE="pae",PAE_IDX="${p}",RUN_DIR="${RUN_DIR}",REPO_ROOT="${REPO_ROOT}",COMMON_ARGS_B64="${COMMON_ARGS_B64}",WANDB_PROJECT="${WANDB_PROJECT}",WANDB_ENTITY="${WANDB_ENTITY}",WANDB_MANUAL_RUN=1,ckpt_root,outputs_root \
        "${WORKER_SH}"
    )"
    pjob="${pjob%%.*}"
    echo -e "pae\t${pjob}\tNA\t${p}" >> "${JOBS_TSV}"
    echo "[submit_all] pae p=${p} jobid=${pjob}"
  done
elif [[ "${PIPELINE_ALGO}" == "FedPAE" ]]; then
  echo "[submit_all] Skipping Phase 2 (PAE) by request. PHASES=${PHASES}"
fi

# -----------------------------------------------------------------------------
# 5) Phase 3 PRIMARY: Sweeps as ARRAY JOBS (task-aligned on graph via -hold_jid_ad)
#    - For each g:
#        * Create sweeps for all b = 1..n_base, record sweep_id in sweep_ids_g${g}.txt
#        * Submit one array job with -t 1-n_base and -hold_jid_ad on graph_g
#    - Each task uses MODE=sweep_agent + SWEEP_IDS_FILE + SGE_TASK_ID to pick its sweep.
# -----------------------------------------------------------------------------
if [[ "${use_meta_sweeps}" -eq 1 && "${PIPELINE_ALGO}" == "FedDES" ]]; then
  if phase_enabled 3; then
    echo "[submit_all] Creating sweeps + submitting META agent arrays (PRIMARY)."

    for g in $(seq 1 "${n_graph}"); do
      gjob="${graph_jobid_by_g[${g}]:-}"
      graph_args_line="$(sed -n "${g}p" "${GRAPH_CFG}")"

    SWEEP_IDS_FILE="${RUN_DIR}/sweep_ids_g${g}.txt"
    : > "${SWEEP_IDS_FILE}"

    # 5a) Create sweeps for all base indices b and record their IDs
    for b in $(seq 1 "${n_base}"); do
      base_args_line="$(sed -n "${b}p" "${BASE_CFG}")"

      sweep_file="${RUN_DIR}/sweep_g${g}_b${b}.yaml"

      # Write sweep spec from configs.yaml (gnn_configs defaults + gnn_configs.sweep).
      python - "${CONFIGS_YAML}" "${REPO_ROOT}" "${COMMON_ARGS}" "${base_args_line}" "${graph_args_line}" "${ckpt_root}" \
               "${SWEEP_METHOD}" "${SWEEP_METRIC_NAME}" "${SWEEP_METRIC_GOAL}" "${sweep_file}" "${CONDA_ENV}" << 'PY'
import sys
from pathlib import Path
import yaml

CONFIGS_YAML, REPO_ROOT, common_args, base_args, graph_args, ckpt_root, method, metric, goal, out_path, conda_env = sys.argv[1:]

cfg = yaml.safe_load(Path(CONFIGS_YAML).read_text())
gnn_cfg = cfg["gnn_configs"]

defaults = {k: v for k, v in gnn_cfg.items() if k != "sweep"}
sweep_cfg = gnn_cfg.get("sweep") or {}

params = {k: {"value": v} for k, v in defaults.items()}
for k, v in sweep_cfg.items():
    params[k] = {"values": v} if isinstance(v, list) else {"value": v}

command_str = " ".join([
    'eval "$(${HOME}/micromamba/bin/micromamba shell hook --shell bash)"',
    "&&",
    f"micromamba activate {conda_env}",
    "&&",
    f"cd {REPO_ROOT}/system",
    "&&",
    "python main.py",
    common_args,
    base_args,
    graph_args,
    "--phase", "3",
    "--ckpt_root", ckpt_root,
    "--outputs_root", os.environ.get("outputs_root", ckpt_root),
])

spec = {
    "method": method,
    "metric": {"name": metric, "goal": goal},
    "parameters": params,
    "command": ["bash", "-lc", command_str],
}

Path(out_path).write_text(yaml.safe_dump(spec, sort_keys=False))
PY

      # --- Create sweep NOW (Option B) ---
      set +e
      sweep_out="$(${WANDB_BIN} sweep --entity "${WANDB_ENTITY}" --project "${WANDB_PROJECT}" "${sweep_file}" 2>&1)"
      sweep_rc=$?
      set -e

      echo "[submit_all] wandb sweep rc=${sweep_rc} for family=${EXP_FAMILY}, g=${g}, b=${b}" \
        | tee -a "${RUN_DIR}/wandb_sweep_debug.log"
      echo "${sweep_out}" >> "${RUN_DIR}/wandb_sweep_debug.log"

      if [[ ${sweep_rc} -ne 0 ]]; then
        echo "[submit_all] ERROR: wandb sweep failed for g=${g} b=${b} (family=${EXP_FAMILY}). See ${RUN_DIR}/wandb_sweep_debug.log"
        exit 2
      fi

      # Extract sweep id robustly from output:
      sweep_id="$(echo "${sweep_out}" | grep -Eo '([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9]+)' | tail -n 1 || true)"
      if [[ -z "${sweep_id}" ]]; then
        sid="$(echo "${sweep_out}" | grep -Eo '/sweeps/[A-Za-z0-9]+' | tail -n 1 | sed 's|/sweeps/||' || true)"
        if [[ -n "${sid}" ]]; then
          sweep_id="${WANDB_ENTITY}/${WANDB_PROJECT}/${sid}"
        fi
      fi
      if [[ -z "${sweep_id}" ]]; then
        echo "[submit_all] ERROR: Could not parse sweep id for g=${g} b=${b} (family=${EXP_FAMILY}). See ${RUN_DIR}/wandb_sweep_debug.log"
        exit 2
      fi

      echo "${sweep_id}" >> "${SWEEP_IDS_FILE}"
      echo -e "${g}\t${b}\t${sweep_id}\t${sweep_file}" >> "${SWEEPS_TSV}"
      echo "[submit_all] sweep created: g=${g} b=${b} sweep_id=${sweep_id}"
    done

    # 5b) Submit ONE sweep agent ARRAY per g, task-aligned to graph_g via -hold_jid_ad
    hold_args=()
    if [[ -n "${gjob}" ]]; then
      hold_args=(-hold_jid_ad "${gjob}")
    fi

    meta_range="1-${n_base}"
    sweep_job_name="des_meta_sweep_${EXP_FAMILY}_g${g}"

    mjob="$(
      qsub -terse -N "${sweep_job_name}" -cwd -j y \
        -q "${queue}" -pe smp "${smp}" -l ${meta_gpu_req},h_rt=${hrt} \
        -o "${LOG_DIR}/meta_sweep_g${g}.\$JOB_ID.\$TASK_ID.log" \
        ${hold_args[@]+"${hold_args[@]}"} \
        -t "${meta_range}" \
        -v MODE="sweep_agent",REPO_ROOT="${REPO_ROOT}",RUN_DIR="${RUN_DIR}",G_IDX="${g}",WANDB_BIN="${WANDB_BIN}",WANDB_PROJECT="${WANDB_PROJECT}",WANDB_ENTITY="${WANDB_ENTITY}",SWEEP_MAX_RUNS="${SWEEP_MAX_RUNS}",SWEEP_IDS_FILE="${SWEEP_IDS_FILE}",ckpt_root \
        "${WORKER_SH}"
    )"
    mjob="${mjob%%.*}"
    echo -e "meta_sweep\t${mjob}\t${g}\tALL" >> "${JOBS_TSV}"
    echo "[submit_all] meta sweep array for g=${g} jobid=${mjob}"
  done
  else
    echo "[submit_all] Skipping Phase 3 meta sweeps because PHASES=${PHASES} does not include 3."
  fi
fi

# -----------------------------------------------------------------------------
# 6) Optional: Phase 3 meta grid (non-sweep)
#    Arrays over (g,k), task_id == base_idx, held task-aligned on graph array via -hold_jid_ad
# -----------------------------------------------------------------------------
meta_grid_jobids=()

wait_for_meta_slot() {
  local max="${MAX_META_JOBS}"
  [[ "${max}" -le 0 ]] && return
  local interval="${META_CHECK_INTERVAL}"
  while true; do
    local running=0
    local keep=()
    for jobid in "${meta_grid_jobids[@]:-}"; do
      if qstat -j "${jobid}" >/dev/null 2>&1; then
        keep+=("${jobid}")
        running=$((running+1))
      fi
    done
    meta_grid_jobids=("${keep[@]}")
    if [[ "${running}" -lt "${max}" ]]; then
      break
    fi
    echo "[submit_all] MAX_META_JOBS=${max} reached (${running} queued/running); sleeping ${interval}s"
    sleep "${interval}"
  done
}

if [[ "${use_meta_grid}" -eq 1 && "${PIPELINE_ALGO}" == "FedDES" ]]; then
  if phase_enabled 3; then
    echo "[submit_all] Submitting META grid (OPTIONAL)."
    echo "[submit_all] MAX_META_JOBS=${MAX_META_JOBS} (0=unlimited), polling every ${META_CHECK_INTERVAL}s"

    for g in $(seq 1 "${n_graph}"); do
      gjob="${graph_jobid_by_g[${g}]:-}"

      for k in $(seq 1 "${n_gnn}"); do
        meta_tlist_pending="$(build_tlist_meta_grid_for_gk "${g}" "${k}")"
        [[ -z "${meta_tlist_pending}" ]] && continue

        hold_opt=()
        if [[ -n "${gjob}" ]]; then
          hold_opt=(-hold_jid_ad "${gjob}")
        fi

        meta_range="1-${n_base}"
        echo "[submit_all] META grid g=${g} k=${k} tasks pending (by marker): ${meta_tlist_pending}"
        echo "[submit_all] Submitting META grid g=${g} k=${k} array over full range: ${meta_range}"

        wait_for_meta_slot
        mgrid_job="$(
          qsub -terse -N "des_meta_${EXP_FAMILY}_g${g}_k${k}" -cwd -j y \
            -q "${queue}" -pe smp "${smp}" -l ${meta_gpu_req},h_rt=${hrt} \
            -o "${LOG_DIR}/meta_g${g}_k${k}.\$JOB_ID.\$TASK_ID.log" -tc "${meta_tc}" \
            ${hold_opt[@]+"${hold_opt[@]}"} \
            -t "${meta_range}" \
            -v MODE="task",PHASE="meta_grid",GRAPH_IDX="${g}",GNN_IDX="${k}",RUN_DIR="${RUN_DIR}",REPO_ROOT="${REPO_ROOT}",COMMON_ARGS_B64="${COMMON_ARGS_B64}",WANDB_PROJECT="${WANDB_PROJECT}",WANDB_ENTITY="${WANDB_ENTITY}",WANDB_MANUAL_RUN=1,ckpt_root,outputs_root \
            "${WORKER_SH}"
        )"
      mgrid_job="${mgrid_job%%.*}"
      meta_grid_jobids+=("${mgrid_job}")
      echo -e "meta_grid\t${mgrid_job}\t${g}\t${k}" >> "${JOBS_TSV}"
      done
    done
  else
    echo "[submit_all] Skipping Phase 3 meta grid because PHASES=${PHASES} does not include 3."
  fi
fi

echo "[submit_all] Done."
echo "[submit_all] Logs: ${LOG_DIR}"
echo "[submit_all] Job list: ${JOBS_TSV}"
echo "[submit_all] Sweep list: ${SWEEPS_TSV}"
echo "[submit_all] Cancel this run:"
echo "  ACTION=cancel RUN_DIR=${RUN_DIR} bash run_pipeline.sh"

#!/bin/bash
# Run lrgs.py over all HOD catalogs as a pool of independent processes.
#
#   bash run_lrgs_parallel.sh --status          # how much is already done
#   bash run_lrgs_parallel.sh --dry-run         # show the plan, launch nothing
#   bash run_lrgs_parallel.sh                   # run
#   NWORKERS=64 BATCH_SIZE=50 bash run_lrgs_parallel.sh
#
# WHY A PROCESS POOL AND NOT srun/mpirun
# --------------------------------------
# lrgs.py calls CurrentMPIComm.get(), so launching it as `srun -n 64 python
# lrgs.py` would make nbodykit treat those 64 ranks as ONE distributed
# computation -- which is the thing that does not parallelise well here.
# Launching 64 *separate* python processes instead gives each its own
# singleton MPI_COMM_WORLD, so they are fully independent. The work is
# embarrassingly parallel across catalogs, so this is the right axis.
#
# Consequence worth knowing: the ENSEMBLE does parallelise across nodes even
# though a single catalog does not. See slurm/perlmutter_lrgs_array.sbatch,
# which runs one of these pools per node.
#
# SAFETY / RESUMABILITY
# ---------------------
# lrgs.py skips any batch entry whose output file already exists, so:
#   * re-running after a crash, timeout or cancellation resumes where it left
#     off -- nothing is recomputed;
#   * two pools racing on the same batch waste CPU but cannot corrupt output
#     (worst case both write the same file);
#   * --status / the end-of-run summary count finished outputs directly.
#
# Each worker is single-threaded (OMP_NUM_THREADS=1 etc.) on purpose: with N
# processes already saturating the node, per-process threading only causes
# oversubscription.

set -uo pipefail

# ---------------------------------------------------------------- config ---
SCRIPT="${SCRIPT:-lrgs.py}"
BATCH_SIZE="${BATCH_SIZE:-50}"
CATDIR="${CATDIR:-/pscratch/sd/j/jsull/lrg_hods/catalogs}"
RESULTSDIR="${RESULTSDIR:-/pscratch/sd/j/jsull/qso_sbp/Heavy_files_Andrej/AbacusBase/results}"
RESULT_TAG="${RESULT_TAG:-AbacusBase}"   # output filename tag: AbacusBase | AbacusSmall
LOGDIR="${LOGDIR:-logs_lrgs}"
CONDA_ENV="${CONDA_ENV:-nbodykit_env_25}"
MEM_PER_WORKER_GB="${MEM_PER_WORKER_GB:-8}"
SHARD="${SHARD:-0}"          # this pool's index among NSHARDS pools (multi-node)
NSHARDS="${NSHARDS:-1}"
NCAT="${NCAT:-}"             # number of catalogs; auto-detected from CATDIR

STATUS_ONLY=0; DRY_RUN=0; RETRY_FAILED=0
for a in "$@"; do case "$a" in
    --status)        STATUS_ONLY=1 ;;
    --dry-run|-n)    DRY_RUN=1 ;;
    --retry-failed)  RETRY_FAILED=1 ;;
    -h|--help)       sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
esac; done

# ------------------------------------------------------------- env setup ---
if [[ -f slurm/env.local.sh ]]; then
    # shellcheck disable=SC1091
    source slurm/env.local.sh
fi
if [[ "${ENV_READY:-0}" != "1" ]]; then
    if [[ -z "${CONDA_BASE:-}" ]]; then
        if [[ -n "${CONDA_EXE:-}" ]]; then
            CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
        elif command -v conda >/dev/null 2>&1; then
            CONDA_BASE="$(conda info --base 2>/dev/null || true)"
        fi
    fi
    if [[ -z "${CONDA_BASE:-}" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        echo "ERROR: cannot locate conda (CONDA_BASE='${CONDA_BASE:-unset}')." >&2
        echo "       Set it in slurm/env.local.sh or export CONDA_BASE=\$(conda info --base)" >&2
        exit 1
    fi
    set +u; source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || { set -u; echo "ERROR: cannot activate '$CONDA_ENV'" >&2; exit 1; }
    set -u
fi

# One thread per worker -- N processes already fill the node.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# ------------------------------------------------------------ discovery ---
[[ -f "$SCRIPT" ]] || { echo "ERROR: $SCRIPT not found (cd to its directory, or set SCRIPT=)" >&2; exit 1; }
[[ -d "$CATDIR" ]] || { echo "ERROR: catalog dir not found: $CATDIR" >&2; exit 1; }

if [[ -z "$NCAT" ]]; then
    NCAT=$(find "$CATDIR" -maxdepth 1 -name '*.npy' | wc -l)
fi
(( NCAT > 0 )) || { echo "ERROR: no .npy catalogs in $CATDIR" >&2; exit 1; }

NBATCH=$(( (NCAT + BATCH_SIZE - 1) / BATCH_SIZE ))
REMAINDER=$(( NCAT % BATCH_SIZE ))

status() {
    local nr nd
    nr=$(find "$RESULTSDIR" -maxdepth 1 -name "results_${RESULT_TAG}_*.npy" 2>/dev/null | wc -l)
    nd=$(find "$RESULTSDIR" -maxdepth 1 -name "results_RSD_${RESULT_TAG}_*.npy" 2>/dev/null | wc -l)
    printf '  real-space outputs : %6d / %d\n' "$nr" "$NCAT"
    printf '  RSD outputs        : %6d / %d\n' "$nd" "$NCAT"
}

echo "=================================================================="
echo " catalogs      : $NCAT in $CATDIR"
echo " batches       : $NBATCH of $BATCH_SIZE"
echo " script        : $SCRIPT   (result tag: $RESULT_TAG)"
echo " results dir   : $RESULTSDIR"
status
echo "=================================================================="
[[ $STATUS_ONLY == 1 ]] && exit 0

if (( REMAINDER != 0 )); then
    echo "WARNING: $BATCH_SIZE does not divide $NCAT evenly; the last batch runs past the"
    echo "         final catalog and will raise FileNotFoundError after doing its real work."
    echo "         Either accept one noisy log, apply lrgs_hoist.diff (which clamps the"
    echo "         range), or pick a divisor -- e.g. 50, 70, 75, 105, 150 for 10500."
    echo
fi

# --------------------------------------------------------------- workers ---
if [[ -z "${NWORKERS:-}" ]]; then
    cores="${SLURM_CPUS_ON_NODE:-$(nproc)}"
    # Perlmutter reports logical CPUs; one worker per physical core at most.
    tpc=$(lscpu 2>/dev/null | awk -F: '/Thread\(s\) per core/{gsub(/ /,"",$2); print $2}')
    [[ -n "${tpc:-}" && "$tpc" -gt 1 ]] && cores=$(( cores / tpc ))
    memgb=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
    by_mem=$(( memgb / MEM_PER_WORKER_GB ))
    NWORKERS=$(( cores < by_mem ? cores : by_mem ))
    (( NWORKERS < 1 )) && NWORKERS=1
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        NWORKERS=4
        echo "NOTE: not inside a SLURM allocation -- this looks like a login node."
        echo "      Defaulting to 4 workers. Login nodes are shared; for the full run use"
        echo "      an allocation (salloc / sbatch slurm/perlmutter_lrgs_array.sbatch)."
        echo
    fi
fi

# ------------------------------------------------------- batch selection ---
# Stride-based sharding so each pool gets an interleaved (statistically
# similar) slice of the index range rather than a contiguous block.
BATCHES=()
for (( b = SHARD; b < NBATCH; b += NSHARDS )); do BATCHES+=("$b"); done

if [[ $RETRY_FAILED == 1 ]]; then
    [[ -s "$LOGDIR/FAILED" ]] || { echo "no $LOGDIR/FAILED to retry"; exit 0; }
    mapfile -t BATCHES < <(awk '{print $3}' "$LOGDIR/FAILED" | sort -un)
    echo "retrying ${#BATCHES[@]} previously failed batches"
fi

echo " workers       : $NWORKERS   (mem/worker ${MEM_PER_WORKER_GB}G, threads 1)"
echo " this pool     : shard $SHARD of $NSHARDS -> ${#BATCHES[@]} batches"
echo " logs          : $LOGDIR/batch_<n>.log"
echo "=================================================================="

if [[ $DRY_RUN == 1 ]]; then
    echo "batches: ${BATCHES[*]}"
    echo "(dry run: nothing launched)"
    exit 0
fi

mkdir -p "$LOGDIR" "$RESULTSDIR"
rm -f "$LOGDIR/FAILED"

# --------------------------------------------------------------- the pool ---
export SCRIPT BATCH_SIZE LOGDIR
run_one() {
    # NB: separate `local` statements. A single `local b=$1 log=...${b}...`
    # expands every argument BEFORE assigning any, so ${b} would be empty and
    # all workers would share one log file, clobbering each other.
    local b="$1"
    local log="$LOGDIR/batch_${b}.log"
    local t0=$SECONDS
    if python -u "$SCRIPT" --batch_size "$BATCH_SIZE" --batch_num "$b" > "$log" 2>&1; then
        printf '[%s] batch %-5s ok      %5ds\n' "$(date +%H:%M:%S)" "$b" "$((SECONDS-t0))"
    else
        printf '[%s] batch %-5s FAILED  %5ds  (%s)\n' "$(date +%H:%M:%S)" "$b" "$((SECONDS-t0))" "$log"
        printf 'FAILED batch %s %s\n' "$b" "$log" >> "$LOGDIR/FAILED"
        return 1
    fi
}
export -f run_one

T0=$SECONDS
printf '%s\n' "${BATCHES[@]}" | xargs -P "$NWORKERS" -I{} bash -c 'run_one {}'
RC=$?

echo "=================================================================="
echo " wall time     : $(( (SECONDS-T0)/60 )) min $(( (SECONDS-T0)%60 )) s"
status
if [[ -s "$LOGDIR/FAILED" ]]; then
    echo " FAILED batches: $(wc -l < "$LOGDIR/FAILED")  (see $LOGDIR/FAILED)"
    echo "   retry with:  bash $0 --retry-failed"
    exit 1
fi
echo " all batches completed"
exit 0

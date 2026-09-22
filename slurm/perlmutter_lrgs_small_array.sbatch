#!/bin/bash
# Same as perlmutter_lrgs_array.sbatch but for the SMALL box (500 Mpc/h):
# runs small_lrgs.py over $PSCRATCH/lrg_hods_small/catalogs.
#
#   sbatch slurm/perlmutter_lrgs_small_array.sbatch
#
# Output goes to .../Heavy_files_Andrej/AbacusSmall/results, never the base
# box's directory, and the result files are tagged AbacusSmall so --status
# counts the right ones.
#
# Note on cost: the small box has ~64x fewer galaxies but the SAME
# Nmesh = 256 grid, so the FFT work per catalog is identical to the base
# box -- only painting and catalog I/O get cheaper. Expect roughly 60-70%
# of the base-box time per catalog, not 1/64.
#
# Memory is much lighter though (catalogs are ~6 MB instead of up to 3 GB;
# per-worker usage is dominated by the ~1.35 GB of shifted fields), so
# MEM_PER_WORKER_GB is lowered to 4, which lets the runner use more workers
# per node.
#
#SBATCH -J lrgs_small
# Uncomment and set if you have more than one NERSC allocation; with a
# single default allocation SLURM picks it automatically.
##SBATCH -A <your_allocation>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256
#SBATCH -t 00:30:00
#SBATCH --array=0-15
#SBATCH -o lrgs_small_%A_%a.out

RUNDIR="${RUNDIR:-$SLURM_SUBMIT_DIR}"
cd "$RUNDIR" || { echo "ERROR: cannot cd to $RUNDIR" >&2; exit 1; }

if [[ ! -f run_lrgs_parallel.sh ]]; then
    echo "ERROR: run_lrgs_parallel.sh not found in $RUNDIR" >&2
    echo "       Submit from the directory holding small_lrgs.py +" >&2
    echo "       run_lrgs_parallel.sh, or pass --export=ALL,RUNDIR=/that/dir" >&2
    exit 1
fi

export SCRIPT="${SCRIPT:-small_lrgs.py}"
export CATDIR="${CATDIR:-$PSCRATCH/lrg_hods_small/catalogs}"
export RESULTSDIR="${RESULTSDIR:-$PSCRATCH/qso_sbp/Heavy_files_Andrej/AbacusSmall/results}"
export RESULT_TAG="${RESULT_TAG:-AbacusSmall}"
export MEM_PER_WORKER_GB="${MEM_PER_WORKER_GB:-4}"

export SHARD="$SLURM_ARRAY_TASK_ID"
export NSHARDS="$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))"
export LOGDIR="${LOGDIR:-logs_lrgs_small/task_${SLURM_ARRAY_TASK_ID}}"

echo "node $(hostname)  shard $SHARD / $NSHARDS  (small box)"
bash run_lrgs_parallel.sh

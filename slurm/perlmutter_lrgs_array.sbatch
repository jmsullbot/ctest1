#!/bin/bash
# Run lrgs.py over all HOD catalogs: one process pool per node, N nodes.
#
#   sbatch slurm/perlmutter_lrgs_array.sbatch
#
# A single catalog's nbodykit computation cannot span nodes, but the 10500
# catalogs are independent, so the ENSEMBLE parallelises perfectly: each
# array task owns one node and runs run_lrgs_parallel.sh there, which in
# turn runs ~64 independent single-rank python processes.
#
# Batches are assigned by stride (task a takes batches a, a+A, a+2A, ...),
# so every node gets a statistically similar slice.
#
# Resumable: lrgs.py skips outputs that already exist, so if the job times
# out or you cancel it, just resubmit -- finished catalogs are not redone.
#
# Tuning:
#   --array=0-7        number of nodes (adjust to taste / queue limits)
#   NWORKERS           processes per node (default: min(physical cores,
#                      mem/8GB) ~= 64 on a 512 GB Perlmutter CPU node)
#   MEM_PER_WORKER_GB  raise if you see OOM; the biggest catalogs are ~3 GB
#                      on disk and peak ~6-8 GB while being painted
#
# Override any of those at submit time:
#   sbatch --export=ALL,NWORKERS=48,BATCH_SIZE=50 slurm/perlmutter_lrgs_array.sbatch
#
#SBATCH -J lrgs
# Uncomment and set if you have more than one NERSC allocation; with a
# single default allocation SLURM picks it automatically.
##SBATCH -A <your_allocation>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256
#SBATCH -t 06:00:00
#SBATCH --array=0-7
#SBATCH -o lrgs_%A_%a.out

# Directory holding lrgs.py (and slurm/env.local.sh for the conda settings).
RUNDIR="${RUNDIR:-$SLURM_SUBMIT_DIR}"
cd "$RUNDIR" || { echo "ERROR: cannot cd to $RUNDIR" >&2; exit 1; }

if [[ ! -f run_lrgs_parallel.sh ]]; then
    echo "ERROR: run_lrgs_parallel.sh not found in $RUNDIR" >&2
    echo "       Submit from the directory holding lrgs.py + run_lrgs_parallel.sh," >&2
    echo "       or pass  --export=ALL,RUNDIR=/path/to/that/dir" >&2
    exit 1
fi

export SHARD="$SLURM_ARRAY_TASK_ID"
export NSHARDS="$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))"
export LOGDIR="${LOGDIR:-logs_lrgs/task_${SLURM_ARRAY_TASK_ID}}"

echo "node $(hostname)  shard $SHARD / $NSHARDS"
bash run_lrgs_parallel.sh

#!/bin/bash
# LRG HOD generation as a SLURM job array — the recommended way to run.
#
# The 10500 table rows are split into NSHARDS disjoint chunks; each array
# task processes one chunk on its own node (~20-40 min for 16 shards at
# the eq. (74) number densities on the base box).  Short walltime requests
# backfill quickly in the Perlmutter queue.
#
# Tasks never conflict: per-run .npy catalogs (which hold ALL galaxy data)
# are named by global row index, and each task writes its own small
# metadata-only lrg_hods_rowsSTART-END.hdf5.
#
# Prerequisites:
#   bash slurm/setup_env.sh                        (once, on a login node)
#   sbatch slurm/perlmutter_prepare_sim.sbatch     (once per box+redshift)
#
# Submit with:   sbatch slurm/perlmutter_lrg_hods_array.sbatch
#
# Can be submitted from the repo root or from slurm/; to submit from
# anywhere else, pass the repo path explicitly:
#   sbatch --export=ALL,REPO_DIR=/path/to/ctest1 slurm/perlmutter_lrg_hods_array.sbatch
#
#SBATCH -J lrg_hods
# Uncomment and set if you have more than one NERSC allocation; with a
# single default allocation SLURM picks it automatically.
##SBATCH -A <your_allocation>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256               # whole CPU node: 128 physical cores x 2 HT
#SBATCH -t 01:00:00
#SBATCH --array=0-15
#SBATCH -o lrg_hods_%A_%a.out

# Resolves REPO_DIR (cd's there) and activates $CONDA_ENV, or exits with
# setup instructions.
if   [[ -f "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh"
elif [[ -f "${SLURM_SUBMIT_DIR:-$PWD}/_common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR:-$PWD}/_common.sh"
elif [[ -n "${REPO_DIR:-}" && -f "$REPO_DIR/slurm/_common.sh" ]]; then
    source "$REPO_DIR/slurm/_common.sh"
else
    echo "ERROR: cannot locate slurm/_common.sh. Submit from the repo root," >&2
    echo "       or use: sbatch --export=ALL,REPO_DIR=/path/to/ctest1 ..." >&2
    exit 1
fi

OUTDIR="${OUTDIR:-$PSCRATCH/lrg_hods}"
mkdir -p "$OUTDIR"

NSHARDS=16                   # must match the --array range above

python generate_lrg_hods.py \
    --path2config config/lrg_hod.yaml \
    --params      lrg_params_eq74.npy \
    --output      "$OUTDIR/lrg_hods.hdf5" \
    --Nthread     128 \
    --shard       "$SLURM_ARRAY_TASK_ID" \
    --nshards     "$NSHARDS"

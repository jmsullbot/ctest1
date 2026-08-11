#!/bin/bash
# NERSC Perlmutter batch script for LRG HOD generation — SERIAL fallback.
#
# Prefer slurm/perlmutter_lrg_hods_array.sbatch: the task is embarrassingly
# parallel across table rows, and the job array finishes in well under an
# hour of wall-clock instead of ~5-8 h (same node-hours charged either way).
#
# Prerequisite (once):  bash slurm/setup_env.sh
# First run for a given box + redshift: set EXTRA_FLAGS="--prepare_sim"
# below so the halo subsamples get built before the HOD loop (they land in
# subsample_dir and are reused afterwards).
#
# Submit with:   sbatch slurm/perlmutter_lrg_hods.sbatch
#
#SBATCH -J lrg_hods
# Uncomment and set if you have more than one NERSC allocation; with a
# single default allocation SLURM picks it automatically.
##SBATCH -A <your_allocation>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256               # whole CPU node: 128 physical cores x 2 HT
#SBATCH -t 12:00:00
#SBATCH -o lrg_hods_%j.out

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

EXTRA_FLAGS=""               # set to "--prepare_sim" on the first run only

python generate_lrg_hods.py \
    --path2config config/lrg_hod.yaml \
    --params      lrg_params_eq74.npy \
    --output      "$OUTDIR/lrg_hods.hdf5" \
    --Nthread     128 \
    $EXTRA_FLAGS

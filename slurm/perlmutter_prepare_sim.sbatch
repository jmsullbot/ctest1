#!/bin/bash
# One-time prepare_sim job — run this ONCE per box + redshift, and wait for
# it to finish before launching the job array (all array tasks read the
# subsamples it writes to subsample_dir).
#
# Prerequisite (once):   bash slurm/setup_env.sh
# Submit with:           sbatch slurm/perlmutter_prepare_sim.sbatch
#
# Can be submitted from the repo root or from slurm/; to submit from
# anywhere else, pass the repo path explicitly:
#   sbatch --export=ALL,REPO_DIR=/path/to/ctest1 slurm/perlmutter_prepare_sim.sbatch
#
#SBATCH -J prep_lrg
#SBATCH -A desi              # your NERSC allocation account
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256
#SBATCH -t 02:00:00          # typically finishes in 30-90 min (I/O-bound)
#SBATCH -o prep_lrg_%j.out

# Resolves REPO_DIR (cd's there) and activates $CONDA_ENV, or exits with
# setup instructions.  Note: sourced by path relative to the submit dir,
# with a fallback for submission from inside slurm/.
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

python generate_lrg_hods.py --prepare_only --path2config config/lrg_hod.yaml

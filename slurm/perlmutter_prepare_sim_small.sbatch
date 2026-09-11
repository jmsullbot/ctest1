#!/bin/bash
# One-time prepare_sim for the SMALL box (500 Mpc/h) — run ONCE per phase +
# redshift, and wait for it to finish before launching the small-box array.
#
# Subsamples land in subsample_dir/<sim_name>/z0.500/ (see
# config/lrg_hod_small.yaml), so they never collide with the base-box run.
#
# Prerequisite (once):   bash slurm/setup_env.sh
# Submit with:           sbatch slurm/perlmutter_prepare_sim_small.sbatch
#
# Another phase:  sbatch --export=ALL,CONFIG=config/my_other_phase.yaml \
#                        slurm/perlmutter_prepare_sim_small.sbatch
#
#SBATCH -J prep_lrg_small
# Uncomment and set if you have more than one NERSC allocation; with a
# single default allocation SLURM picks it automatically.
##SBATCH -A <your_allocation>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256
#SBATCH -t 00:30:00          # 64x less data than the base box; minutes, not an hour
#SBATCH -o prep_lrg_small_%j.out

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

CONFIG="${CONFIG:-config/lrg_hod_small.yaml}"

python generate_lrg_hods.py --prepare_only --path2config "$CONFIG"

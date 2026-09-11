#!/bin/bash
# LRG HOD generation on an AbacusSummit SMALL box (500 Mpc/h) — job array.
#
# Same 10500-row parameter table as the base-box run, so catalog NNNNNN here
# matches catalog NNNNNN from $PSCRATCH/lrg_hods row-for-row.  Output goes to
# a SEPARATE directory ($PSCRATCH/lrg_hods_small) so nothing overwrites the
# base-box catalogs.
#
# The small box has 64x fewer halos and galaxies, so each HOD run takes tens
# of ms rather than seconds: 4 shards of ~2600 rows finish in roughly 5-15
# min each.  (A single serial job would also be fine; the array just keeps
# the walltime request short so it backfills fast.)
#
# Prerequisites:
#   bash slurm/setup_env.sh                              (once, on a login node)
#   sbatch slurm/perlmutter_prepare_sim_small.sbatch     (once per phase+redshift)
#
# Submit with:   sbatch slurm/perlmutter_lrg_hods_small_array.sbatch
#
# Another phase (after preparing it):
#   sbatch --export=ALL,CONFIG=config/my_other_phase.yaml,OUTDIR=$PSCRATCH/lrg_hods_small_ph3001 \
#          slurm/perlmutter_lrg_hods_small_array.sbatch
#
#SBATCH -J lrg_hods_small
# Uncomment and set if you have more than one NERSC allocation; with a
# single default allocation SLURM picks it automatically.
##SBATCH -A <your_allocation>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256               # whole CPU node: 128 physical cores x 2 HT
#SBATCH -t 00:30:00
#SBATCH --array=0-3
#SBATCH -o lrg_hods_small_%A_%a.out

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
OUTDIR="${OUTDIR:-$PSCRATCH/lrg_hods_small}"
mkdir -p "$OUTDIR"

NSHARDS=4                    # must match the --array range above

python generate_lrg_hods.py \
    --path2config "$CONFIG" \
    --params      lrg_params_eq74.npy \
    --output      "$OUTDIR/lrg_hods_small.hdf5" \
    --Nthread     128 \
    --shard       "$SLURM_ARRAY_TASK_ID" \
    --nshards     "$NSHARDS"

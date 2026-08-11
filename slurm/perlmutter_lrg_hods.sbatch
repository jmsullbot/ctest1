#!/bin/bash
# NERSC Perlmutter batch script for LRG HOD generation.
#
# First-ever run for a given box + redshift: set EXTRA_FLAGS="--prepare_sim"
# below so the halo subsamples get built before the HOD loop
# (prepare_sim output lands in subsample_dir and is reused afterwards).
#
# Submit with:   sbatch slurm/perlmutter_lrg_hods.sbatch
#
#SBATCH -J lrg_hods
#SBATCH -A desi              # your NERSC allocation account
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256               # whole CPU node: 128 physical cores x 2 HT
#SBATCH -t 12:00:00
#SBATCH -o lrg_hods_%j.out

module load conda
conda activate abacus

cd $SLURM_SUBMIT_DIR

OUTDIR=$PSCRATCH/lrg_hods
mkdir -p $OUTDIR

EXTRA_FLAGS=""               # set to "--prepare_sim" on the first run only

python generate_lrg_hods.py \
    --path2config config/lrg_hod.yaml \
    --params      lrg_params_eq74.npy \
    --output      $OUTDIR/lrg_hods.hdf5 \
    --Nthread     128 \
    $EXTRA_FLAGS

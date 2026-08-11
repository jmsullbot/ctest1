#!/bin/bash
# One-time prepare_sim job — run this ONCE per box + redshift, and wait for
# it to finish before launching the job array (all array tasks read the
# subsamples it writes to subsample_dir).
#
# Submit with:   sbatch slurm/perlmutter_prepare_sim.sbatch
#
#SBATCH -J prep_lrg
#SBATCH -A desi              # your NERSC allocation account
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256
#SBATCH -t 02:00:00        # typically finishes in 30-90 min (I/O-bound)
#SBATCH -o prep_lrg_%j.out

module load conda
conda activate abacus

cd $SLURM_SUBMIT_DIR

python generate_lrg_hods.py --prepare_only --path2config config/lrg_hod.yaml

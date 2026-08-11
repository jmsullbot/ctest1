# Shared preamble for the Perlmutter sbatch scripts.  Sourced, not executed.
#
# Responsibilities:
#   1. Locate the repo root regardless of where sbatch was invoked from
#      (repo root, slurm/, or anywhere else via REPO_DIR=...).
#   2. Activate the conda environment, failing early and loudly with setup
#      instructions if it does not exist.
#
# Overridable from the submit line, e.g.:
#   sbatch --export=ALL,REPO_DIR=/path/to/ctest1,CONDA_ENV=myenv <script>

set -euo pipefail

# --- 1. Locate the repo root -----------------------------------------------
# SLURM_SUBMIT_DIR is the directory sbatch was run from.  We can't rely on
# $0 because SLURM copies the batch script into a spool directory.
_anchor="generate_lrg_hods.py"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

if [[ ! -f "$REPO_DIR/$_anchor" && -f "$REPO_DIR/../$_anchor" ]]; then
    # submitted from inside slurm/ — step up one level
    REPO_DIR="$(cd "$REPO_DIR/.." && pwd)"
fi

if [[ ! -f "$REPO_DIR/$_anchor" ]]; then
    echo "ERROR: cannot find $_anchor (looked in '$REPO_DIR')." >&2
    echo "       Submit from the repo root:" >&2
    echo "         cd /path/to/ctest1 && sbatch slurm/$(basename "${BASH_SOURCE[1]:-<script>}")" >&2
    echo "       ...or point REPO_DIR at it explicitly:" >&2
    echo "         sbatch --export=ALL,REPO_DIR=/path/to/ctest1 slurm/<script>" >&2
    exit 1
fi

cd "$REPO_DIR"
echo "Repo dir   : $REPO_DIR"

# --- 2. Activate the conda environment -------------------------------------
CONDA_ENV="${CONDA_ENV:-abacus}"

module load conda

if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$CONDA_ENV"; then
    echo "ERROR: conda environment '$CONDA_ENV' does not exist." >&2
    echo "" >&2
    echo "       Create it ONCE on a login node:" >&2
    echo "         cd $REPO_DIR" >&2
    echo "         bash slurm/setup_env.sh" >&2
    echo "" >&2
    echo "       (or by hand:)" >&2
    echo "         module load conda" >&2
    echo "         conda create -n $CONDA_ENV python=3.10 -y" >&2
    echo "         conda activate $CONDA_ENV" >&2
    echo "         pip install -r requirements.txt" >&2
    exit 1
fi

conda activate "$CONDA_ENV"
echo "Conda env  : $CONDA_ENV  ($(python -c 'import sys; print(sys.executable)'))"

# Fail early if the dependencies were never installed into the env
if ! python -c "import numpy, h5py, yaml" 2>/dev/null; then
    echo "ERROR: environment '$CONDA_ENV' is missing dependencies." >&2
    echo "       conda activate $CONDA_ENV && pip install -r $REPO_DIR/requirements.txt" >&2
    exit 1
fi

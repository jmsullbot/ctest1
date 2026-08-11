#!/bin/bash
# One-time environment setup for NERSC Perlmutter.  Run on a LOGIN node
# (not via sbatch) before submitting any jobs:
#
#     cd /path/to/ctest1
#     bash slurm/setup_env.sh
#
# Override the environment name with:  CONDA_ENV=myenv bash slurm/setup_env.sh

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-abacus}"
PYVER="${PYVER:-3.10}"

# Resolve the repo root from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

echo "Repo dir    : $REPO_DIR"
echo "Environment : $CONDA_ENV  (python $PYVER)"
echo

module load conda

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$CONDA_ENV"; then
    echo "Environment '$CONDA_ENV' already exists — reusing it."
else
    echo "Creating conda environment '$CONDA_ENV'…"
    conda create -n "$CONDA_ENV" "python=$PYVER" -y
fi

# 'conda activate' needs the shell hook in a non-interactive script
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo
echo "Installing dependencies from requirements.txt…"
pip install -r requirements.txt

echo
echo "Verifying…"
python - <<'PY'
import importlib, sys
print(f"  python      {sys.version.split()[0]}  ({sys.executable})")
for mod in ("numpy", "scipy", "h5py", "yaml", "abacusnbody"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:11s} {getattr(m, '__version__', 'ok')}")
    except ImportError as exc:
        print(f"  {mod:11s} MISSING — {exc}")
        sys.exit(1)
PY

echo
echo "Done.  Now submit jobs FROM THE REPO ROOT ($REPO_DIR):"
echo "    sbatch slurm/perlmutter_prepare_sim.sbatch"
echo "    sbatch slurm/perlmutter_lrg_hods_array.sbatch"

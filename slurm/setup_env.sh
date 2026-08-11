#!/bin/bash
# One-time environment setup.  Run on a LOGIN node (not via sbatch):
#
#     cd /path/to/ctest1
#     bash slurm/setup_env.sh
#
# Uses your own conda installation by default (auto-detected, or set
# CONDA_BASE).  Configure persistently in slurm/env.local.sh — see
# slurm/env.local.sh.example.  Per-invocation overrides:
#
#     CONDA_BASE=/your/conda CONDA_ENV=myenv bash slurm/setup_env.sh
#     USE_CONDA_MODULE=1 bash slurm/setup_env.sh      # NERSC conda module

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# Site-local configuration
if [[ -f "$REPO_DIR/slurm/env.local.sh" ]]; then
    echo "Site config : slurm/env.local.sh"
    # shellcheck disable=SC1091
    source "$REPO_DIR/slurm/env.local.sh"
fi

CONDA_ENV="${CONDA_ENV:-abacus}"
PYVER="${PYVER:-3.10}"

if [[ "${USE_CONDA_MODULE:-0}" == "1" ]]; then
    module load conda
fi

# Locate the conda installation
if [[ -z "${CONDA_BASE:-}" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    fi
fi

if [[ -z "${CONDA_BASE:-}" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: could not locate a conda installation." >&2
    echo "       CONDA_BASE='${CONDA_BASE:-<unset>}'" >&2
    echo "       Find your conda root with:   conda info --base" >&2
    echo "       Then re-run:  CONDA_BASE=/your/conda bash slurm/setup_env.sh" >&2
    exit 1
fi

echo "Repo dir    : $REPO_DIR"
echo "Conda base  : $CONDA_BASE"
echo "Environment : $CONDA_ENV  (python $PYVER)"
echo

# conda's init scripts reference unset variables; relax -u around them
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
set -u

# Create the env if needed.  CONDA_ENV may be a name or a full prefix path.
if [[ "$CONDA_ENV" == */* ]]; then
    if [[ -d "$CONDA_ENV" ]]; then
        echo "Environment prefix already exists — reusing it."
    else
        echo "Creating conda environment at prefix '$CONDA_ENV'…"
        conda create -p "$CONDA_ENV" "python=$PYVER" -y
    fi
elif conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$CONDA_ENV"; then
    echo "Environment '$CONDA_ENV' already exists — reusing it."
else
    echo "Creating conda environment '$CONDA_ENV'…"
    conda create -n "$CONDA_ENV" "python=$PYVER" -y
fi

set +u
conda activate "$CONDA_ENV"
set -u

echo
echo "Installing dependencies from requirements.txt…"
pip install -r requirements.txt

echo
echo "Verifying…"
python - <<'PY'
import importlib, sys
print(f"  python      {sys.version.split()[0]}  ({sys.executable})")
missing = []
for mod in ("numpy", "scipy", "h5py", "yaml", "abacusnbody"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:11s} {getattr(m, '__version__', 'ok')}")
    except ImportError as exc:
        print(f"  {mod:11s} MISSING — {exc}")
        missing.append(mod)
sys.exit(1 if missing else 0)
PY

echo
if [[ ! -f "$REPO_DIR/slurm/env.local.sh" ]]; then
    echo "TIP: record your settings so every job picks them up automatically:"
    echo "    cp slurm/env.local.sh.example slurm/env.local.sh"
    echo "    # then set CONDA_BASE=$CONDA_BASE and CONDA_ENV=$CONDA_ENV"
    echo
fi
echo "Done.  Submit jobs from the repo root ($REPO_DIR):"
echo "    sbatch slurm/perlmutter_prepare_sim.sbatch"
echo "    sbatch slurm/perlmutter_lrg_hods_array.sbatch"

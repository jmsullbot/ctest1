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

# Prefix of env $1 (a name or a path), if it exists. A name is looked up the way
# conda itself does it: <envs_dir>/<name> for each of its envs_dirs, in order.
# "conda env list" is NOT used: it fails outright when one envs_dir is
# unreadable, and "conda create -y" DELETES an existing env of the same name.
env_prefix() {
    local name="$1" d
    if [[ "$name" == */* ]]; then
        if [[ -d "$name" ]]; then printf '%s\n' "${name%/}"; return 0; fi
        return 1
    fi
    while IFS= read -r d; do
        if [[ -n "$d" && -d "$d/$name" ]]; then printf '%s\n' "$d/$name"; return 0; fi
    done < <( set +eu
              conda config --show envs_dirs 2>/dev/null | sed -n 's/^[[:space:]]*-[[:space:]]*//p'
              printf '%s\n' "$CONDA_BASE/envs" "$HOME/.conda/envs" )
    return 1
}

# Create the env if needed.  CONDA_ENV may be a name or a full prefix path.
if ENV_DIR="$(env_prefix "$CONDA_ENV")"; then
    echo "Environment '$CONDA_ENV' already exists — reusing it ($ENV_DIR)."
elif [[ "$CONDA_ENV" == */* ]]; then
    echo "Creating conda environment at prefix '$CONDA_ENV'…"
    conda create -p "$CONDA_ENV" "python=$PYVER" -y
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

# Corrfunc is deliberately NOT in requirements.txt: pip builds it from source
# and needs GSL >= 2.4 plus a working compiler, which fails on many clusters
# and would take the whole pip step down with it.  Install it from conda-forge
# (prebuilt, pulls GSL) and only fall back to pip.
#
# It is required even though this pipeline never computes clustering:
# abacusnbody.hod.abacus_hod imports ..analysis.tpcf_corrfunc at module level,
# which raises ImportError when Corrfunc is missing.
if python -c "import Corrfunc" 2>/dev/null; then
    echo
    echo "Corrfunc already present."
else
    echo
    echo "Installing Corrfunc (needed to import AbacusHOD)…"
    if ! conda install -c conda-forge -y corrfunc; then
        echo "conda-forge install failed; falling back to pip (needs GSL + compiler)…"
        pip install Corrfunc
    fi
fi

echo
echo "Verifying…"
python - <<'PY'
import importlib, sys
print(f"  python      {sys.version.split()[0]}  ({sys.executable})")
missing = []
for mod in ("numpy", "scipy", "h5py", "yaml", "Corrfunc", "abacusnbody"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:11s} {getattr(m, '__version__', 'ok')}")
    except ImportError as exc:
        print(f"  {mod:11s} MISSING — {exc}")
        missing.append(mod)

# The real test: this is the import every array task performs.
try:
    from abacusnbody.hod.abacus_hod import AbacusHOD  # noqa: F401
    print("  AbacusHOD   importable")
except Exception as exc:
    print(f"  AbacusHOD   FAILED — {type(exc).__name__}: {exc}")
    missing.append("abacusnbody.hod.abacus_hod")

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

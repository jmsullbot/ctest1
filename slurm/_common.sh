# Shared preamble for the Perlmutter sbatch scripts.  Sourced, not executed.
#
# Responsibilities:
#   1. Locate the repo root regardless of where sbatch was invoked from
#      (repo root, slurm/, or anywhere else via REPO_DIR=...).
#   2. Activate the Python environment, failing early and loudly.
#
# ---------------------------------------------------------------------------
# Environment activation
# ---------------------------------------------------------------------------
# Batch jobs inherit your PATH but NOT shell functions, so a bare
# `conda activate` fails in a job script even when conda works fine in your
# login shell.  We therefore source conda's shell hook explicitly.
#
# Configure via slurm/env.local.sh (gitignored, see env.local.sh.example)
# or by exporting these on the submit line:
#
#   CONDA_BASE   root of your conda install (the dir containing
#                etc/profile.d/conda.sh).  Auto-detected from $CONDA_EXE or
#                `conda info --base` when unset.
#   CONDA_ENV    env name, or a full path to an env prefix.  Default: abacus
#   REPO_DIR     repo location, if submitting from outside it
#
# If slurm/env.local.sh fully sets up your environment by itself (e.g. it
# sources your own activation script), set ENV_READY=1 in it and the conda
# handling below is skipped entirely.

set -euo pipefail

# --- 1. Locate the repo root -----------------------------------------------
# Can't rely on $0: SLURM copies the batch script into a spool directory.
_anchor="generate_lrg_hods.py"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

if [[ ! -f "$REPO_DIR/$_anchor" && -f "$REPO_DIR/../$_anchor" ]]; then
    REPO_DIR="$(cd "$REPO_DIR/.." && pwd)"      # submitted from slurm/
fi

if [[ ! -f "$REPO_DIR/$_anchor" ]]; then
    echo "ERROR: cannot find $_anchor (looked in '$REPO_DIR')." >&2
    echo "       Submit from the repo root, or point REPO_DIR at it:" >&2
    echo "         sbatch --export=ALL,REPO_DIR=/path/to/ctest1 slurm/<script>" >&2
    exit 1
fi

cd "$REPO_DIR"
echo "Repo dir   : $REPO_DIR"

# --- 2. Site-local configuration -------------------------------------------
if [[ -f "$REPO_DIR/slurm/env.local.sh" ]]; then
    echo "Site config: slurm/env.local.sh"
    # shellcheck disable=SC1091
    source "$REPO_DIR/slurm/env.local.sh"
fi

# --- 3. Activate the Python environment ------------------------------------
if [[ "${ENV_READY:-0}" != "1" ]]; then

    CONDA_ENV="${CONDA_ENV:-abacus}"

    # Optional: NERSC's own conda module (off by default; most users have
    # their own conda installation).  Set USE_CONDA_MODULE=1 to enable.
    if [[ "${USE_CONDA_MODULE:-0}" == "1" ]]; then
        module load conda
    fi

    # Find the conda installation root
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
        echo "" >&2
        echo "       Point CONDA_BASE at your conda root — the directory" >&2
        echo "       containing etc/profile.d/conda.sh.  Find it with:" >&2
        echo "         conda info --base" >&2
        echo "" >&2
        echo "       Then either put it in slurm/env.local.sh:" >&2
        echo "         echo 'CONDA_BASE=/your/conda' >> slurm/env.local.sh" >&2
        echo "       or pass it per-job:" >&2
        echo "         sbatch --export=ALL,CONDA_BASE=/your/conda slurm/<script>" >&2
        exit 1
    fi

    # conda's init scripts reference unset variables; relax -u around them
    set +u
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    _activate_ok=1
    conda activate "$CONDA_ENV" || _activate_ok=0
    set -u

    if [[ "$_activate_ok" != "1" ]]; then
        echo "ERROR: could not activate conda environment '$CONDA_ENV'" >&2
        echo "       (conda base: $CONDA_BASE)" >&2
        echo "" >&2
        echo "       Available environments:" >&2
        conda env list >&2 || true
        echo "" >&2
        echo "       Create it once on a login node:" >&2
        echo "         cd $REPO_DIR && bash slurm/setup_env.sh" >&2
        echo "       Or select an existing one (name or full prefix path):" >&2
        echo "         sbatch --export=ALL,CONDA_ENV=myenv slurm/<script>" >&2
        exit 1
    fi

    echo "Conda base : $CONDA_BASE"
    echo "Conda env  : $CONDA_ENV"
fi

echo "Python     : $(command -v python)"

# --- 4. Verify dependencies -------------------------------------------------
# Checked here so a broken env fails once, at job start, rather than once per
# array task after each has already loaded simulation data.
if ! python -c "import numpy, h5py, yaml" 2>/dev/null; then
    echo "ERROR: the active environment is missing core dependencies." >&2
    echo "       pip install -r $REPO_DIR/requirements.txt" >&2
    exit 1
fi

# abacusnbody.hod.abacus_hod imports Corrfunc unconditionally at module level
# (via ..analysis.tpcf_corrfunc), even though we never compute clustering.
# find_spec is cheap and avoids paying the numba import cost here.
if ! python -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('Corrfunc') else 1)" 2>/dev/null; then
    echo "ERROR: Corrfunc is not installed." >&2
    echo "" >&2
    echo "       AbacusHOD cannot be imported without it: abacus_hod.py has a" >&2
    echo "       module-level 'from ..analysis.tpcf_corrfunc import ...', and" >&2
    echo "       that module raises ImportError when Corrfunc is absent.  We" >&2
    echo "       never call the clustering routines, but the import still runs." >&2
    echo "" >&2
    echo "       Install it into '$CONDA_ENV' (prebuilt, pulls GSL):" >&2
    echo "         conda install -c conda-forge -y corrfunc" >&2
    echo "       Building from source instead needs GSL >= 2.4 and a compiler:" >&2
    echo "         pip install Corrfunc" >&2
    exit 1
fi

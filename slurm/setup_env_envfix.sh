#!/bin/bash
# One-time: a conda environment with abacusutils from GitHub main, which fixes
# the environment used by the assembly-bias parameters Bcent and Bsat
# (abacusorg/abacusutils PR #207 and #208, merged 2026-04-14). No release has
# the fix: up to v2.1.2 prepare_sim computes the environment one halo file at
# a time, and catalogs with B != 0 get a pattern of period L/34 along x.
#
# Run on a LOGIN node, from the repo:
#
#     bash slurm/setup_env_envfix.sh
#
# It creates the env "abacus_envfix" with Corrfunc and the numerical stack from
# conda-forge, installs abacusutils at a pinned commit with pip, and checks that
# the fix is there and AbacusHOD imports. Your existing env is not touched; its
# abacusutils version is printed for the record.
#
# Settings (environment variables):
#   NEW_ENV          env name or prefix path           (default abacus_envfix)
#   OLD_ENV          env to report the old version of  (default abacus)
#   PYVER            python version, >= 3.10           (default 3.11)
#   ABACUSUTILS_REF  commit, tag or branch to install  (default: main as of 2026-11-02)
#   CONDA_BASE       your conda root, if not found automatically
# slurm/env.local.sh is read for CONDA_BASE only, so its CONDA_ENV cannot
# change which env this script creates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

NEW_ENV="${NEW_ENV:-abacus_envfix}"
OLD_ENV="${OLD_ENV:-abacus}"
PYVER="${PYVER:-3.11}"
ABACUSUTILS_REF="${ABACUSUTILS_REF:-b4e33ed7254a7e2faf7d9ed3d399447a8484ef57}"
ABACUSUTILS_URL="${ABACUSUTILS_URL:-https://github.com/abacusorg/abacusutils}"

# ---- conda: your own installation, never a module
if [[ -z "${CONDA_BASE:-}" && -f "$REPO_DIR/slurm/env.local.sh" ]]; then
    # shellcheck disable=SC1091
    CONDA_BASE="$(set +u; source "$REPO_DIR/slurm/env.local.sh" >/dev/null 2>&1; echo "${CONDA_BASE:-}")"
fi
if [[ -z "${CONDA_BASE:-}" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    fi
fi
if [[ -z "${CONDA_BASE:-}" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: could not locate your conda installation (CONDA_BASE='${CONDA_BASE:-<unset>}')." >&2
    echo "       Re-run as:  CONDA_BASE=\$(conda info --base) bash slurm/setup_env_envfix.sh" >&2
    exit 1
fi
command -v git >/dev/null 2>&1 || { echo "ERROR: git is needed to install abacusutils from GitHub." >&2; exit 1; }

echo "Conda base  : $CONDA_BASE"
echo "New env     : $NEW_ENV  (python $PYVER)"
echo "abacusutils : $ABACUSUTILS_URL @ $ABACUSUTILS_REF"
echo

set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
set -u

env_exists() {
    if [[ "$1" == */* ]]; then [[ -d "$1" ]]; else conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$1"; fi
}

# ---- create the env: Corrfunc and the compiled stack from conda-forge in one solve
if env_exists "$NEW_ENV"; then
    echo "Environment '$NEW_ENV' already exists: reusing it."
else
    echo "Creating '$NEW_ENV' from conda-forge (this takes a few minutes)..."
    if [[ "$NEW_ENV" == */* ]]; then target=(-p "$NEW_ENV"); else target=(-n "$NEW_ENV"); fi
    conda create "${target[@]}" -c conda-forge -y "python=$PYVER" corrfunc numba scipy h5py pyyaml astropy pip
fi

set +u
conda activate "$NEW_ENV"
set -u
echo "python      : $(command -v python)"

# ---- abacusutils from GitHub at the pinned commit (pure Python; pip adds asdf, blosc, ...)
echo
echo "Installing abacusutils from GitHub..."
python -m pip install --upgrade "abacusutils @ git+${ABACUSUTILS_URL}@${ABACUSUTILS_REF}"

# ---- verify
echo
echo "Verifying..."
python - <<'PY'
import importlib, sys
print("  python      %s  (%s)" % (sys.version.split()[0], sys.executable))
bad = []
for mod in ("numpy", "scipy", "numba", "h5py", "yaml", "asdf", "Corrfunc", "abacusnbody"):
    try:
        m = importlib.import_module(mod)
        print("  %-11s %s" % (mod, getattr(m, "__version__", "ok")))
    except ImportError as exc:
        print("  %-11s MISSING: %s" % (mod, exc)); bad.append(mod)
try:
    from abacusnbody.hod import prepare_sim
    fixed = hasattr(prepare_sim, "load_env_halos")
    print("  local-environment fix (PR #207): %s" % ("yes" if fixed else "NO"))
    if not fixed:
        bad.append("the fix")
except Exception as exc:
    print("  prepare_sim FAILED: %s: %s" % (type(exc).__name__, exc)); bad.append("prepare_sim")
try:
    from abacusnbody.hod.abacus_hod import AbacusHOD  # noqa: F401  (what every HOD task imports)
    print("  AbacusHOD   importable")
except Exception as exc:
    print("  AbacusHOD   FAILED: %s: %s" % (type(exc).__name__, exc)); bad.append("AbacusHOD")
sys.exit(1 if bad else 0)
PY

# ---- for the record: what the old env has
echo
if env_exists "$OLD_ENV"; then
    if [[ "$OLD_ENV" == */* ]]; then old=(-p "$OLD_ENV"); else old=(-n "$OLD_ENV"); fi
    echo "Old env '$OLD_ENV' for the record:"
    conda run "${old[@]}" python -c "
import abacusnbody
from abacusnbody.hod import prepare_sim as p
print('  abacusutils', abacusnbody.__version__, '| has the fix:', hasattr(p, 'load_env_halos'))" 2>/dev/null \
        || echo "  (could not query it)"
else
    echo "Old env '$OLD_ENV' not found: nothing to compare."
fi

echo
echo "Done. The phase jobs use this env by default (HOD_ENV=$NEW_ENV):"
echo "    sbatch slurm/perlmutter_phases_hod_array.sbatch"

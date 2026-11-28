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
# The env is made self-contained: a conda env puts your user site-packages
# (~/.local/lib/pythonX.Y) BEFORE its own on sys.path, so a stale package there
# shadows what was just installed. (Seen: an old coverage < 7.4 there makes
# numba fail with "module 'coverage.types' has no attribute 'Tracer'".) An
# activation hook therefore sets PYTHONNOUSERSITE=1 whenever this env is
# active, in jobs too (the phase job also exports it itself). Re-running this
# script on an existing env is safe: it adds the hook, completes the install
# with ~/.local hidden from pip, and verifies again. The check lists every
# package that is loaded from outside the env, with its path.
#
# Settings (environment variables):
#   NEW_ENV          env name or prefix path           (default abacus_envfix)
#   CLEAR_PYTHONPATH=1  the hook also unsets PYTHONPATH while the env is active
#                    (use it if the check reports packages coming from PYTHONPATH)
#   OLD_ENV          env to report the old version of  (default abacus)
#   PYVER            python version, >= 3.10           (default 3.11)
#   ABACUSUTILS_REF  commit, tag or branch to install  (default: main as of 2026-11-02)
#   CONDA_BASE       your conda root, if not found automatically
# slurm/env.local.sh is read for CONDA_BASE only, so its CONDA_ENV cannot
# change which env this script creates.
# An existing env is found by looking in conda's envs_dirs, never with
# "conda env list" (which fails if one of those folders is unreadable); it is
# then reused, never recreated.

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
export CONDA_REPORT_ERRORS=false    # if a conda command crashes: no "send this report?" prompt, nothing uploaded

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

# ---- create the env: Corrfunc and the compiled stack from conda-forge in one solve
if ENV_DIR="$(env_prefix "$NEW_ENV")"; then
    echo "Environment '$NEW_ENV' already exists: reusing it ($ENV_DIR)."
    if [[ ! -x "$ENV_DIR/bin/python" ]]; then
        echo "ERROR: $ENV_DIR has no bin/python: a create was probably interrupted." >&2
        echo "       Remove that folder (rm -rf) and run this script again." >&2
        exit 1
    fi
else
    echo "Creating '$NEW_ENV' from conda-forge (this takes a few minutes)..."
    if [[ "$NEW_ENV" == */* ]]; then target=(-p "$NEW_ENV"); else target=(-n "$NEW_ENV"); fi
    conda create "${target[@]}" -c conda-forge -y "python=$PYVER" corrfunc numba scipy h5py pyyaml astropy pip
    ENV_DIR="$(env_prefix "$NEW_ENV")" || { echo "ERROR: cannot find '$NEW_ENV' after creating it." >&2; exit 1; }
fi

set +u
conda activate "$NEW_ENV"
set -u
echo "python      : $(command -v python)"

# ---- keep the env self-contained (see the header): hooks run on every activation
PREFIX="${CONDA_PREFIX:?conda activate did not set CONDA_PREFIX}"
mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"
{
    echo '# written by slurm/setup_env_envfix.sh: keep ~/.local site-packages out of this env'
    echo 'export _ENVFIX_OLD_NOUSERSITE="${PYTHONNOUSERSITE-__unset__}"'
    echo 'export PYTHONNOUSERSITE=1'
    if [[ "${CLEAR_PYTHONPATH:-0}" == 1 ]]; then
        echo 'export _ENVFIX_OLD_PYTHONPATH="${PYTHONPATH-__unset__}"'
        echo 'unset PYTHONPATH'
    fi
} > "$PREFIX/etc/conda/activate.d/zz_envfix_isolate.sh"
{
    echo 'if [ "${_ENVFIX_OLD_NOUSERSITE-__unset__}" = "__unset__" ]; then unset PYTHONNOUSERSITE; else export PYTHONNOUSERSITE="$_ENVFIX_OLD_NOUSERSITE"; fi'
    echo 'unset _ENVFIX_OLD_NOUSERSITE'
    echo 'if [ "${_ENVFIX_OLD_PYTHONPATH-__unset__}" != "__unset__" ]; then export PYTHONPATH="$_ENVFIX_OLD_PYTHONPATH"; fi'
    echo 'unset _ENVFIX_OLD_PYTHONPATH'
} > "$PREFIX/etc/conda/deactivate.d/zz_envfix_isolate.sh"
export PYTHONNOUSERSITE=1          # for the rest of this script; pip must not count ~/.local packages as installed
if [[ "${CLEAR_PYTHONPATH:-0}" == 1 ]]; then unset PYTHONPATH; fi
iso="PYTHONNOUSERSITE=1 on every activation"
if [[ "${CLEAR_PYTHONPATH:-0}" == 1 ]]; then iso+=", PYTHONPATH cleared"; fi
echo "isolation   : $iso  ($PREFIX/etc/conda/activate.d/zz_envfix_isolate.sh)"

# ---- abacusutils from GitHub at the pinned commit (pure Python; pip adds asdf, blosc, ...)
echo
echo "Installing abacusutils from GitHub..."
python -m pip install --upgrade "abacusutils @ git+${ABACUSUTILS_URL}@${ABACUSUTILS_REF}"

# numba uses the coverage package when it finds one, and needs coverage >= 7.4.
# Upgrade one that sits in THIS env; one from outside is reported by the check below.
python - <<'PY'
import importlib.util, os, subprocess, sys
try:
    spec = importlib.util.find_spec("coverage")
    origin = os.path.realpath(spec.origin) if spec is not None and spec.origin else ""
    if origin.startswith(os.path.realpath(sys.prefix) + os.sep):
        import coverage
        import coverage.types
        if not hasattr(coverage.types, "Tracer"):
            print("coverage %s in this env is too old for numba: upgrading it" % coverage.__version__)
            subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "coverage>=7.4"])
except Exception as exc:                          # never stop here: the verification reports what is wrong
    print("(coverage check skipped: %s: %s)" % (type(exc).__name__, exc))
PY

# ---- verify
echo
echo "Verifying..."
python - <<'PY'
import importlib, os, site, sys, sysconfig
startup = set(sys.modules)        # loaded before anything is imported here: the interpreter and site hooks
prefix = os.path.realpath(sys.prefix) + os.sep
stdlib = tuple(os.path.realpath(sysconfig.get_path(k)) + os.sep for k in ("stdlib", "platstdlib"))


def from_env(path):
    """True for files of this env or of its standard library."""
    path = os.path.realpath(path)
    if path.startswith(prefix):
        return True
    return path.startswith(stdlib) and "site-packages" not in path and "dist-packages" not in path


print("  python      %s  (%s)" % (sys.version.split()[0], sys.executable))
bad = []
for mod in ("numpy", "scipy", "numba", "h5py", "yaml", "asdf", "astropy", "Corrfunc", "abacusnbody"):
    try:
        m = importlib.import_module(mod)
        print("  %-11s %s" % (mod, getattr(m, "__version__", "ok")))
    except Exception as exc:                     # not only ImportError: numba fails with AttributeError
        print("  %-11s FAILED: %s: %s" % (mod, type(exc).__name__, exc)); bad.append(mod)
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
# Where do the imported packages come from? A package from ~/.local or PYTHONPATH
# is fatal if an import failed or if it shadows a package installed in this env.
# Modules that were already loaded at interpreter start-up are site hooks (on
# Perlmutter: NERSC's nersc_pymon, via the system-wide PYTHONPATH) and harmless.
site_dirs = {os.path.realpath(sysconfig.get_path(k)) for k in ("purelib", "platlib")}


def in_env(name):
    return any(os.path.exists(os.path.join(d, name + ext)) for d in site_dirs for ext in ("", ".py"))


pulled, hooks = {}, {}
for name, m in list(sys.modules.items()):
    f = getattr(m, "__file__", None)
    if not f or "." in name or name.startswith("__") or not f.endswith((".py", ".so", ".pyc")) or from_env(f):
        continue
    if name in startup or name in ("sitecustomize", "usercustomize"):
        hooks[name] = os.path.realpath(f)
    else:
        pulled[name] = os.path.realpath(f)
shadowing = sorted(n for n in list(pulled) + list(hooks) if in_env(n))
if pulled or shadowing:
    print("  packages loaded from OUTSIDE this env:")
    for name in sorted(set(pulled) | set(shadowing)):
        print("    %-12s %s%s" % (name, pulled.get(name) or hooks[name],
                                 "   <- shadows this env's own" if name in shadowing else ""))
    print("    user site enabled: %s   PYTHONPATH: %s" % (site.ENABLE_USER_SITE, os.environ.get("PYTHONPATH") or "(unset)"))
    if bad or shadowing:
        if os.environ.get("PYTHONPATH"):
            print("    If they come from PYTHONPATH, re-run as:  CLEAR_PYTHONPATH=1 bash slurm/setup_env_envfix.sh")
        bad.append("outside packages")
    else:
        print("    Everything imports and nothing in this env is shadowed, so this is not an error.")
else:
    print("  every package comes from this env")
if hooks and not shadowing:
    print("  (start-up hooks from outside the env, harmless: %s)"
          % ", ".join("%s in %s" % (n, os.path.dirname(hooks[n])) for n in sorted(hooks)))
sys.exit(1 if bad else 0)
PY

# ---- jobs activate the env afresh: check that the hook runs then
if ( set +u
     conda deactivate >/dev/null 2>&1 || true
     unset PYTHONNOUSERSITE
     conda activate "$NEW_ENV" >/dev/null 2>&1 && [[ "${PYTHONNOUSERSITE:-}" == 1 ]] ); then
    echo "  a fresh 'conda activate' of this env sets PYTHONNOUSERSITE=1"
else
    echo "  WARNING: a fresh 'conda activate $NEW_ENV' did not set PYTHONNOUSERSITE=1 (hook not run)."
    echo "           The phase job exports it itself; in your own shells run:  export PYTHONNOUSERSITE=1"
fi

# ---- for the record: what the old env has
echo
if OLD_DIR="$(env_prefix "$OLD_ENV")" && [[ -x "$OLD_DIR/bin/python" ]]; then
    echo "Old env '$OLD_ENV' for the record:"
    # read the version and the source without importing the package (no numba needed);
    # the old env keeps its usual surroundings, ~/.local included
    env -u PYTHONNOUSERSITE "$OLD_DIR/bin/python" -c "
import importlib.util, os
from importlib.metadata import version
pkg = importlib.util.find_spec('abacusnbody').submodule_search_locations[0]
src = open(os.path.join(pkg, 'hod', 'prepare_sim.py')).read()
print('  abacusutils', version('abacusutils'), '| has the fix:', 'def load_env_halos(' in src, '|', pkg)" 2>/dev/null \
        || echo "  (could not query it)"
else
    echo "Old env '$OLD_ENV' not found: nothing to compare."
fi

echo
if [[ "$NEW_ENV" == abacus_envfix ]]; then
    echo "Done. The phase job uses this env by default:"
    echo "    sbatch slurm/perlmutter_phases_hod_array.sbatch"
else
    echo "Done. Name this env when you submit the phase job:"
    echo "    sbatch --export=ALL,HOD_ENV=$NEW_ENV slurm/perlmutter_phases_hod_array.sbatch"
fi

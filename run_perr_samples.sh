#!/bin/bash
# Run perr_from_ic.py on several catalogs, then plot them together (plot_fig4.py).
# Paste the IC and the catalogs below (label without spaces, then path), set
# CONDA_ENV, and from the directory holding perr_from_ic.py and plot_fig4.py:
#   bash run_perr_samples.sh       (login node: OK for a few 256^3 grids)
#   sbatch run_perr_samples.sh     (compute node, with the #SBATCH lines)
# The script activates CONDA_ENV itself (your own conda, found from $CONDA_EXE,
# `conda info --base` or CONDA_BASE in slurm/env.local.sh); ENV_READY=1 skips that.
#SBATCH -J perr_samples
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o perr_samples_%j.out
##SBATCH -A <your_allocation>
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-nbodykit_env_25}"   # env with numpy, scipy, matplotlib (name or prefix)
IC=/path/to/ic_256.npy             # linear density grid at z = 49
ZOUT=3
LOS="all"                          # z | x | y | several, e.g. "x z" | all (mean of the three)
OUTDIR=perr_runs
CATALOGS=(                         # "label  catalog.npy"
    "LAE_run1    /path/to/catalog_1.npy"
    "LAE_run1b   /path/to/catalog_2.npy"
    "LBG_run1i   /path/to/catalog_3.npy"
    "LBG_run1ii  /path/to/catalog_4.npy"
)

# Where perr_from_ic.py and plot_fig4.py are: next to this script (bash, also inside
# salloc), else the submit directory (sbatch runs a spooled copy), else here.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z ${CODE:-} ]]; then
    for d in "$here" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        [[ -n $d && -f $d/perr_from_ic.py && -f $d/plot_fig4.py ]] && { CODE="$d"; break; }
    done
fi
[[ -n ${CODE:-} && -f $CODE/perr_from_ic.py && -f $CODE/plot_fig4.py ]] || {
    echo "cannot find perr_from_ic.py and plot_fig4.py (looked in $here ${SLURM_SUBMIT_DIR:-} $PWD); set CODE=/their/dir"; exit 1; }
WORKERS="${WORKERS:-$([[ -n ${SLURM_JOB_ID:-} ]] && echo 32 || echo 8)}"   # FFT threads

# conda: your own install, never a module. A batch job inherits PATH but not the
# `conda` shell function, so source conda.sh and activate here.
if [[ "${ENV_READY:-0}" != 1 ]]; then
    want="$CONDA_ENV"                  # slurm/env.local.sh is read for CONDA_BASE only
    set +u
    # shellcheck disable=SC1091
    [[ -f $CODE/slurm/env.local.sh ]] && source "$CODE/slurm/env.local.sh"
    CONDA_ENV="$want"
    if [[ -z ${CONDA_BASE:-} ]]; then
        if [[ -n ${CONDA_EXE:-} ]]; then CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
        else CONDA_BASE="$(conda info --base 2>/dev/null || true)"; fi
    fi
    [[ -f ${CONDA_BASE:-}/etc/profile.d/conda.sh ]] || { echo "cannot find conda: pass CONDA_BASE=\$(conda info --base)"; exit 1; }
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || { echo "cannot activate conda env '$CONDA_ENV' (set CONDA_ENV=)"; exit 1; }
    set -u
fi
echo "python: $(command -v python)  (conda env: ${CONDA_DEFAULT_ENV:-?})"
python -c "import numpy, scipy, matplotlib" || { echo "conda env '$CONDA_ENV' lacks numpy, scipy or matplotlib"; exit 1; }
mkdir -p "$OUTDIR"

runs=()
for entry in "${CATALOGS[@]}"; do
    read -r label cat <<< "$entry"
    echo "=== $label: $cat"
    # shellcheck disable=SC2086  (LOS may hold several axes)
    python "$CODE/perr_from_ic.py" --ic "$IC" --catalog "$cat" --z-out "$ZOUT" --los $LOS \
        --out "$OUTDIR/$label" --overwrite --workers "$WORKERS" 2>&1 | tee "$OUTDIR/$label.log"
    runs+=(--run "$OUTDIR/$label" "$label")
done

python "$CODE/plot_fig4.py" "${runs[@]}" --z-out "$ZOUT" --out "$OUTDIR/samples"
echo; echo "=== catalog layout and RSD shifts (expect rms ~1.5-2.5 Mpc/h, mean and correlation ~0)"
grep -H -E "columns read|field '|RSD shift" "$OUTDIR"/*.log || true
grep -H "WARNING" "$OUTDIR"/*.log || echo "no warnings"
echo "figures (.png and .pdf): $OUTDIR/samples_fig4  $OUTDIR/samples_fig5  $OUTDIR/samples_rsdcheck"

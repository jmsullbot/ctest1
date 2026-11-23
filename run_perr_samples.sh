#!/bin/bash
# Run perr_from_ic.py on several catalogs, then plot them together (plot_fig4.py).
# Paste the IC and the catalogs below (label without spaces, then path), activate
# your conda env, and from the directory holding perr_from_ic.py and plot_fig4.py:
#   bash run_perr_samples.sh       (login node: OK for a few 256^3 grids)
#   sbatch run_perr_samples.sh     (compute node, with the #SBATCH lines)
#SBATCH -J perr_samples
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o perr_samples_%j.out
##SBATCH -A <your_allocation>
set -euo pipefail

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

CODE="${CODE:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
WORKERS="${WORKERS:-$([[ -n ${SLURM_JOB_ID:-} ]] && echo 32 || echo 8)}"   # FFT threads
[[ -f $CODE/perr_from_ic.py && -f $CODE/plot_fig4.py ]] || { echo "no perr_from_ic.py / plot_fig4.py in $CODE (set CODE=)"; exit 1; }
python -c "import numpy, scipy, matplotlib" || { echo "activate the conda env with numpy, scipy, matplotlib first"; exit 1; }
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
echo "figures: $OUTDIR/samples_fig4.png  $OUTDIR/samples_fig5.png  $OUTDIR/samples_rsdcheck.png"

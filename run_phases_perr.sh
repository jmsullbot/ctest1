#!/bin/bash
# Field-level transfer functions and P_err (perr_from_ic.py) for one HOD row on
# every AbacusSummit_base_c000 phase box, then the ensemble summary
# (phase_ensemble.py). Run after the HOD job array and the IC transfer:
#
#   bash slurm/check_phases.sh                     # which phases have catalog + IC
#   sbatch run_phases_perr.sh                      # compute node, 8 phases at a time
#   bash run_phases_perr.sh                        # or a login node (2 at a time; slow at 512)
#   sbatch --export=ALL,NMESH=256,LOS=z run_phases_perr.sh
#
# Per phase NNN: $OUTROOT/phNNN/perr/hod<ROW>_N<NMESH>_{real,rsd}.npy (+ _ops
# sidecars, per-LOS files, figures, .log). Phases with results are skipped
# (FORCE=1 redoes them), so resubmitting continues where it stopped.
# The ensemble goes to $OUTROOT/ensemble_hod<ROW>_N<NMESH>*.
#
# Settings (environment or --export=ALL,...):
#   ROW 1003 | NMESH 512 (from the N576 IC by k-space truncation) | LOS all
#   PHASES "0 1 ... 24" | COSMO c000 | ZOUT 0.5
#   OUTROOT $PSCRATCH/lrg_hods_phases | IC_ROOT $PSCRATCH/AbacusSummit | IC_FILE ic_dens_N576.asdf
#   COSMO_ARGS  growth cosmology; the default reproduces AbacusSummit c000's own
#               D(0.5)/D(99) = 59.9115 and f = 0.75907 (checked against the IC header)
#   EXTRA       more perr_from_ic.py flags, e.g. "--real-no-dz" (lrgs.py-like real space)
#   NPAR        phases at once (default 8 in a job, 2 elsewhere; ~25-40 GB each at 512)
#   CONDA_ENV   needs numpy, scipy, asdf and abacusutils (Blosc); default abacus
#SBATCH -J perr_phases
# Uncomment and set if you have more than one NERSC allocation; with a
# single default allocation SLURM picks it automatically.
##SBATCH -A <your_allocation>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256
#SBATCH -t 02:00:00
#SBATCH -o perr_phases_%j.out
set -euo pipefail

ROW="${ROW:-1003}"
NMESH="${NMESH:-512}"
LOS="${LOS:-all}"
COSMO="${COSMO:-c000}"
ZOUT="${ZOUT:-0.5}"
PHASES="${PHASES:-$(seq 0 24)}"
OUTROOT="${OUTROOT:-$PSCRATCH/lrg_hods_phases}"
IC_ROOT="${IC_ROOT:-$PSCRATCH/AbacusSummit}"
IC_FILE="${IC_FILE:-ic_dens_N576.asdf}"
COSMO_ARGS="${COSMO_ARGS---omega-m 0.315192 --omega-nu 0.0014197 --omega-r 0}"   # set "" for perr_from_ic.py's defaults
EXTRA="${EXTRA:-}"
CONDA_ENV="${CONDA_ENV:-abacus}"
FORCE="${FORCE:-0}"
if [[ -n ${SLURM_JOB_ID:-} ]]; then NPAR="${NPAR:-8}"; else NPAR="${NPAR:-2}"; fi
THREADS="${THREADS:-$(( $( [[ -n ${SLURM_JOB_ID:-} ]] && echo 256 || echo 16 ) / NPAR ))}"
(( THREADS >= 1 )) || THREADS=1

# perr_from_ic.py, fit_bias.py and phase_ensemble.py: next to this script, else
# the submit directory (sbatch runs a spooled copy), else here
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z ${CODE:-} ]]; then
    for d in "$here" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        [[ -n $d && -f $d/perr_from_ic.py && -f $d/phase_ensemble.py ]] && { CODE="$d"; break; }
    done
fi
[[ -n ${CODE:-} && -f $CODE/perr_from_ic.py ]] || { echo "cannot find perr_from_ic.py / phase_ensemble.py; set CODE=" >&2; exit 1; }

# conda: your own install, never a module (a job inherits PATH but not the conda function)
if [[ "${ENV_READY:-0}" != 1 ]]; then
    want="$CONDA_ENV"
    set +u
    # shellcheck disable=SC1091
    [[ -f $CODE/slurm/env.local.sh ]] && source "$CODE/slurm/env.local.sh"   # for CONDA_BASE only
    CONDA_ENV="$want"
    if [[ -z ${CONDA_BASE:-} ]]; then
        if [[ -n ${CONDA_EXE:-} ]]; then CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
        else CONDA_BASE="$(conda info --base 2>/dev/null || true)"; fi
    fi
    [[ -f ${CONDA_BASE:-}/etc/profile.d/conda.sh ]] || { echo "cannot find conda: pass CONDA_BASE=\$(conda info --base)" >&2; exit 1; }
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || { echo "cannot activate conda env '$CONDA_ENV' (set CONDA_ENV=)" >&2; exit 1; }
    set -u
fi
echo "python: $(command -v python)  (conda env: ${CONDA_DEFAULT_ENV:-?})"
python -c "import numpy, scipy, asdf" || { echo "env '$CONDA_ENV' lacks numpy, scipy or asdf" >&2; exit 1; }
python -c "import abacusnbody" 2>/dev/null || echo "WARNING: abacusutils not importable: Blosc-compressed Abacus ICs will not read"
python -c "import matplotlib" 2>/dev/null || echo "WARNING: no matplotlib: figures are skipped"
echo "row $ROW, Nmesh $NMESH, LOS $LOS, $NPAR phases at a time x $THREADS FFT threads; outputs under $OUTROOT"

one_phase() {
    local ph sim ic cat out
    ph="$(printf '%03d' "$((10#$1))")"
    sim="AbacusSummit_base_${COSMO}_ph$ph"
    ic="$IC_ROOT/$sim/ic/$IC_FILE"
    cat="$OUTROOT/ph$ph/catalogs/$(printf '%06d' "$ROW").npy"
    out="$OUTROOT/ph$ph/perr/hod${ROW}_N${NMESH}"
    [[ -s $ic ]]  || { echo "ph$ph: skipped, no IC $ic"; return 0; }
    [[ -s $cat ]] || { echo "ph$ph: skipped, no catalog $cat"; return 0; }
    if [[ -s ${out}_real.npy && -s ${out}_rsd.npy && $FORCE != 1 ]]; then echo "ph$ph: done already"; return 0; fi
    mkdir -p "$(dirname "$out")"
    # shellcheck disable=SC2086
    if python "$CODE/perr_from_ic.py" --ic "$ic" --catalog "$cat" --z-out "$ZOUT" --nmesh "$NMESH" \
           --los $LOS $COSMO_ARGS $EXTRA --out "$out" --workers "$THREADS" --overwrite > "$out.log" 2>&1; then
        echo "ph$ph: ok ($(grep -c WARNING "$out.log") warnings)"
    else
        echo "ph$ph: FAILED, see $out.log"
    fi
}
export -f one_phase
export ROW NMESH LOS COSMO ZOUT OUTROOT IC_ROOT IC_FILE COSMO_ARGS EXTRA FORCE THREADS CODE

t0=$SECONDS
printf '%s\n' $PHASES | xargs -P "$NPAR" -I{} bash -c 'one_phase "$@"' _ {}
echo "phases finished in $(( (SECONDS - t0) / 60 )) min"

python "$CODE/phase_ensemble.py" --root "$OUTROOT" --row "$ROW" --nmesh "$NMESH" --z-out "$ZOUT" \
    $COSMO_ARGS --out "$OUTROOT/ensemble_hod${ROW}_N${NMESH}"
echo; grep -H "WARNING" "$OUTROOT"/ph*/perr/hod${ROW}_N${NMESH}.log 2>/dev/null || echo "no warnings in the phase logs"

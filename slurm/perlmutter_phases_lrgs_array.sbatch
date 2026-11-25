#!/bin/bash
# Your field-level pipeline, my_generate_shifted_fields.py then lrgs.py, for one
# HOD row on every AbacusSummit_base_c000 phase box at Nmesh 512, exactly as for
# ph000. One array task per phase, each on a whole CPU node.
#
#   cd <folder holding my_generate_shifted_fields.py and lrgs.py, both patched>
#   sbatch /pscratch/sd/j/jsull/qso_sbp/old_lrgs/ctest1/slurm/perlmutter_phases_lrgs_array.sbatch
#   sbatch --array=7 <same path>                          # one phase
#   sbatch --export=ALL,ROWS="1003 2155" <same path>      # more rows
#
# The patches add --ic/--out/--axes to my_generate_shifted_fields.py and
# --ph/--path/--pathD to lrgs.py. Without those options both scripts behave
# exactly as before (the ph000 run). This job refuses to start with an
# unpatched generator, which would rewrite the ph000 fields.
#
# Each task, for phase NNN (= the task id), works under $OUTROOT/phNNN/, next to
# the catalogs/ the HOD array wrote:
#   1. python my_generate_shifted_fields.py --ic <phase IC> --out phNNN/shifted_fields/ --axes 2
#      Real space plus redshift-space axis 2, the only axis lrgs.py reads.
#      Skipped when the files exist at full size (FORCE_FIELDS=1 redoes them).
#      [mem] lines log node memory every minute; ph000's run peaked at 306 GiB.
#   2. python lrgs.py --batch_size 1 --batch_num <row> --ph NNN --path phNNN/ --pathD phNNN/catalogs/
#      for each row in ROWS. lrgs.py itself skips results that exist.
# Outputs:
#   phNNN/shifted_fields/shifted_fields_real_N_512_c_0.npy                 2.1 GB
#   phNNN/shifted_fields/shifted_fields_redshift_axisrsd_2_N_512_c_0.npy   3.2 GB
#   phNNN/results/results_AbacusBase_c=0_ph=NNN_hod=1003_z=0.5000_Nmesh_512.npy
#   phNNN/results/results_RSD_AbacusBase_c=0_ph=NNN_hod=1003_z=0.5000_Nmesh_512_axisrsd_2.npy
# ph000 is left out of the default array: its fields and results already exist
# in Heavy_files_Andrej/AbacusBase/ (ALLOW_PH000=1 recomputes it here anyway).
# Resubmitting is safe: finished fields and results are not redone.
#
# Settings (--export=ALL,VAR=...):
#   ROWS 1003 | NMESH 512 (checked against both scripts) | COSMO c000
#   AXES "2" (must include 2; "0 1 2" writes all three, then use -t 04:00:00)
#   OUTROOT $PSCRATCH/lrg_hods_phases | IC_ROOT $PSCRATCH/AbacusSummit | IC_FILE ic_dens_N576.asdf
#   GEN, LRGS        the two scripts (default: in the submit directory)
#   CONDA_ENV        default nbodykit_env_25 (nbodykit, asdf, abacusutils); ENV_READY=1 skips activation
#   FORCE_FIELDS=1   regenerate the fields even if they exist
#   DELETE_FIELDS=1  delete this phase's fields once its results are written (5.4 GB each)
#   TRACE_SEC        memory logging interval (default 60)
#SBATCH -J lrgs_phases
# Uncomment and set if you have more than one NERSC allocation; with a
# single default allocation SLURM picks it automatically.
##SBATCH -A <your_allocation>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256
#SBATCH -t 03:00:00
#SBATCH --array=1-24
#SBATCH -o lrgs_phases_%A_%a.out
set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ROWS="${ROWS:-1003}"
NMESH="${NMESH:-512}"
AXES="${AXES:-2}"
COSMO="${COSMO:-c000}"
OUTROOT="${OUTROOT:-$PSCRATCH/lrg_hods_phases}"
IC_ROOT="${IC_ROOT:-$PSCRATCH/AbacusSummit}"
IC_FILE="${IC_FILE:-ic_dens_N576.asdf}"
GEN="${GEN:-$SUBMIT_DIR/my_generate_shifted_fields.py}"
LRGS="${LRGS:-$SUBMIT_DIR/lrgs.py}"
CONDA_ENV="${CONDA_ENV:-nbodykit_env_25}"
TRACE_SEC="${TRACE_SEC:-60}"
PH="$(printf '%03d' "$((10#${PHASE:-${SLURM_ARRAY_TASK_ID:?run as an array job, or set PHASE=}}))")"
SIM="AbacusSummit_base_${COSMO}_ph${PH}"
PHDIR="$OUTROOT/ph$PH"
IC="$IC_ROOT/$SIM/ic/$IC_FILE"
FIELDS="$PHDIR/shifted_fields"
REAL="$FIELDS/shifted_fields_real_N_${NMESH}_c_0.npy"

echo "=== $SIM: shifted fields + lrgs.py, task ${SLURM_ARRAY_TASK_ID:-local} on $(hostname), $(date)"
echo "rows: $ROWS   Nmesh: $NMESH   redshift-space axes: $AXES   phase dir: $PHDIR"

if [[ $PH == 000 && "${ALLOW_PH000:-0}" != 1 ]]; then
    echo "ph000 already has its fields and results in Heavy_files_Andrej/AbacusBase/: nothing to do"
    echo "(ALLOW_PH000=1 recomputes it under $PHDIR)."
    exit 0
fi

# ------------------------------------------------ the two scripts and inputs
for s in "$GEN" "$LRGS"; do
    [[ -f $s ]] || { echo "ERROR: $s not found. Submit from its folder, or set GEN= / LRGS=." >&2; exit 2; }
done
if ! grep -q "'--out'" "$GEN"; then
    echo "ERROR: $GEN has no --ic/--out options (not patched?). Unpatched, it would read the" >&2
    echo "       ph000 IC and rewrite the ph000 fields. Apply my_generate_shifted_fields_phases.diff." >&2
    exit 2
fi
grep -q "'--pathD'" "$LRGS" || { echo "ERROR: $LRGS has no --ph/--path/--pathD options (not patched?)" >&2; exit 2; }
for s in "$GEN" "$LRGS"; do
    n="$(grep -E '^Nmesh *= *[0-9]+' "$s" | tail -1 | sed -E 's/^Nmesh *= *([0-9]+).*/\1/')"
    [[ $n == "$NMESH" ]] || { echo "ERROR: $s sets Nmesh = ${n:-?}, but this job expects $NMESH" >&2; exit 2; }
done
[[ " $AXES " == *" 2 "* ]] || { echo "ERROR: AXES='$AXES' must include 2 (lrgs.py reads axis 2)" >&2; exit 2; }
[[ -s $IC ]] || { echo "ERROR: no IC at $IC" >&2; exit 3; }
for r in $ROWS; do
    c="$PHDIR/catalogs/$(printf '%06d' "$r").npy"
    [[ -s $c ]] || { echo "ERROR: no catalog $c (run slurm/perlmutter_phases_hod_array.sbatch first)" >&2; exit 3; }
done

# ------------------------------------------------ conda (your own install, never a module)
if [[ "${ENV_READY:-0}" != 1 ]]; then
    if [[ -z "${CONDA_BASE:-}" ]]; then
        if [[ -n "${CONDA_EXE:-}" ]]; then CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
        elif command -v conda >/dev/null 2>&1; then CONDA_BASE="$(conda info --base 2>/dev/null || true)"; fi
    fi
    if [[ -z "${CONDA_BASE:-}" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        echo "ERROR: cannot locate conda (CONDA_BASE='${CONDA_BASE:-unset}'); pass --export=ALL,CONDA_BASE=\$(conda info --base)" >&2
        exit 1
    fi
    set +u; source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || { echo "ERROR: cannot activate '$CONDA_ENV' (pass CONDA_ENV=...)" >&2; exit 1; }
    set -u
fi
echo "python: $(command -v python)  (conda env: ${CONDA_DEFAULT_ENV:-?})"
python -c "import nbodykit, asdf" || { echo "ERROR: this env lacks nbodykit or asdf" >&2; exit 4; }
export PYTHONUNBUFFERED=1

# ------------------------------------------------ 1. shifted fields
complete() {   # file, number of NMESH^3 float32 grids it must hold
    local f=$1 n=$2
    [[ -f $f ]] && (( $(stat -c %s "$f") >= n * NMESH * NMESH * NMESH * 4 ))
}
fields_done() {
    complete "$REAL" 4 || return 1
    local a
    for a in $AXES; do complete "$FIELDS/shifted_fields_redshift_axisrsd_${a}_N_${NMESH}_c_0.npy" 6 || return 1; done
}
if fields_done && [[ "${FORCE_FIELDS:-0}" != 1 ]]; then
    echo "shifted fields exist in $FIELDS: generator skipped (FORCE_FIELDS=1 redoes them)"
else
    mkdir -p "$FIELDS"
    echo; echo "=== generator started $(date +%T): python $GEN --ic $IC --out $FIELDS/ --axes $AXES"
    t0=$SECONDS
    # shellcheck disable=SC2086
    python "$GEN" --ic "$IC" --out "$FIELDS/" --axes $AXES &
    gen=$!
    (
        sp=""; trap '[[ -n "$sp" ]] && kill "$sp" 2>/dev/null; exit 0' TERM
        while kill -0 "$gen" 2>/dev/null; do
            used="$(free -g | awk '/^Mem:/{print $3}')"
            rss="$(ps -o rss= -p "$gen" 2>/dev/null | awk '{printf "%.1f", $1/1048576}')"
            echo "[mem $(date +%T)] node used ${used} GiB, generator RSS ${rss:-?} GiB"
            sleep "$TRACE_SEC" & sp=$!
            wait "$sp"
        done
    ) &
    tracer=$!
    wait "$gen"; rc=$?
    kill "$tracer" 2>/dev/null; wait "$tracer" 2>/dev/null
    echo "=== generator exit code $rc after $(( (SECONDS - t0) / 60 )) min"
    (( rc == 137 )) && echo "killed by SIGKILL: almost certainly out of memory (see the last [mem] lines)"
    (( rc == 0 )) || exit 5
    fields_done || { echo "ERROR: the generator finished but the fields in $FIELDS are missing or short" >&2; exit 5; }
    ls -l "$FIELDS"
fi

# ------------------------------------------------ 2. lrgs.py, one row at a time
for r in $ROWS; do
    tag="c=0_ph=${PH}_hod=${r}_z=0.5000_Nmesh_${NMESH}"
    res_real="$PHDIR/results/results_AbacusBase_${tag}.npy"
    res_rsd="$PHDIR/results/results_RSD_AbacusBase_${tag}_axisrsd_2.npy"
    echo; echo "=== lrgs.py row $r started $(date +%T)"
    t0=$SECONDS
    python "$LRGS" --batch_size 1 --batch_num "$r" --ph "$PH" --path "$PHDIR/" --pathD "$PHDIR/catalogs/" \
        || { echo "ERROR: lrgs.py failed for row $r" >&2; exit 6; }
    for f in "$res_real" "$res_rsd"; do
        [[ -s $f ]] || { echo "ERROR: $f was not written" >&2; exit 6; }
    done
    echo "row $r done in $(( (SECONDS - t0) / 60 )) min:"; echo "  $res_real"; echo "  $res_rsd"
done

if [[ "${DELETE_FIELDS:-0}" == 1 ]]; then
    echo "DELETE_FIELDS=1: removing $FIELDS ($(du -sh "$FIELDS" | cut -f1))"
    rm -rf "$FIELDS"
fi
echo; echo "=== $SIM done $(date)"

#!/bin/bash
# HOD catalogs for a few table rows (default: row 1003) on every
# AbacusSummit_base_c000 phase box, ph000-ph024: one array task per phase.
# Each task runs prepare_sim for its box (skipped when that box's subsamples
# are complete), then generates the rows.
#
# This job REQUIRES abacusutils with the local-environment fix (PR #207, on
# GitHub main since 2026-04-14, in no release up to v2.1.2). Without it the
# assembly-bias environment is computed one halo file at a time, and Bcent /
# Bsat imprint a pattern of period L/34 along x on the catalogs. Create the
# env once on a login node:
#
#   bash slurm/setup_env_envfix.sh
#   bash slurm/check_phases.sh                          # which boxes are on disk
#   sbatch slurm/perlmutter_phases_hod_array.sbatch      # all 25 phases
#   sbatch --array=7 slurm/perlmutter_phases_hod_array.sbatch
#   sbatch --export=ALL,ROWS="1003 2155" slurm/perlmutter_phases_hod_array.sbatch
#
# The fixed code needs its own prepare_sim run (it writes one extra
# env_xcom_<i>_abacushod_localenv_new.h5 per slab), so everything goes to NEW
# folders and nothing from the earlier, unfixed run is read or overwritten:
#   $SUBSAMPLE_DIR/<sim>/z0.500/                  subsamples, one folder per box
#   $OUTROOT/phNNN/catalogs/<row>.npy            galaxy catalogs
#   $OUTROOT/phNNN/lrg_hods_rows<a>-<b>.hdf5     metadata (sim_name, ...)
#   $OUTROOT/phNNN/config_<sim>.yaml             the config this box ran with
# Check the size of one box's subsamples before running all 25:
#   du -sh <old subsample_dir>/AbacusSummit_base_c000_ph000
#
# Settings (--export=ALL,VAR=...):
#   CONFIG           base config (default config/lrg_hod.yaml); only sim_name and
#                    subsample_dir are replaced, in a copy
#   HOD_ENV          conda env with the fixed abacusutils (default abacus_envfix).
#                    It is activated here even if slurm/env.local.sh names another.
#   SUBSAMPLE_DIR    default $PSCRATCH/lrg_hods_envfix/subsamples
#   OUTROOT          default $PSCRATCH/lrg_hods_phases_envfix
#   ROWS             table rows (default 1003)
#   COSMO            default c000
#   FORCE_PREPARE=1  rerun prepare_sim even if the box's subsamples are complete
#   CLEAN_SUBSAMPLES=1  delete the subsamples THIS task created, once the rows are
#                    written (frees scratch; more rows later need prepare_sim again)
#   PROGRESS_SEC     seconds between "[prepare_sim ...] n of N slabs written" lines (default 300)
#   ALLOW_UNFIXED=1  run with an abacusutils that lacks the fix (not recommended)
#SBATCH -J lrg_phases
# Uncomment and set if you have more than one NERSC allocation; with a
# single default allocation SLURM picks it automatically.
##SBATCH -A <your_allocation>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256
#SBATCH -t 03:00:00          # prepare_sim 30-90 min (I/O-bound) + one row
#SBATCH --array=0-24
#SBATCH -o lrg_phases_%A_%a.out

# Resolves REPO_DIR (cd's there) and activates the env of slurm/env.local.sh, or
# exits with setup instructions.
if   [[ -f "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh"
elif [[ -f "${SLURM_SUBMIT_DIR:-$PWD}/_common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR:-$PWD}/_common.sh"
elif [[ -n "${REPO_DIR:-}" && -f "$REPO_DIR/slurm/_common.sh" ]]; then
    source "$REPO_DIR/slurm/_common.sh"
else
    echo "ERROR: cannot locate slurm/_common.sh. Submit from the repo root," >&2
    echo "       or use: sbatch --export=ALL,REPO_DIR=/path/to/ctest1 ..." >&2
    exit 1
fi

# ---- the env with the fixed abacusutils, whatever slurm/env.local.sh activated
HOD_ENV="${HOD_ENV:-abacus_envfix}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$HOD_ENV" && "${CONDA_PREFIX:-}" != "$HOD_ENV" ]]; then
    if ! declare -F conda >/dev/null; then
        echo "ERROR: need conda env '$HOD_ENV', but conda is not set up in this job" >&2
        echo "       (ENV_READY=1 in slurm/env.local.sh?). Activate '$HOD_ENV' there, or set HOD_ENV." >&2
        exit 1
    fi
    set +u
    conda activate "$HOD_ENV" || {
        echo "ERROR: cannot activate conda env '$HOD_ENV'. Create it once on a login node:" >&2
        echo "         bash slurm/setup_env_envfix.sh" >&2
        exit 1
    }
    set -u
    echo "HOD env    : $HOD_ENV  ($(command -v python))"
fi
FIXED=1
if ! python - <<'EOF'
import sys
import abacusnbody
from abacusnbody.hod import prepare_sim
ok = hasattr(prepare_sim, "load_env_halos")
print("abacusutils: %s   local-environment fix (PR #207): %s" % (abacusnbody.__version__, "yes" if ok else "NO"))
sys.exit(0 if ok else 1)
EOF
then
    if [[ "${ALLOW_UNFIXED:-0}" == 1 ]]; then
        echo "WARNING: ALLOW_UNFIXED=1: running WITHOUT the local-environment fix."
        FIXED=0
    else
        echo "ERROR: the abacusutils in this env lacks the local-environment fix, so Bcent/Bsat" >&2
        echo "       would imprint the L/34 pattern again. Create the fixed env with" >&2
        echo "         bash slurm/setup_env_envfix.sh" >&2
        echo "       (ALLOW_UNFIXED=1 runs anyway.)" >&2
        exit 5
    fi
fi

CONFIG="${CONFIG:-config/lrg_hod.yaml}"
OUTROOT="${OUTROOT:-$PSCRATCH/lrg_hods_phases_envfix}"
SUBSAMPLE_DIR="${SUBSAMPLE_DIR:-$PSCRATCH/lrg_hods_envfix/subsamples}"
ROWS="${ROWS:-1003}"
COSMO="${COSMO:-c000}"
PH="$(printf '%03d' "$((10#${PHASE:-${SLURM_ARRAY_TASK_ID:?run as an array job, or set PHASE=}}))")"
SIM="AbacusSummit_base_${COSMO}_ph${PH}"
OUT="$OUTROOT/ph$PH"
mkdir -p "$OUT"

# The config for this run: CONFIG with subsample_dir replaced (prepare_sim joins
# it to the sim name as a string, hence the trailing slash). generate_lrg_hods.py
# --sim_name then writes config_<sim>.yaml from it.
BASECFG="$OUT/config_base.yaml"
read -r SIM_DIR SUB_DIR ZDIR CLEANED < <(python - "$CONFIG" "$BASECFG" "$SUBSAMPLE_DIR" <<'EOF'
import sys, yaml
src, dst, sub = sys.argv[1:4]
cfg = yaml.safe_load(open(src))
s = cfg["sim_params"]
s["subsample_dir"] = sub.rstrip("/") + "/"
with open(dst, "w") as fh:
    fh.write("# generated from %s with subsample_dir = %s\n" % (src, s["subsample_dir"]))
    yaml.safe_dump(cfg, fh, sort_keys=False)
print(s["sim_dir"].rstrip("/"), s["subsample_dir"].rstrip("/"), "z%.3f" % float(s.get("z_mock", 0.5)),
      int(bool(s.get("cleaned_halos", True))))
EOF
)
[[ -n "${SIM_DIR:-}" && -n "${SUB_DIR:-}" ]] || { echo "ERROR: could not read sim_dir from $CONFIG" >&2; exit 2; }
echo "=== $SIM  (task ${SLURM_ARRAY_TASK_ID:-local}, $(date))"
echo "rows: $ROWS   config: $CONFIG   subsamples: $SUB_DIR   output: $OUT"

# 1. Is the box on disk? (some AbacusSummit boxes are on HPSS tape only)
missing=0
for d in halo_info halo_rv_A field_rv_A; do
    [[ -d "$SIM_DIR/$SIM/halos/$ZDIR/$d" ]] || { echo "MISSING: $SIM_DIR/$SIM/halos/$ZDIR/$d"; missing=1; }
done
if [[ $CLEANED == 1 && ! -d "$SIM_DIR/cleaning/$SIM/$ZDIR" ]]; then
    echo "MISSING: $SIM_DIR/cleaning/$SIM/$ZDIR (cleaned_halos is True in the config)"; missing=1
fi
(( missing )) && { echo "ERROR: $SIM is not fully on disk; skipping it (see slurm/check_phases.sh)." >&2; exit 3; }

# 2. prepare_sim, unless this box's subsamples are complete: one halo file, one
#    particle file and (with the fix) one environment file per halo_info slab
SUBS="$SUB_DIR/$SIM/$ZDIR"
count() { if [[ -d $1 ]]; then find "$1" -maxdepth 1 -name "$2" | wc -l; else echo 0; fi; }   # 0 for a folder not there yet
n_slab="$(count "$SIM_DIR/$SIM/halos/$ZDIR/halo_info" '*.asdf')"
n_h="$(count "$SUBS" 'halos_xcom_*_new.h5')"
n_p="$(count "$SUBS" 'particles_xcom_*_new.h5')"
n_e="$(count "$SUBS" 'env_xcom_*_abacushod_localenv_new.h5')"
echo "subsamples in $SUBS: $n_h halo, $n_p particle, $n_e environment files for $n_slab slabs"
complete=0
if (( n_slab > 0 && n_h == n_slab && n_p == n_slab )) && { (( FIXED == 0 )) || (( n_e == n_slab )); }; then complete=1; fi
if (( FIXED == 1 && n_h > 0 && n_e == 0 )) && [[ "${FORCE_PREPARE:-0}" != 1 ]]; then
    echo "ERROR: $SUBS holds subsamples without environment files: they come from the" >&2
    echo "       unfixed code. Use a new SUBSAMPLE_DIR, or FORCE_PREPARE=1 to overwrite them." >&2
    exit 3
fi
prepared_here=0
if (( complete )) && [[ "${FORCE_PREPARE:-0}" != 1 ]]; then
    echo "subsamples complete: skipping prepare_sim (FORCE_PREPARE=1 to redo)"
else
    t0=$SECONDS
    # abacusutils main reports through Python logging, which stays silent here:
    # print our own heartbeat while prepare_sim runs
    (
        sp=""; trap '[[ -n "$sp" ]] && kill "$sp" 2>/dev/null; exit 0' TERM
        while :; do
            sleep "${PROGRESS_SEC:-300}" & sp=$!
            wait "$sp"
            echo "[prepare_sim $(( (SECONDS - t0) / 60 )) min] $(count "$SUBS" 'halos_xcom_*_new.h5') of $n_slab slabs written"
        done
    ) &
    monitor=$!
    rc=0
    python generate_lrg_hods.py --prepare_only --path2config "$BASECFG" --sim_name "$SIM" \
        --output "$OUT/lrg_hods.hdf5" || rc=$?
    kill "$monitor" 2>/dev/null || true
    wait "$monitor" 2>/dev/null || true
    (( rc == 0 )) || { echo "ERROR: prepare_sim failed for $SIM (exit $rc)" >&2; exit "$rc"; }
    prepared_here=1
    if [[ -d "$SUBS" ]]; then
        touch "$SUBS/.prepared_by_phases_job"
        echo "prepare_sim done in $(( (SECONDS - t0) / 60 )) min; subsamples: $(du -sh "$SUBS" | cut -f1)"
        n_e="$(count "$SUBS" 'env_xcom_*_abacushod_localenv_new.h5')"
        if (( FIXED == 1 && n_e != n_slab )); then
            echo "ERROR: prepare_sim wrote $n_e environment files for $n_slab slabs" >&2; exit 4
        fi
    else
        echo "WARNING: prepare_sim finished but $SUBS does not exist (different layout?); CLEAN_SUBSAMPLES is disabled"
        prepared_here=0
    fi
fi

# 3. the HOD rows
for r in $ROWS; do
    python generate_lrg_hods.py --path2config "$BASECFG" --sim_name "$SIM" \
        --params lrg_params_eq74.npy --output "$OUT/lrg_hods.hdf5" \
        --Nthread 128 --start "$r" --end "$((r + 1))"
    f="$OUT/catalogs/$(printf '%06d' "$r").npy"
    [[ -s "$f" ]] || { echo "ERROR: $f was not written" >&2; exit 4; }
    echo "row $r -> $f ($(du -h "$f" | cut -f1))"
done

# 4. optionally free the subsamples this task created
if [[ "${CLEAN_SUBSAMPLES:-0}" == 1 ]]; then
    if (( prepared_here )); then
        echo "CLEAN_SUBSAMPLES=1: removing $SUBS ($(du -sh "$SUBS" | cut -f1))"
        rm -rf "$SUBS"
    else
        echo "CLEAN_SUBSAMPLES=1: keeping $SUBS (it existed before this task)"
    fi
fi
echo "=== $SIM done $(date)"

#!/bin/bash
# HOD catalogs for a few table rows (default: row 1003) on every
# AbacusSummit_base_c000 phase box, ph000-ph024: one array task per phase.
# Each task runs prepare_sim for its box (skipped when that box's subsamples
# already exist), then generates the rows.
#
#   bash slurm/check_phases.sh                          # first: which boxes are on disk
#   sbatch slurm/perlmutter_phases_hod_array.sbatch      # all 25 phases
#   sbatch --array=1-24 slurm/perlmutter_phases_hod_array.sbatch     # e.g. skip ph000
#   sbatch --export=ALL,ROWS="1003 2155" slurm/perlmutter_phases_hod_array.sbatch
#
# Output, per phase NNN (task id = phase number):
#   $OUTROOT/phNNN/catalogs/<row>.npy            galaxy catalogs (as for ph000)
#   $OUTROOT/phNNN/lrg_hods_rows<a>-<b>.hdf5     metadata (sim_name, box_size, ...)
#   $OUTROOT/phNNN/config_<sim>.yaml             the config this box ran with
# Subsamples go to <subsample_dir>/<sim>/z0.500/ from the config: one directory
# per box, so boxes never collide. Check the size of one box's subsamples before
# running all 25:  du -sh <subsample_dir>/AbacusSummit_base_c000_ph000
#
# Settings (--export=ALL,VAR=...):
#   CONFIG           base config (default config/lrg_hod.yaml); only sim_name changes
#   OUTROOT          default $PSCRATCH/lrg_hods_phases
#   ROWS             table rows (default 1003)
#   COSMO            default c000
#   FORCE_PREPARE=1  rerun prepare_sim even if the box's subsamples exist
#   CLEAN_SUBSAMPLES=1  delete the subsamples THIS task created, once the rows are
#                    written (frees scratch; more rows later need prepare_sim again).
#                    Subsamples that existed before the task (e.g. ph000) are kept.
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

# Resolves REPO_DIR (cd's there) and activates $CONDA_ENV (the HOD env), or
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

CONFIG="${CONFIG:-config/lrg_hod.yaml}"
OUTROOT="${OUTROOT:-$PSCRATCH/lrg_hods_phases}"
ROWS="${ROWS:-1003}"
COSMO="${COSMO:-c000}"
PH="$(printf '%03d' "$((10#${PHASE:-${SLURM_ARRAY_TASK_ID:?run as an array job, or set PHASE=}}))")"
SIM="AbacusSummit_base_${COSMO}_ph${PH}"
OUT="$OUTROOT/ph$PH"
mkdir -p "$OUT"

# sim_dir, subsample_dir, z_mock and cleaned_halos as the config sets them
read -r SIM_DIR SUB_DIR ZDIR CLEANED < <(python - "$CONFIG" <<'EOF'
import sys, yaml
s = yaml.safe_load(open(sys.argv[1]))["sim_params"]
print(s["sim_dir"].rstrip("/"), s["subsample_dir"].rstrip("/"), "z%.3f" % float(s.get("z_mock", 0.5)),
      int(bool(s.get("cleaned_halos", True))))
EOF
)
[[ -n "${SIM_DIR:-}" && -n "${SUB_DIR:-}" ]] || { echo "ERROR: could not read sim_dir/subsample_dir from $CONFIG" >&2; exit 2; }
echo "=== $SIM  (task ${SLURM_ARRAY_TASK_ID:-local}, $(date))"
echo "rows: $ROWS   config: $CONFIG   output: $OUT"

# 1. Is the box on disk? (some AbacusSummit boxes are on HPSS tape only)
missing=0
for d in halo_info halo_rv_A field_rv_A; do
    [[ -d "$SIM_DIR/$SIM/halos/$ZDIR/$d" ]] || { echo "MISSING: $SIM_DIR/$SIM/halos/$ZDIR/$d"; missing=1; }
done
if [[ $CLEANED == 1 && ! -d "$SIM_DIR/cleaning/$SIM/$ZDIR" ]]; then
    echo "MISSING: $SIM_DIR/cleaning/$SIM/$ZDIR (cleaned_halos is True in the config)"; missing=1
fi
(( missing )) && { echo "ERROR: $SIM is not fully on disk; skipping it (see slurm/check_phases.sh)." >&2; exit 3; }

# 2. prepare_sim, unless this box's subsamples are already there
SUBS="$SUB_DIR/$SIM/$ZDIR"
prepared_here=0
if [[ -d "$SUBS" && -n "$(ls -A "$SUBS" 2>/dev/null)" && "${FORCE_PREPARE:-0}" != 1 ]]; then
    echo "subsamples exist ($SUBS): skipping prepare_sim (FORCE_PREPARE=1 to redo)"
else
    t0=$SECONDS
    python generate_lrg_hods.py --prepare_only --path2config "$CONFIG" --sim_name "$SIM" \
        --output "$OUT/lrg_hods.hdf5"
    prepared_here=1
    if [[ -d "$SUBS" ]]; then
        touch "$SUBS/.prepared_by_phases_job"
        echo "prepare_sim done in $(( (SECONDS - t0) / 60 )) min; subsamples: $(du -sh "$SUBS" | cut -f1)"
    else
        echo "WARNING: prepare_sim finished but $SUBS does not exist (different layout?); CLEAN_SUBSAMPLES is disabled"
        prepared_here=0
    fi
fi

# 3. the HOD rows
for r in $ROWS; do
    python generate_lrg_hods.py --path2config "$CONFIG" --sim_name "$SIM" \
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

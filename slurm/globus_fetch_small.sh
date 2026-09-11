#!/bin/bash
# Fetch ONE AbacusSummit small box at ONE redshift from the public AbacusSummit
# Globus collection into your Perlmutter scratch -- only the products that
# AbacusHOD's prepare_sim actually reads (a few GB, not the whole box).
#
#   bash slurm/globus_fetch_small.sh                       # defaults below
#   SIM=AbacusSummit_small_c000_ph3001 bash slurm/globus_fetch_small.sh
#   bash slurm/globus_fetch_small.sh --list                # just show available small phases
#   bash slurm/globus_fetch_small.sh --dry-run             # show the batch, submit nothing
#
# One-time setup (login node, inside your conda env):
#   pip install globus-cli
#   globus login            # opens a browser link; paste the code back
#
# What gets transferred, mirroring the official tree so abacusutils'
# cleaned-catalog auto-detection works unchanged:
#
#   small/<SIM>/halos/z<Z>/halo_info/            <- prepare_sim reads these three
#   small/<SIM>/halos/z<Z>/halo_rv_A/
#   small/<SIM>/halos/z<Z>/field_rv_A/
#   cleaning/small/<SIM>/z<Z>/cleaned_halo_info/ <- needed for cleaned_halos: True
#   cleaning/small/<SIM>/z<Z>/cleaned_rvpid/
#
# Afterwards point config/lrg_hod_small.yaml at the copy:
#   sim_dir: '<DEST_PATH>/small/'        e.g. '/pscratch/sd/j/jsull/AbacusSummit/small/'
#
# Collections are found by display name at runtime (docs UUIDs are not
# hardcoded here).  Override with SRC_ID= / DST_ID= if the search is ambiguous.

set -euo pipefail

LIST_ONLY=0; DRY_RUN=0
for a in "$@"; do case "$a" in --list) LIST_ONLY=1;; --dry-run) DRY_RUN=1;; -h|--help) sed -n '2,32p' "$0"; exit 0;; esac; done

_user="${USER:-$(id -un)}"
SIM="${SIM:-AbacusSummit_small_c000_ph3000}"
Z="${Z:-0.500}"
DEST_PATH="${DEST_PATH:-${PSCRATCH:-/pscratch/sd/${_user:0:1}/$_user}/AbacusSummit}"
SRC_NAME="${SRC_NAME:-AbacusSummit}"          # public collection display name
DST_NAME="${DST_NAME:-NERSC Perlmutter}"      # reaches /pscratch directly

command -v globus >/dev/null || { echo "ERROR: globus CLI not found.  pip install globus-cli && globus login" >&2; exit 1; }
globus whoami >/dev/null 2>&1 || { echo "ERROR: not logged in.  Run: globus login" >&2; exit 1; }

# --- resolve collections by name unless UUIDs were given -------------------
resolve() {   # $1 = display name -> prints UUID, or fails listing candidates
    local name="$1"
    local hits
    hits="$(globus endpoint search "$name" --filter-scope all --format unix --jmespath 'DATA[].[id,display_name,owner_string]' 2>/dev/null || true)"
    local exact
    exact="$(printf '%s\n' "$hits" | awk -F'\t' -v n="$name" '$2==n {print $1}' | head -1)"
    if [[ -n "$exact" ]]; then echo "$exact"; return 0; fi
    echo "ERROR: no collection with exact display name '$name'.  Candidates:" >&2
    printf '%s\n' "$hits" | awk -F'\t' '{printf "   %s   %-40s %s\n", $1, $2, $3}' >&2
    echo "       Re-run with SRC_ID=<uuid> or DST_ID=<uuid> set explicitly." >&2
    return 1
}
SRC_ID="${SRC_ID:-$(resolve "$SRC_NAME")}"
DST_ID="${DST_ID:-$(resolve "$DST_NAME")}"

echo "Source      : $SRC_NAME  ($SRC_ID)"
echo "Destination : $DST_NAME  ($DST_ID)  path $DEST_PATH"
echo "Simulation  : $SIM   z=$Z"
echo

# --- inspect the source: which small phases exist, and does ours? ----------
echo "--- small boxes on the source collection (first 20) ---"
globus ls "$SRC_ID:/small/" --format unix 2>/dev/null | head -20 || {
    echo "ERROR: cannot list $SRC_ID:/small/ -- the collection may need activation:" >&2
    echo "       https://app.globus.org/file-manager?origin_id=$SRC_ID" >&2; exit 1; }
NPH="$(globus ls "$SRC_ID:/small/" --format unix 2>/dev/null | grep -c AbacusSummit_small || true)"
echo "... $NPH small boxes total"
[[ $LIST_ONLY == 1 ]] && exit 0

HALOS="small/$SIM/halos/z$Z"
CLEAN="cleaning/small/$SIM/z$Z"
echo
echo "--- verifying required directories exist on the source ---"
missing=0
for d in "$HALOS/halo_info" "$HALOS/halo_rv_A" "$HALOS/field_rv_A" "$CLEAN/cleaned_halo_info" "$CLEAN/cleaned_rvpid"; do
    if globus ls "$SRC_ID:/$d/" >/dev/null 2>&1; then
        n="$(globus ls "$SRC_ID:/$d/" --format unix 2>/dev/null | wc -l)"
        printf '  ok       %-55s (%s files)\n' "$d" "$n"
    else
        printf '  MISSING  %s\n' "$d"; missing=1
    fi
done
if [[ $missing == 1 ]]; then
    echo >&2
    echo "Some directories are missing.  If only the cleaning/ ones are absent you can" >&2
    echo "set cleaned_halos: False in the config; otherwise pick another phase" >&2
    echo "(bash $0 --list) -- ph3000..ph4999 are irregularly numbered." >&2
    exit 1
fi

# --- build the batch (one recursive line per directory) --------------------
BATCH="$(mktemp)"
for d in "$HALOS/halo_info" "$HALOS/halo_rv_A" "$HALOS/field_rv_A" "$CLEAN/cleaned_halo_info" "$CLEAN/cleaned_rvpid"; do
    echo "--recursive /$d/ $DEST_PATH/$d/" >> "$BATCH"
done
echo
echo "--- transfer batch ---"; cat "$BATCH"

if [[ $DRY_RUN == 1 ]]; then echo; echo "(dry run: nothing submitted)"; rm -f "$BATCH"; exit 0; fi

# --- submit --------------------------------------------------------------
echo
TASK="$(globus transfer "$SRC_ID" "$DST_ID" \
          --batch "$BATCH" \
          --label "AbacusSummit $SIM z$Z for AbacusHOD" \
          --sync-level size --preserve-mtime --notify off \
          --format unix --jmespath task_id)"
rm -f "$BATCH"
echo "Submitted Globus task: $TASK"
echo
echo "Monitor:   globus task show $TASK"
echo "           globus task wait $TASK --timeout 3600 && echo DONE"
echo "Web:       https://app.globus.org/activity/$TASK"
echo
echo "When complete, set in config/lrg_hod_small.yaml:"
echo "    sim_dir: '$DEST_PATH/small/'"
echo "then:  sbatch slurm/perlmutter_prepare_sim_small.sbatch"

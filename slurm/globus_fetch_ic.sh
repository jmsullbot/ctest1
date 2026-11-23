#!/bin/bash
# Fetch the initial-conditions density field for ONE AbacusSummit box from the
# public AbacusSummit Globus collection into your Perlmutter scratch.
#
#   bash slurm/globus_fetch_ic.sh                        # base_c000_ph000, ic_dens_N576.asdf
#   FILES="ic_dens_N576.asdf ic_dens_N1152.asdf" bash slurm/globus_fetch_ic.sh
#   SIM=AbacusSummit_base_c000_ph001 bash slurm/globus_fetch_ic.sh
#   PHASES=0-24 bash slurm/globus_fetch_ic.sh            # all 25 c000 phases, ONE transfer
#   PHASES="3 7 12" COSMO=c000 bash slurm/globus_fetch_ic.sh
#   bash slurm/globus_fetch_ic.sh --list                 # just show what is in ic/
#   bash slurm/globus_fetch_ic.sh --dry-run
#
# PHASES ("A-B" or a list) builds AbacusSummit_base_<COSMO>_phNNN names; SIMS
# takes explicit names. Files already at the destination with the same size are
# skipped (--sync-level size), so rerunning only fetches what is missing.
#
# One-time setup (login node, inside your conda env):
#   pip install globus-cli && globus login
#
# NEVER copies the whole ic/ directory: alongside the few-GB density fields
# it holds the Zel'dovich displacement files for all 6912^3 particles, which
# are terabytes.  Only the files named in FILES are transferred, and their
# sizes are printed before anything is submitted.
#
# Destination layout mirrors the official tree:
#   <DEST_PATH>/<SIM>/ic/ic_dens_N576.asdf

set -euo pipefail

LIST_ONLY=0; DRY_RUN=0
for a in "$@"; do case "$a" in --list) LIST_ONLY=1;; --dry-run) DRY_RUN=1;; -h|--help) sed -n '2,20p' "$0"; exit 0;; esac; done

_user="${USER:-$(id -un)}"
COSMO="${COSMO:-c000}"
if [[ -n "${SIMS:-}" ]]; then
    :
elif [[ -n "${PHASES:-}" ]]; then
    if [[ "$PHASES" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        plist="$(seq "$((10#${BASH_REMATCH[1]}))" "$((10#${BASH_REMATCH[2]}))")"
    else
        plist="$PHASES"
    fi
    SIMS=""; for p in $plist; do SIMS+="AbacusSummit_base_${COSMO}_ph$(printf '%03d' "$((10#$p))") "; done
else
    SIMS="${SIM:-AbacusSummit_base_c000_ph000}"
fi
read -r -a SIM_LIST <<< "$SIMS"
SIM="${SIM_LIST[0]}"
FILES="${FILES:-ic_dens_N576.asdf}"
DEST_PATH="${DEST_PATH:-${PSCRATCH:-/pscratch/sd/${_user:0:1}/$_user}/AbacusSummit}"
SRC_NAME="${SRC_NAME:-AbacusSummit}"
DST_NAME="${DST_NAME:-NERSC Perlmutter}"

command -v globus >/dev/null || { echo "ERROR: globus CLI not found.  pip install globus-cli && globus login" >&2; exit 1; }
globus whoami >/dev/null 2>&1 || { echo "ERROR: not logged in.  Run: globus login" >&2; exit 1; }

resolve() {
    local name="$1" hits exact
    hits="$(globus endpoint search "$name" --filter-scope all --format unix --jmespath 'DATA[].[id,display_name,owner_string]' 2>/dev/null || true)"
    exact="$(printf '%s\n' "$hits" | awk -F'\t' -v n="$name" '$2==n {print $1}' | head -1)"
    [[ -n "$exact" ]] && { echo "$exact"; return 0; }
    echo "ERROR: no collection with exact display name '$name'.  Candidates:" >&2
    printf '%s\n' "$hits" | awk -F'\t' '{printf "   %s   %-40s %s\n", $1, $2, $3}' >&2
    echo "       Re-run with SRC_ID=<uuid> or DST_ID=<uuid>." >&2
    return 1
}
SRC_ID="${SRC_ID:-$(resolve "$SRC_NAME")}"
DST_ID="${DST_ID:-$(resolve "$DST_NAME")}"

echo "Source      : $SRC_NAME  ($SRC_ID)"
echo "Destination : $DST_NAME  ($DST_ID)  -> $DEST_PATH/<sim>/ic/"
echo "Simulations : ${#SIM_LIST[@]} (${SIM_LIST[0]}${SIM_LIST[1]:+ ... ${SIM_LIST[${#SIM_LIST[@]}-1]}})"
echo

# The base boxes sit at the collection root; small boxes under small/
rel_of() { if [[ "$1" == *_small_* ]]; then echo "small/$1/ic"; else echo "$1/ic"; fi; }
REL="$(rel_of "$SIM")"

echo "--- contents of $REL on the source (name, size) ---"
if ! globus ls "$SRC_ID:/$REL/" --long --format unix 2>/dev/null | awk -F'\t' '{printf "  %-40s %12s bytes\n", $NF, $4}'; then
    echo "ERROR: cannot list $SRC_ID:/$REL/ -- wrong path, or the collection needs activating:" >&2
    echo "       https://app.globus.org/file-manager?origin_id=$SRC_ID" >&2
    exit 1
fi
(( ${#SIM_LIST[@]} > 1 )) && echo "  (listing shown for the first box only)"
[[ $LIST_ONLY == 1 ]] && exit 0

echo
echo "--- verifying requested files exist ---"
BATCH="$(mktemp)"; missing=0
for s in "${SIM_LIST[@]}"; do
    rel="$(rel_of "$s")"
    for f in $FILES; do
        if globus ls "$SRC_ID:/$rel/$f" >/dev/null 2>&1; then
            printf '  ok       %s/%s\n' "$s" "$f"; echo "/$rel/$f $DEST_PATH/$rel/$f" >> "$BATCH"
        else
            printf '  MISSING  %s/%s\n' "$s" "$f"; missing=1
        fi
    done
done
if [[ $missing == 1 ]]; then
    echo "Some files are absent on the source (see above)." >&2
    if [[ "${ALLOW_MISSING:-0}" != 1 ]]; then
        echo "Pick others via FILES=..., drop those boxes from PHASES, or ALLOW_MISSING=1 to fetch the rest." >&2
        rm -f "$BATCH"; exit 1
    fi
fi
[[ -s "$BATCH" ]] || { echo "Nothing to transfer." >&2; rm -f "$BATCH"; exit 1; }

echo; echo "--- transfer batch ($(wc -l < "$BATCH") files) ---"; cat "$BATCH"
[[ $DRY_RUN == 1 ]] && { echo; echo "(dry run: nothing submitted)"; rm -f "$BATCH"; exit 0; }

echo
TASK="$(globus transfer "$SRC_ID" "$DST_ID" --batch "$BATCH" \
          --label "AbacusSummit IC density ${#SIM_LIST[@]} boxes" --sync-level size --notify off \
          --format unix --jmespath task_id)"
rm -f "$BATCH"
echo "Submitted Globus task: $TASK"
echo "  globus task wait $TASK --timeout 14400 && ls -la $DEST_PATH/*/ic/"
echo "  https://app.globus.org/activity/$TASK"

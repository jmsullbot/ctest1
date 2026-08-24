#!/bin/bash
# Grant another NERSC user read access to a directory tree.
#
#   bash slurm/share_dir.sh <username> <directory> [more users...]
#
# Examples:
#   bash slurm/share_dir.sh alice $PSCRATCH/lrg_hods
#   bash slurm/share_dir.sh alice,bob $PSCRATCH/lrg_hods
#   bash slurm/share_dir.sh alice $PSCRATCH/lrg_hods --check   # diagnose only
#
# What it does, and why:
#
#   1. Adds a u:<user>:rx ACL to EVERY ancestor directory you own, up to /.
#      Unix requires execute permission on every component of a path; a
#      recursive chmod/setfacl on the target alone is not enough, and
#      $PSCRATCH/<user> is created 0700 by NERSC.  Only 'x' is strictly
#      needed on ancestors, but 'rx' is applied so they can also list.
#
#   2. Adds u:<user>:rX recursively inside the target (capital X = execute
#      on directories only, never on plain files).
#
#   3. Adds a *default* ACL so files created later inherit the access.
#
#   4. Repairs the ACL mask.  Running chmod on a file that has ACLs
#      rewrites the mask and can silently neuter every ACL entry -- getfacl
#      then shows "#effective:" lines with reduced permissions.  This is the
#      most common reason "I already ran setfacl" does not work.
#
#   5. Verifies the result and prints exactly which component still blocks
#      access, if any.
#
# Uses user ACLs rather than group ACLs on purpose: group-based access
# depends on the other user's login session having the group resolved, so a
# recently-added group member can still be denied until they log out and in.
#
# NOTE: $PSCRATCH is purged (~8 weeks at NERSC) and is not backed up.  For
# durable sharing, put the data on CFS (/global/cfs/cdirs/<project>/), which
# is designed for group access.

set -uo pipefail          # NOT -e: we want to report failures, not abort

# --- arguments -------------------------------------------------------------
CHECK_ONLY=0
ARGS=()
for a in "$@"; do
    case "$a" in
        --check|-c) CHECK_ONLY=1 ;;
        -h|--help)  sed -n '2,40p' "$0"; exit 0 ;;
        *)          ARGS+=("$a") ;;
    esac
done

if [[ ${#ARGS[@]} -lt 2 ]]; then
    echo "usage: bash $0 <username[,username...]> <directory> [--check]" >&2
    echo "       bash $0 --help" >&2
    exit 2
fi

USERS_RAW="${ARGS[0]}"
TARGET_RAW="${ARGS[1]}"

IFS=',' read -r -a USERS <<< "$USERS_RAW"

if [[ ! -d "$TARGET_RAW" ]]; then
    echo "ERROR: '$TARGET_RAW' is not a directory (or you cannot see it)." >&2
    exit 1
fi
TARGET="$(cd "$TARGET_RAW" && pwd -P)"      # resolve symlinks; ACLs follow the real path

ME="$(id -un)"

echo "============================================================"
echo " Sharing : $TARGET"
echo " With    : ${USERS[*]}"
echo " As      : $ME"
[[ $CHECK_ONLY == 1 ]] && echo " Mode    : CHECK ONLY (no changes)"
echo "============================================================"

# Sanity-check the usernames exist before touching anything
for u in "${USERS[@]}"; do
    if ! id "$u" >/dev/null 2>&1; then
        echo "ERROR: no such user '$u' -- check the spelling of their NERSC login." >&2
        exit 1
    fi
done

# Does this filesystem support ACLs at all?
if ! command -v setfacl >/dev/null 2>&1; then
    echo "ERROR: setfacl not found on this system." >&2
    exit 1
fi

# --- build the ancestor list (target first, then upward to /) --------------
ancestors=()
d="$TARGET"
while [[ "$d" != "/" && -n "$d" ]]; do
    ancestors+=("$d")
    d="$(dirname "$d")"
done

# --- 1-4. apply ------------------------------------------------------------
if [[ $CHECK_ONLY == 0 ]]; then
    echo
    echo "--- Granting traversal on ancestors (rx) ---"
    # Walk from / downward so parents are opened before children
    for (( i=${#ancestors[@]}-1 ; i>=0 ; i-- )); do
        d="${ancestors[$i]}"
        owner="$(stat -c '%U' "$d" 2>/dev/null)"
        if [[ "$owner" != "$ME" ]]; then
            printf '  %-8s %s (owned by %s)\n' "skip" "$d" "${owner:-?}"
            continue
        fi
        ok=1
        for u in "${USERS[@]}"; do
            setfacl -m "u:$u:rx" "$d" 2>/dev/null || ok=0
        done
        # Ensure the mask does not mask the entries we just added
        setfacl -m "m::rx" "$d" 2>/dev/null || true
        [[ $ok == 1 ]] && printf '  %-8s %s\n' "ok" "$d" \
                       || printf '  %-8s %s (setfacl failed)\n' "FAIL" "$d"
    done

    echo
    echo "--- Granting read access inside the tree (recursive) ---"
    echo "    (this can take a while on large trees)"
    for u in "${USERS[@]}"; do
        setfacl -R  -m "u:$u:rX" "$TARGET" 2>/dev/null \
            && echo "  ok       recursive ACL for $u" \
            || echo "  FAIL     recursive ACL for $u"
        setfacl -R -d -m "u:$u:rX" "$TARGET" 2>/dev/null \
            && echo "  ok       default ACL for $u (future files inherit)" \
            || echo "  warn     default ACL for $u not set"
    done
    # Repair masks throughout, in case a previous chmod -R clobbered them
    setfacl -R -m "m::rX" "$TARGET" 2>/dev/null || true
fi

# --- 5. verify -------------------------------------------------------------
echo
echo "--- Verification ---"

blocked=0
for u in "${USERS[@]}"; do
    echo
    echo "  User: $u"
    for (( i=${#ancestors[@]}-1 ; i>=0 ; i-- )); do
        d="${ancestors[$i]}"

        perms="$(stat -c '%A' "$d" 2>/dev/null)"
        other_x="${perms:9:1}"          # world execute bit

        # Effective ACL for this user, if any
        acl_line="$(getfacl -p "$d" 2>/dev/null | grep -E "^user:$u:")"
        eff="$(getfacl -p "$d" 2>/dev/null | grep -E "^user:$u:" \
               | grep -o '#effective:[^ ]*' | cut -d: -f2)"

        verdict=""
        if [[ -n "$acl_line" ]]; then
            granted="${eff:-$(echo "$acl_line" | cut -d: -f3)}"
            if [[ "$granted" == *x* ]]; then
                verdict="OK   via ACL ($granted)"
            else
                verdict="BLOCK ACL present but no x (effective '$granted' -- mask?)"
                blocked=1
            fi
        elif [[ "$other_x" == "x" ]]; then
            verdict="OK   via world-execute"
        else
            # group check
            gname="$(stat -c '%G' "$d" 2>/dev/null)"
            grp_x="${perms:6:1}"
            if [[ "$grp_x" == "x" ]] && id -nG "$u" 2>/dev/null | tr ' ' '\n' | grep -qx "$gname"; then
                verdict="OK   via group '$gname'"
            else
                verdict="BLOCK no ACL, no world-x, no group-x"
                blocked=1
            fi
        fi
        printf '    %-11s %-52s %s\n' "$perms" "$d" "$verdict"
    done
done

echo
if [[ $blocked == 0 ]]; then
    echo "RESULT: every path component is traversable by ${USERS[*]}."
    echo
    echo "Ask them to confirm with:"
    echo "    ls $TARGET | head"
    echo
    echo "If they STILL get permission denied, have them run 'id' -- a"
    echo "recently-granted group needs a fresh login to take effect, though"
    echo "the user ACLs set here should not depend on that."
else
    echo "RESULT: at least one component still blocks access (marked BLOCK)."
    echo
    echo "If the blocking directory is not owned by you, you cannot fix it"
    echo "yourself -- that path simply cannot be shared this way.  Options:"
    echo "  * put the data on CFS:  /global/cfs/cdirs/<project>/"
    echo "  * hand over single files with:  give -u <user> <file>"
    echo "  * ask consult@nersc.gov if a system-owned directory is at fault"
fi

echo
echo "Reminder: \$PSCRATCH is purged (~8 weeks) and not backed up."

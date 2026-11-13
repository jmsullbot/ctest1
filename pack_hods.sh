#!/bin/bash
# Bundle lrgs.py results (base 256, base 512, small 256; real and redshift
# space) and the HOD catalogs for a list of table rows, ready to download.
#
#   bash pack_hods.sh                    # the row list below
#   bash pack_hods.sh 570 1003 2155      # or any rows on the command line
#
# Writes two archives to OUTDIR (default $PSCRATCH/downloads):
#   hod_results_<tag>.tar.gz   results, each row's parameters, README   (a few MB)
#   hod_catalogs_<tag>.tar     base and small catalogs (sizes printed first;
#                              a base catalog can be several GB)
# CATALOGS=0 skips the catalog archive; BASE_CATALOGS=0 keeps only the small
# ones. Anything missing is reported (also in MANIFEST.txt); everything that
# exists is still packed. Paths: BASE_RES, SMALL_RES, BASE_CAT, SMALL_CAT.
set -euo pipefail

ROWS_DEFAULT=(570 1003 2155 2254 2340 2468 2993 3128 3332 3568 4259 4597 5339 5371
              5477 6155 6336 7377 7547 8924 8953 8993 9009 9495 9507 9993 10386)
if (( $# )); then ROWS=("$@"); else ROWS=("${ROWS_DEFAULT[@]}"); fi
for r in "${ROWS[@]}"; do
    [[ $r =~ ^[0-9]+$ ]] || { echo "ERROR: row '$r' is not a number" >&2; exit 2; }
done

PS="${PSCRATCH:?PSCRATCH is not set}"
BASE_RES="${BASE_RES:-$PS/qso_sbp/Heavy_files_Andrej/AbacusBase/results}"
SMALL_RES="${SMALL_RES:-$PS/qso_sbp/Heavy_files_Andrej/AbacusSmall/results}"
BASE_CAT="${BASE_CAT:-$PS/lrg_hods/catalogs}"
SMALL_CAT="${SMALL_CAT:-$PS/lrg_hods_small/catalogs}"
OUTDIR="${OUTDIR:-$PS/downloads}"
TAG="${TAG:-${#ROWS[@]}hods_$(date +%Y%m%d)}"
CATALOGS="${CATALOGS:-1}"; BASE_CATALOGS="${BASE_CATALOGS:-1}"

mkdir -p "$OUTDIR"
STAGE="$(mktemp -d "$PS/.hodpack.XXXXXX")"     # symlinks only; tar -h copies the data
trap 'rm -rf "$STAGE"' EXIT
R="$STAGE/hod_results_$TAG"; C="$STAGE/hod_catalogs_$TAG"
mkdir -p "$R"/{base_Nmesh256,base_Nmesh512,small_Nmesh256} "$C"/{base,small}
shopt -s nullglob
missing=()

take() {   # $1 dest dir, $2 description, $3 glob or plain path
    local hits=( $3 )      # a plain path survives nullglob even when absent: test -e
    if (( ${#hits[@]} == 0 )) || [[ ! -e ${hits[0]} ]]; then missing+=("$2"); return; fi
    ln -s "${hits[0]}" "$1/"
}

for r in "${ROWS[@]}"; do
    take "$R/base_Nmesh256"  "row $r base 256 real"   "$BASE_RES/results_AbacusBase_*hod=${r}_*Nmesh_256.npy"
    take "$R/base_Nmesh256"  "row $r base 256 RSD"    "$BASE_RES/results_RSD_AbacusBase_*hod=${r}_*Nmesh_256_axisrsd_2.npy"
    take "$R/base_Nmesh512"  "row $r base 512 real"   "$BASE_RES/results_AbacusBase_*hod=${r}_*Nmesh_512.npy"
    take "$R/base_Nmesh512"  "row $r base 512 RSD"    "$BASE_RES/results_RSD_AbacusBase_*hod=${r}_*Nmesh_512_axisrsd_2.npy"
    take "$R/small_Nmesh256" "row $r small 256 real"  "$SMALL_RES/results_AbacusSmall_*hod=${r}_*Nmesh_256.npy"
    take "$R/small_Nmesh256" "row $r small 256 RSD"   "$SMALL_RES/results_RSD_AbacusSmall_*hod=${r}_*Nmesh_256_axisrsd_2.npy"
    cat6="$(printf '%06d' "$r").npy"
    take "$C/small" "row $r small catalog" "$SMALL_CAT/$cat6"
    if [[ $BASE_CATALOGS == 1 ]]; then take "$C/base" "row $r base catalog" "$BASE_CAT/$cat6"; fi
done

# each row's HOD parameters, n_gal, nbar, f_sat (from hod_summary.py, if it was run)
for pair in "base:$PS/lrg_hods" "small:$PS/lrg_hods_small"; do
    name="${pair%%:*}"; dir="${pair#*:}"
    if [[ -f "$dir/hod_summary.csv" ]]; then
        awk -F, -v rows="${ROWS[*]}" 'BEGIN { n = split(rows, a, " "); for (i = 1; i <= n; i++) want[a[i]] = 1 }
                                      NR == 1 || ($1 in want)' "$dir/hod_summary.csv" > "$R/params_$name.csv"
    else
        missing+=("params_$name.csv (no $dir/hod_summary.csv; run: python hod_summary.py $dir)")
    fi
done

cat > "$R/README.txt" <<'EOF'
lrgs.py results and HOD catalogs for selected rows of lrg_params_eq74.npy
(z = 0.5, AbacusSummit c000; base box 2000 Mpc/h ph000, small box 500 Mpc/h ph3000).
Row NNNN is the same HOD in every directory.

base_Nmesh256/   base box, lrgs.py at Nmesh = 256 (k_Nyq = 0.40 h/Mpc)
                 CAVEAT: these shifted fields used a much weaker cubic operator (d3)
                 than base_Nmesh512 and small_Nmesh256, and k >~ 0.25 is affected by
                 the mesh. Prefer base_Nmesh512.
base_Nmesh512/   base box, Nmesh = 512 (k_Nyq = 0.80); reliable to k ~ 0.5-0.6
small_Nmesh256/  small box, Nmesh = 256 (k_Nyq = 1.61)

Each results file is a pickled object array:  np.load(f, allow_pickle=True)
  real:  k, P_err, P_true, P_model, r, beta1, beta2, betaG2, beta3, nbar
  RSD:   k, mu_centers, P_err(k,mu), P_true(k,mu), P_model(k,mu), r, ..., nbar
  (axisrsd_2: line of sight along z)

params_base.csv / params_small.csv: row, the HOD parameters, logsigma, n_gal,
nbar [(h/Mpc)^3], n_cent, f_sat for these rows (n_gal and nbar differ by box).

Catalogs (other archive): catalogs/{base,small}/NNNNNN.npy, structured arrays with
fields x, y, z, z_rsd, vx, vy, vz, mass, id.  Positions in Mpc/h are centred on 0
(range -L/2..L/2; lrgs.py adds L/2 before painting); z_rsd is z displaced along the
line of sight (z axis). The first n_cent galaxies are centrals, the rest satellites.
EOF

# ---------------------------------------------------------------- report ---
bytes() { find -L "$1" -type f -printf '%s\n' | awk '{s += $1} END {print s + 0}'; }
human() { awk -v b="$1" 'BEGIN { split("B KB MB GB TB", u); i = 1; while (b >= 1024 && i < 5) { b /= 1024; i++ }; printf "%.1f %s", b, u[i] }'; }
nres=$(find -L "$R" -name 'results_*.npy' | wc -l)
{
    echo "rows (${#ROWS[@]}): ${ROWS[*]}"
    echo "results files: $nres of $(( ${#ROWS[@]} * 6 ))"
    echo "catalogs: $(find -L "$C" -name '*.npy' | wc -l)"
    echo "missing (${#missing[@]}):"
    for m in "${missing[@]+"${missing[@]}"}"; do echo "  $m"; done
} > "$R/MANIFEST.txt"
cat "$R/MANIFEST.txt"

tar -C "$STAGE" -czhf "$OUTDIR/hod_results_$TAG.tar.gz" "hod_results_$TAG"
echo
echo "results : $OUTDIR/hod_results_$TAG.tar.gz  ($(human "$(stat -c %s "$OUTDIR/hod_results_$TAG.tar.gz")"))"

if [[ $CATALOGS == 1 ]]; then
    echo "catalogs: small $(human "$(bytes "$C/small")"), base $(human "$(bytes "$C/base")") -- writing the tar..."
    tar -C "$STAGE" -chf "$OUTDIR/hod_catalogs_$TAG.tar" "hod_catalogs_$TAG"
    echo "catalogs: $OUTDIR/hod_catalogs_$TAG.tar  ($(human "$(stat -c %s "$OUTDIR/hod_catalogs_$TAG.tar")"))"
fi

echo
me="${USER:-$(id -un)}"
echo "download (on your own machine):"
echo "  scp $me@perlmutter.nersc.gov:$OUTDIR/hod_results_$TAG.tar.gz ."
if [[ $CATALOGS == 1 ]]; then
    echo "  scp $me@perlmutter.nersc.gov:$OUTDIR/hod_catalogs_$TAG.tar .     (or Globus if it is large)"
    echo "unpack: tar xzf hod_results_$TAG.tar.gz ; tar xf hod_catalogs_$TAG.tar"
else
    echo "unpack: tar xzf hod_results_$TAG.tar.gz"
fi

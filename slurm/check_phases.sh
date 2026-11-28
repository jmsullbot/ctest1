#!/bin/bash
# Status of every AbacusSummit_base_c000 phase box for the multi-phase run made
# with the fixed abacusutils (local-environment fix, PR #207): is the box on
# disk, are its subsamples / IC / catalog / shifted fields / lrgs.py results there.
# Run it before submitting anything, and again to follow progress.
#
#   bash slurm/check_phases.sh                  # ph000-ph024, row 1003, Nmesh 512
#   PHASES="0 3 7" ROW=2155 NMESH=256 bash slurm/check_phases.sh
#   OUTROOT=$PSCRATCH/lrg_hods_phases bash slurm/check_phases.sh    # the earlier, unfixed run
#
# Columns:
#   halos   halo_info, halo_rv_A, field_rv_A present at z_mock (what prepare_sim reads)
#   clean   cleaning/<sim>/<z>
#   subs    prepare_sim output in SUBSAMPLE_DIR/<sim>/<z>: ok = one halo, particle
#           and environment file per slab; old = no environment files (made by
#           the unfixed code); part = incomplete
#   IC      <IC_ROOT>/<sim>/ic/<IC_FILE>
#   cat     the row's catalog in OUTROOT/phNNN/catalogs/
#   fields  shifted fields at NMESH (real space and redshift-space axis 2, full
#           size): ok = in OUTROOT/phNNN/shifted_fields; link = made earlier, the
#           lrgs job will link them (FIELDS_FROM/phNNN/shifted_fields, or
#           PH000_FIELDS for ph000)
#   lrgs    lrgs.py results for the row (real and RSD) in OUTROOT/phNNN/results/
# sim_dir and z_mock come from CONFIG; override the rest with OUTROOT,
# SUBSAMPLE_DIR, IC_ROOT, IC_FILE, FIELDS_FROM, PH000_FIELDS.
set -uo pipefail

CONFIG="${CONFIG:-config/lrg_hod.yaml}"
OUTROOT="${OUTROOT:-${PSCRATCH:-.}/lrg_hods_phases_envfix}"
SUBSAMPLE_DIR="${SUBSAMPLE_DIR:-${PSCRATCH:-.}/lrg_hods_envfix/subsamples}"
FIELDS_FROM="${FIELDS_FROM:-${PSCRATCH:-.}/lrg_hods_phases}"
PH000_FIELDS="${PH000_FIELDS:-${PSCRATCH:-.}/qso_sbp/Heavy_files_Andrej/AbacusBase/shifted_fields}"
IC_ROOT="${IC_ROOT:-${PSCRATCH:-.}/AbacusSummit}"
IC_FILE="${IC_FILE:-ic_dens_N576.asdf}"
ROW="${ROW:-1003}"
NMESH="${NMESH:-512}"
COSMO="${COSMO:-c000}"
PHASES="${PHASES:-$(seq 0 24)}"

SIM_DIR=/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit; ZDIR=z0.500
if [[ -f "$CONFIG" ]] && command -v python >/dev/null; then
    read -r SIM_DIR ZDIR < <(python - "$CONFIG" 2>/dev/null <<'EOF'
import sys, yaml
s = yaml.safe_load(open(sys.argv[1]))["sim_params"]
print(s["sim_dir"].rstrip("/"), "z%.3f" % float(s.get("z_mock", 0.5)))
EOF
) || true
fi
SUB_DIR="${SUBSAMPLE_DIR%/}"
echo "sim_dir $SIM_DIR | subsamples $SUB_DIR | IC $IC_ROOT/<sim>/ic/$IC_FILE | outputs $OUTROOT/phNNN"
echo "row $ROW, Nmesh $NMESH, $ZDIR"
printf '%-6s %-6s %-6s %-6s %-6s %-6s %-6s %-6s\n' phase halos clean subs IC cat fields lrgs
yes_no() { [[ $1 == 1 ]] && echo ok || echo "--"; }
count() { if [[ -d $1 ]]; then find "$1" -maxdepth 1 -name "$2" | wc -l; else echo 0; fi; }   # 0 for a folder not there yet
full() { [[ -f $1 ]] && (( $(stat -L -c %s "$1") >= $2 * NMESH * NMESH * NMESH * 4 )); }   # file, number of grids
fields_in() { full "$1/shifted_fields_real_N_${NMESH}_c_0.npy" 4 && full "$1/shifted_fields_redshift_axisrsd_2_N_${NMESH}_c_0.npy" 6; }
n_ok=0; n_box=0; n_fields=0; n_cat=0; n_subs=0
for p in $PHASES; do
    ph="$(printf '%03d' "$((10#$p))")"; sim="AbacusSummit_base_${COSMO}_ph$ph"
    h=1; for d in halo_info halo_rv_A field_rv_A; do [[ -d "$SIM_DIR/$sim/halos/$ZDIR/$d" ]] || h=0; done
    c=0; [[ -d "$SIM_DIR/cleaning/$sim/$ZDIR" ]] && c=1
    n_slab="$(count "$SIM_DIR/$sim/halos/$ZDIR/halo_info" '*.asdf')"
    n_h="$(count "$SUB_DIR/$sim/$ZDIR" 'halos_xcom_*_new.h5')"
    n_p="$(count "$SUB_DIR/$sim/$ZDIR" 'particles_xcom_*_new.h5')"
    n_e="$(count "$SUB_DIR/$sim/$ZDIR" 'env_xcom_*_abacushod_localenv_new.h5')"
    if (( n_h == 0 && n_p == 0 && n_e == 0 )); then s="--"
    elif (( n_slab > 0 && n_h == n_slab && n_p == n_slab && n_e == n_slab )); then s=ok; n_subs=$((n_subs + 1))
    elif (( n_h > 0 && n_e == 0 )); then s=old
    else s=part; fi
    i=0; [[ -s "$IC_ROOT/$sim/ic/$IC_FILE" ]] && i=1
    k=0; [[ -s "$OUTROOT/ph$ph/catalogs/$(printf '%06d' "$ROW").npy" ]] && k=1
    base="$OUTROOT/ph$ph"
    if [[ $ph == 000 ]]; then src="$PH000_FIELDS"; else src="$FIELDS_FROM/ph$ph/shifted_fields"; fi
    if fields_in "$base/shifted_fields"; then f=ok; n_fields=$((n_fields + 1))
    elif [[ ! -e "$base/shifted_fields" ]] && fields_in "$src"; then f=link; n_fields=$((n_fields + 1))
    else f="--"; fi
    tag="c=0_ph=${ph}_hod=${ROW}_z=0.5000_Nmesh_${NMESH}"
    r=0; [[ -s "$base/results/results_AbacusBase_${tag}.npy" && -s "$base/results/results_RSD_AbacusBase_${tag}_axisrsd_2.npy" ]] && r=1
    printf '%-6s %-6s %-6s %-6s %-6s %-6s %-6s %-6s\n' "ph$ph" "$(yes_no $h)" "$(yes_no $c)" "$s" "$(yes_no $i)" "$(yes_no $k)" "$f" "$(yes_no $r)"
    (( h && c )) && n_box=$((n_box + 1))
    (( k )) && n_cat=$((n_cat + 1))
    (( r )) && n_ok=$((n_ok + 1))
done
echo "boxes fully on disk: $n_box; subsamples: $n_subs; catalogs: $n_cat; shifted fields available: $n_fields; lrgs.py results: $n_ok"

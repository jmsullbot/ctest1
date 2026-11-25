#!/bin/bash
# Status of every AbacusSummit_base_c000 phase box for the multi-phase run:
# is the box on disk, are its subsamples / IC / catalog / shifted fields / lrgs.py
# results there.
# Run it before submitting anything, and again to follow progress.
#
#   bash slurm/check_phases.sh                  # ph000-ph024, row 1003, Nmesh 512
#   PHASES="0 3 7" ROW=2155 NMESH=256 bash slurm/check_phases.sh
#
# Columns: halos = halo_info, halo_rv_A, field_rv_A present at z_mock (prepare_sim
# reads exactly these); clean = cleaning/<sim>/<z>; subs = prepare_sim output;
# IC = <IC_ROOT>/<sim>/ic/<IC_FILE> (globus_fetch_ic.sh layout); cat = the row's
# catalog; fields = my_generate_shifted_fields.py output at NMESH (real space and
# redshift-space axis 2, full size); lrgs = lrgs.py results for the row (real and
# RSD). Both come from slurm/perlmutter_phases_lrgs_array.sbatch and live in
# OUTROOT/phNNN/{shifted_fields,results}/, except ph000's, which are in
# PH000_DIR/{shifted_fields,results}/ (Heavy_files_Andrej/AbacusBase).
# Paths come from CONFIG (sim_dir, subsample_dir, z_mock); override with
# OUTROOT, IC_ROOT, IC_FILE, PH000_DIR.
set -uo pipefail

CONFIG="${CONFIG:-config/lrg_hod.yaml}"
OUTROOT="${OUTROOT:-${PSCRATCH:-.}/lrg_hods_phases}"
IC_ROOT="${IC_ROOT:-${PSCRATCH:-.}/AbacusSummit}"
IC_FILE="${IC_FILE:-ic_dens_N576.asdf}"
ROW="${ROW:-1003}"
NMESH="${NMESH:-512}"
COSMO="${COSMO:-c000}"
PHASES="${PHASES:-$(seq 0 24)}"
PH000_DIR="${PH000_DIR:-${PSCRATCH:-.}/qso_sbp/Heavy_files_Andrej/AbacusBase}"

SIM_DIR=/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit; SUB_DIR=""; ZDIR=z0.500
if [[ -f "$CONFIG" ]] && command -v python >/dev/null; then
    read -r SIM_DIR SUB_DIR ZDIR < <(python - "$CONFIG" 2>/dev/null <<'EOF'
import sys, yaml
s = yaml.safe_load(open(sys.argv[1]))["sim_params"]
print(s["sim_dir"].rstrip("/"), s.get("subsample_dir", "").rstrip("/"), "z%.3f" % float(s.get("z_mock", 0.5)))
EOF
) || true
fi
[[ "$SUB_DIR" == *"<u>"* ]] && SUB_DIR=""        # config not filled in yet
echo "sim_dir $SIM_DIR | subsamples ${SUB_DIR:-?} | IC $IC_ROOT/<sim>/ic/$IC_FILE | outputs $OUTROOT/phNNN"
echo "row $ROW, Nmesh $NMESH, $ZDIR"
printf '%-6s %-6s %-6s %-6s %-6s %-6s %-6s %-6s\n' phase halos clean subs IC cat fields lrgs
yes_no() { [[ $1 == 1 ]] && echo ok || echo "--"; }
full() { [[ -f $1 ]] && (( $(stat -c %s "$1") >= $2 * NMESH * NMESH * NMESH * 4 )); }   # file, number of grids
n_ok=0; n_box=0; n_fields=0
for p in $PHASES; do
    ph="$(printf '%03d' "$((10#$p))")"; sim="AbacusSummit_base_${COSMO}_ph$ph"
    h=1; for d in halo_info halo_rv_A field_rv_A; do [[ -d "$SIM_DIR/$sim/halos/$ZDIR/$d" ]] || h=0; done
    c=0; [[ -d "$SIM_DIR/cleaning/$sim/$ZDIR" ]] && c=1
    s=0; [[ -n "$SUB_DIR" && -n "$(ls -A "$SUB_DIR/$sim/$ZDIR" 2>/dev/null)" ]] && s=1
    i=0; [[ -s "$IC_ROOT/$sim/ic/$IC_FILE" ]] && i=1
    k=0; [[ -s "$OUTROOT/ph$ph/catalogs/$(printf '%06d' "$ROW").npy" ]] && k=1
    if [[ $ph == 000 ]]; then base="$PH000_DIR"; else base="$OUTROOT/ph$ph"; fi
    f=0; full "$base/shifted_fields/shifted_fields_real_N_${NMESH}_c_0.npy" 4 &&
         full "$base/shifted_fields/shifted_fields_redshift_axisrsd_2_N_${NMESH}_c_0.npy" 6 && f=1
    tag="c=0_ph=${ph}_hod=${ROW}_z=0.5000_Nmesh_${NMESH}"
    r=0; [[ -s "$base/results/results_AbacusBase_${tag}.npy" && -s "$base/results/results_RSD_AbacusBase_${tag}_axisrsd_2.npy" ]] && r=1
    printf '%-6s %-6s %-6s %-6s %-6s %-6s %-6s %-6s\n' "ph$ph" "$(yes_no $h)" "$(yes_no $c)" "$(yes_no $s)" "$(yes_no $i)" "$(yes_no $k)" "$(yes_no $f)" "$(yes_no $r)"
    (( h && c )) && n_box=$((n_box + 1))
    (( f )) && n_fields=$((n_fields + 1))
    (( r )) && n_ok=$((n_ok + 1))
done
echo "boxes fully on disk: $n_box; shifted fields: $n_fields; lrgs.py results: $n_ok"

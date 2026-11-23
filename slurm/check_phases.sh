#!/bin/bash
# Status of every AbacusSummit_base_c000 phase box for the multi-phase run:
# is the box on disk, are its subsamples / IC / catalog / field-level results there.
# Run it before submitting anything, and again to follow progress.
#
#   bash slurm/check_phases.sh                  # ph000-ph024, row 1003, Nmesh 512
#   PHASES="0 3 7" ROW=2155 NMESH=256 bash slurm/check_phases.sh
#
# Columns: halos = halo_info, halo_rv_A, field_rv_A present at z_mock (prepare_sim
# reads exactly these); clean = cleaning/<sim>/<z>; subs = prepare_sim output;
# IC = <IC_ROOT>/<sim>/ic/<IC_FILE> (globus_fetch_ic.sh layout); cat = the row's
# catalog; perr = perr_from_ic.py results (real and redshift space).
# Paths come from CONFIG (sim_dir, subsample_dir, z_mock); override with
# OUTROOT, IC_ROOT, IC_FILE, PERR_DIR.
set -uo pipefail

CONFIG="${CONFIG:-config/lrg_hod.yaml}"
OUTROOT="${OUTROOT:-${PSCRATCH:-.}/lrg_hods_phases}"
IC_ROOT="${IC_ROOT:-${PSCRATCH:-.}/AbacusSummit}"
IC_FILE="${IC_FILE:-ic_dens_N576.asdf}"
ROW="${ROW:-1003}"
NMESH="${NMESH:-512}"
COSMO="${COSMO:-c000}"
PHASES="${PHASES:-$(seq 0 24)}"

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
printf '%-6s %-6s %-6s %-6s %-6s %-6s %-6s\n' phase halos clean subs IC cat perr
yes_no() { [[ $1 == 1 ]] && echo ok || echo "--"; }
n_ok=0; n_box=0
for p in $PHASES; do
    ph="$(printf '%03d' "$((10#$p))")"; sim="AbacusSummit_base_${COSMO}_ph$ph"
    h=1; for d in halo_info halo_rv_A field_rv_A; do [[ -d "$SIM_DIR/$sim/halos/$ZDIR/$d" ]] || h=0; done
    c=0; [[ -d "$SIM_DIR/cleaning/$sim/$ZDIR" ]] && c=1
    s=0; [[ -n "$SUB_DIR" && -n "$(ls -A "$SUB_DIR/$sim/$ZDIR" 2>/dev/null)" ]] && s=1
    i=0; [[ -s "$IC_ROOT/$sim/ic/$IC_FILE" ]] && i=1
    k=0; [[ -s "$OUTROOT/ph$ph/catalogs/$(printf '%06d' "$ROW").npy" ]] && k=1
    pd="${PERR_DIR:-$OUTROOT/ph$ph/perr}"
    r=0; [[ -s "$pd/hod${ROW}_N${NMESH}_real.npy" && -s "$pd/hod${ROW}_N${NMESH}_rsd.npy" ]] && r=1
    printf '%-6s %-6s %-6s %-6s %-6s %-6s %-6s\n' "ph$ph" "$(yes_no $h)" "$(yes_no $c)" "$(yes_no $s)" "$(yes_no $i)" "$(yes_no $k)" "$(yes_no $r)"
    (( h && c )) && n_box=$((n_box + 1))
    (( r )) && n_ok=$((n_ok + 1))
done
echo "boxes fully on disk: $n_box; field-level results done: $n_ok"

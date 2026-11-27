#!/usr/bin/env python3
"""
Strength of the single-bin spikes at k = 34 and 68 k_f (k_f = 2π/L) in lrgs.py
real-space results, row by row, against each row's HOD parameters.

    python spike_scan.py $PSCRATCH/qso_sbp/Heavy_files_Andrej/AbacusBase/results \\
        --ph 0 --nmesh 512 --params lrg_params_eq74.npy --out spike_scan_ph000_N512

For every results_AbacusBase_c=0_ph=PH_hod=ROW_..._Nmesh_N.npy it takes n̄P_gal
and n̄P_err at k = 34 and 68 k_f minus the median of the ±4 neighbouring bins.
Writes
  <out>.csv   row, the four excesses, n̄, and the row's HOD parameters
  <out>.png/.pdf  the 34 k_f galaxy excess against Bcent, Bsat, Acent, Asat,
              logsigma and logM_cut
and prints Spearman correlations. If the pattern comes from the environment
used by AbacusHOD's assembly bias, the excess grows with |Bcent| or |Bsat| and
is ~0 at B = 0. Takes a minute or two for 10500 rows.
"""

import argparse
import os
import re
import sys

import numpy as np

COLS = "logM_cut logM1 logsigma alpha alpha_c alpha_s kappa s Acent Asat Bcent Bsat sigma nbar".split()
NAME = re.compile(r"results_AbacusBase_c=?(\d+)_ph=?(\d+)_hod=?(\d+)_z=?([0-9.]+?)_Nmesh_(\d+)\.npy$")


def excess(y, i, w=4):
    lo, hi = max(0, i - w), min(len(y), i + w + 1)
    nb = np.r_[y[lo:i], y[i + 1:hi]]
    nb = nb[np.isfinite(nb)]
    return float(y[i] - np.median(nb)) if len(nb) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="results directories (searched recursively) or files")
    ap.add_argument("--ph", type=int, default=0)
    ap.add_argument("--nmesh", type=int, default=512)
    ap.add_argument("--box", type=float, default=2000.0)
    ap.add_argument("--params", default="lrg_params_eq74.npy")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or "spike_scan_ph%03d_N%d" % (a.ph, a.nmesh)
    kf = 2 * np.pi / a.box
    table = np.load(a.params)

    files = {}
    for p in a.paths:
        walk = os.walk(p, followlinks=True) if os.path.isdir(p) else [(os.path.dirname(p), [], [os.path.basename(p)])]
        for d, _, names in walk:
            for nm in names:
                m = NAME.search(nm)
                if m and int(m.group(2)) == a.ph and int(m.group(5)) == a.nmesh:
                    files.setdefault(int(m.group(3)), os.path.join(d, nm))
    if not files:
        sys.exit("no real-space lrgs.py results for ph%03d at Nmesh %d" % (a.ph, a.nmesh))
    rows = sorted(files)
    print("%d rows" % len(rows))

    rec = []
    for j, row in enumerate(rows):
        r = np.load(files[row], allow_pickle=True)
        k = np.asarray(r[0], float)
        nb = float(r[-1])
        perr, pgal = nb * np.asarray(r[1], float), nb * np.asarray(r[2], float)
        i34, i68 = int(np.argmin(np.abs(k - 34 * kf))), int(np.argmin(np.abs(k - 68 * kf)))
        rec.append([row, excess(pgal, i34), excess(perr, i34), excess(pgal, i68), excess(perr, i68), nb]
                   + list(table[row][:13]))
        if (j + 1) % 2000 == 0:
            print("  %d / %d" % (j + 1, len(rows)))
    R = np.array(rec)
    head = ["row", "gal34", "err34", "gal68", "err68", "nbar_measured"] + COLS[:13]
    np.savetxt(out + ".csv", R, delimiter=",", header=",".join(head), comments="",
               fmt=["%d"] + ["%.6g"] * (R.shape[1] - 1))
    print("wrote %s.csv" % out)

    from scipy.stats import spearmanr
    g = R[:, 1]
    ok = np.isfinite(g)
    print("\nexcess n̄P_gal at 34 k_f: median %.4f, 90th percentile %.4f, max %.4f (row %d)"
          % (np.median(g[ok]), np.percentile(g[ok], 90), np.nanmax(g), int(R[np.nanargmax(g), 0])))
    print("Spearman correlation of that excess with:")
    for c in ["Bcent", "Bsat", "Acent", "Asat", "logsigma", "logM_cut", "logM1", "alpha", "kappa"]:
        v = R[:, 6 + COLS.index(c)]
        rho = spearmanr(g[ok], v[ok]).correlation
        rho_abs = spearmanr(g[ok], np.abs(v[ok])).correlation
        print("  %-9s rho %+.3f   with |%s| %+.3f" % (c, rho, c, rho_abs))
    for c in ["Bcent", "Bsat"]:
        v = np.abs(R[:, 6 + COLS.index(c)])
        print("  median excess for |%s| < 0.05: %.4f (%d rows);  > 0.8: %.4f (%d rows)" % (
            c, np.median(g[ok & (v < 0.05)]), np.sum(ok & (v < 0.05)), np.median(g[ok & (v > 0.8)]), np.sum(ok & (v > 0.8))))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 6, figsize=(24, 3.8), sharey=True)
    for ax, c in zip(axes, ["Bcent", "Bsat", "Acent", "Asat", "logsigma", "logM_cut"]):
        v = R[:, 6 + COLS.index(c)]
        ax.plot(v, g, ".", ms=2, alpha=0.4, color="k")
        ax.set_xlabel(c)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"excess $\bar n P_{\rm gal}$ at $34\,k_f$")
    fig.suptitle("ph%03d, Nmesh %d: single-bin excess at k = 34 k_f, one point per HOD row" % (a.ph, a.nmesh), fontsize=11)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=130, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print("wrote %s.png and .pdf" % out)


if __name__ == "__main__":
    main()

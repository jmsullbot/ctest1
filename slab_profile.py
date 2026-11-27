#!/usr/bin/env python3
"""
Galaxy density folded with a period of L/NSLAB along x, y and z: the test for a
pattern tied to the AbacusSummit superslab files (34 per base box).

    python slab_profile.py $PSCRATCH/lrg_hods_phases/ph000/catalogs/001003.npy \\
        $PSCRATCH/lrg_hods/catalogs/000123.npy  [--box 2000 --nslab 34 --bins 17]

For each catalog (structured x, y, z, id, centred on 0 or in [0, L)) and each
axis it prints the first-harmonic amplitude

    A1 = 2 |< exp(2πi NSLAB x / L) >|          (noise level ≈ 2/√N)

and the folded density, relative to the mean, in --bins bins across one period,
for all galaxies, centrals and satellites. The first Ncent galaxies of a
catalog are the centrals: Ncent is the single drop in the halo-id sequence, as
in compute_fsat.py. A pure sinusoid of relative amplitude A has A1 = A.

The spikes at k = 34 k_f in the row-1003 lrgs.py results (excess n̄P ≈ 1.0 in
real space) correspond to A1 ≈ 0.18 if they come from one axis, or ≈ 0.13 each
from two.
"""

import argparse
import sys

import numpy as np


def n_centrals(ids):
    """Ncent from the single drop in the halo-id sequence (compute_fsat.py), or -1."""
    if len(ids) == 0:
        return 0
    drops = np.flatnonzero(np.diff(np.asarray(ids).astype(np.int64)) < 0)
    return int(drops[0]) + 1 if len(drops) == 1 else -1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("catalogs", nargs="+")
    ap.add_argument("--box", type=float, default=2000.0)
    ap.add_argument("--nslab", type=int, default=34)
    ap.add_argument("--bins", type=int, default=17)
    a = ap.parse_args()
    L, ns = a.box, a.nslab
    period = L / ns
    for fn in a.catalogs:
        c = np.load(fn, mmap_mode="r")
        names = c.dtype.names or ()
        if not all(q in names for q in "xyz"):
            sys.exit("%s: need a structured catalog with x, y, z (fields: %s)" % (fn, ", ".join(names)))
        n = len(c)
        lo = min(float(np.min(c[q])) for q in "xyz")
        shift = 0.5 * L if lo < -0.01 * L else 0.0
        nc = n_centrals(c["id"]) if "id" in names else -1
        groups = [("all", slice(None))]
        if nc > 0:
            groups += [("centrals", slice(0, nc)), ("satellites", slice(nc, None))]
        print("%s: %d galaxies%s, period L/%d = %.2f Mpc/h, %d bins"
              % (fn, n, (", %d centrals (%.1f %%)" % (nc, 100.0 * nc / n)) if nc > 0 else
                 " (no single halo-id drop: centrals not split)", ns, period, a.bins))
        for name, sl in groups:
            print("  %-10s" % name)
            for q in "xyz":
                x = (np.asarray(c[q][sl], np.float64) + shift) % L
                m = len(x)
                if m == 0:
                    continue
                A1 = 2 * np.abs(np.mean(np.exp(2j * np.pi * ns * x / L)))
                h = np.histogram(np.mod(x, period), bins=a.bins, range=(0, period))[0]
                rel = h / (m / float(a.bins)) - 1
                print("    %s  A1 = %.4f  (noise %.4f, %5.1fσ)  folded δ: %s"
                      % (q, A1, 2 / np.sqrt(m), A1 / (2 / np.sqrt(m)), " ".join("%+.3f" % v for v in rel)))
        print()


if __name__ == "__main__":
    main()

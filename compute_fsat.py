#!/usr/bin/env python3
"""
Recover the satellite fraction of every existing catalog.

AbacusHOD returns the central/satellite split only as a scalar ``Ncent``
(the FIRST ``Ncent`` galaxies of each catalog are centrals, the rest are
satellites -- see abacusnbody/hod/abacus_hod.py, run_hod docstring), and
the first generation runs did not record it.  It can be recovered from the
``id`` column: both blocks are written in halo-processing order, so halo id
is non-decreasing through the centrals, then DROPS at the first satellite,
then is non-decreasing again.  The single drop marks ``Ncent``.

Every row is validated by counting drops: exactly one -> clean recovery;
zero -> ambiguous (an all-central catalog, or empty); more than one ->
ordering assumption violated, the row is flagged and its f_sat set to NaN.
The summary reports how many rows fell into each class, so you can see at a
glance whether the method held.

LRG satellites in AbacusHOD are NOT conditioned on a central being present
(only ELG satellites are), so counting unique halo ids would undercount
centrals -- hence this approach rather than that one.

Usage
-----
    python compute_fsat.py $PSCRATCH/lrg_hods                   # -> $PSCRATCH/lrg_hods/fsat_table.npz
    python compute_fsat.py $PSCRATCH/lrg_hods_small --nproc 32

Output ``fsat_table.npz`` fields (all indexed by global table row):
    row, n_gal, n_cent, f_sat, n_drops        (n_cent = -1 where ambiguous)

Only the ``id`` column is read, via memory map, so this is fast even for
multi-GB catalogs.
"""

import argparse
import glob
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def split_from_ids(ids: np.ndarray) -> tuple[int, int]:
    """Return (n_cent, n_drops) for one catalog's halo-id column."""
    n = len(ids)
    if n == 0:
        return 0, 0
    drops = np.flatnonzero(np.diff(ids.astype(np.int64)) < 0)
    if len(drops) == 1:
        return int(drops[0]) + 1, 1
    return -1, int(len(drops))          # ambiguous or violated


def _one(path: str) -> tuple[int, int, int, int]:
    row = int(Path(path).stem)
    cat = np.load(path, mmap_mode="r")
    n = len(cat)
    if n == 0:
        return row, 0, 0, 0
    n_cent, n_drops = split_from_ids(cat["id"])
    return row, n, n_cent, n_drops


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir", help="run directory containing catalogs/")
    p.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--output", default=None, help="default: <outdir>/fsat_table.npz")
    args = p.parse_args()

    files = sorted(glob.glob(str(Path(args.outdir) / "catalogs" / "*.npy")))
    if not files:
        raise SystemExit(f"no catalogs under {args.outdir}/catalogs/")
    out = Path(args.output or Path(args.outdir) / "fsat_table.npz")

    print(f"{len(files)} catalogs, {args.nproc} processes …")
    with Pool(args.nproc) as pool:
        res = pool.map(_one, files, chunksize=16)

    res = np.array(res, dtype=np.int64)
    row, n_gal, n_cent, n_drops = res.T
    f_sat = np.where(n_cent >= 0, 1.0 - n_cent / np.maximum(n_gal, 1), np.nan)
    f_sat[n_gal == 0] = np.nan

    np.savez(out, row=row, n_gal=n_gal, n_cent=n_cent, f_sat=f_sat, n_drops=n_drops)

    clean = (n_drops == 1).sum()
    empty = (n_gal == 0).sum()
    zero  = ((n_drops == 0) & (n_gal > 0)).sum()
    multi = (n_drops > 1).sum()
    print(f"wrote {out}")
    print(f"  clean recovery (1 drop) : {clean}")
    print(f"  empty catalogs          : {empty}")
    print(f"  0 drops, non-empty      : {zero}   (all-central? f_sat set NaN — inspect)")
    print(f"  >1 drops                : {multi}  (ordering assumption violated — f_sat NaN)")
    ok = np.isfinite(f_sat)
    if ok.any():
        q = np.nanpercentile(f_sat, [5, 50, 95])
        print(f"  f_sat  5/50/95 %        : {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}")
    if multi or zero:
        bad = row[(n_drops != 1) & (n_gal > 0)][:20]
        print(f"  flagged rows (first 20) : {bad.tolist()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
One table per HOD run: parameters, galaxy count, number density, f_sat.

    python hod_summary.py $PSCRATCH/lrg_hods_small
    python hod_summary.py $PSCRATCH/lrg_hods --nproc 32

Writes <dir>/hod_summary.npz and <dir>/hod_summary.csv, one row per
catalog, ordered by global row (row N <-> catalogs/{N:06d}.npy).

Columns: row, the 12 AbacusHOD params, logsigma, n_gal, nbar, n_cent, f_sat

Where each value comes from:
  * params, logsigma, n_gal  -- the metadata shards (*_rows*.hdf5)
  * box size                 -- 'box_size' attr; else inferred from 'sim_name'
                                (small -> 500, base -> 2000); else --box
  * nbar = n_gal / box**3    -- the stored nbar column is NaN by design
  * n_cent, f_sat            -- the shards if they have them; else
                                <dir>/fsat_table.npz; else recovered from the
                                catalogs' id column (compute_fsat.py method)
                                and cached to fsat_table.npz for next time
"""

import argparse
import glob
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np


def box_from_attrs(attrs, override):
    if override:
        return float(override), "--box"
    if "box_size" in attrs:
        return float(attrs["box_size"]), "box_size attr"
    name = attrs.get("sim_name", "")
    name = name.decode() if isinstance(name, bytes) else str(name)
    if "_small_" in name:
        return 500.0, f"sim_name {name}"
    if "_base_" in name:
        return 2000.0, f"sim_name {name}"
    raise SystemExit("cannot determine box size (no box_size attr, unrecognised "
                     f"sim_name {name!r}) -- pass --box")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", help="run directory, e.g. $PSCRATCH/lrg_hods_small")
    ap.add_argument("--box", type=float, default=None, help="override box side [Mpc/h]")
    ap.add_argument("--nproc", type=int, default=16, help="processes for f_sat recovery")
    ap.add_argument("--no-fsat", action="store_true", help="skip f_sat")
    a = ap.parse_args()
    D = Path(a.outdir)

    shards = sorted(glob.glob(str(D / "*_rows*.hdf5"))) or sorted(glob.glob(str(D / "*.hdf5")))
    if not shards:
        raise SystemExit(f"no metadata .hdf5 files in {D}")

    rows, pars, logs, ngal, ncent, fsat = [], [], [], [], [], []
    boxes = set()
    for fn in shards:
        with h5py.File(fn, "r") as f:
            cols = [c.decode() if isinstance(c, bytes) else str(c) for c in f["params"].attrs["columns"]]
            L, src = box_from_attrs(f.attrs, a.box)
            boxes.add((L, src))
            n = len(f["row_index"])
            rows.append(f["row_index"][:]); pars.append(f["params"][:])
            logs.append(f["logsigma"][:]);  ngal.append(f["n_gal"][:])
            ncent.append(f["n_cent"][:] if "n_cent" in f else np.full(n, -1))
            fsat.append(f["f_sat"][:] if "f_sat" in f else np.full(n, np.nan))
    if len({b for b, _ in boxes}) != 1:
        raise SystemExit(f"shards disagree on box size: {boxes}")
    L, src = boxes.pop()

    rows, pars, logs, ngal, ncent, fsat = map(np.concatenate, (rows, pars, logs, ngal, ncent, fsat))
    o = np.argsort(rows)
    rows, pars, logs, ngal, ncent, fsat = rows[o], pars[o], logs[o], ngal[o], ncent[o], fsat[o]
    nbar = ngal / L**3

    missing = np.setdiff1d(np.arange(rows.max() + 1), rows)
    print(f"{len(shards)} shards, {len(rows)} runs, box {L:g} Mpc/h (from {src})")
    if len(missing):
        print(f"WARNING: {len(missing)} rows missing from the shards, e.g. {missing[:10].tolist()}")

    # ---- f_sat -----------------------------------------------------------
    fsat_src = "shards"
    if not a.no_fsat and np.isnan(fsat).all():
        tbl = D / "fsat_table.npz"
        if not tbl.exists():
            from compute_fsat import _one        # same method, validated per row
            files = [str(D / "catalogs" / f"{r:06d}.npy") for r in rows]
            print(f"recovering f_sat from {len(files)} catalogs ({a.nproc} procs) ...")
            with Pool(a.nproc) as p:
                res = np.array(p.map(_one, files, chunksize=16), dtype=np.int64)
            r_, ng_, nc_, nd_ = res.T
            fs_ = np.where((nc_ >= 0) & (ng_ > 0), 1 - nc_ / np.maximum(ng_, 1), np.nan)
            np.savez(tbl, row=r_, n_gal=ng_, n_cent=nc_, f_sat=fs_, n_drops=nd_)
            print(f"cached to {tbl}")
        t = np.load(tbl)
        m = dict(zip(t["row"], zip(t["n_cent"], t["f_sat"])))
        ncent = np.array([m.get(r, (-1, np.nan))[0] for r in rows])
        fsat  = np.array([m.get(r, (-1, np.nan))[1] for r in rows])
        fsat_src = tbl.name

    # ---- save ------------------------------------------------------------
    names = ["row"] + cols + ["logsigma", "n_gal", "nbar", "n_cent", "f_sat"]
    table = np.column_stack([rows, pars, logs, ngal, nbar, ncent, fsat])
    np.savez(D / "hod_summary.npz", table=table, columns=np.array(names), box_size=L)
    np.savetxt(D / "hod_summary.csv", table, delimiter=",", header=",".join(names), comments="",
               fmt=["%d"] + ["%.6g"] * len(cols) + ["%.6g", "%d", "%.6e", "%d", "%.5f"])

    ok = np.isfinite(fsat)
    print(f"nbar  : median {np.median(nbar):.3e}, range {nbar.min():.3e} .. {nbar.max():.3e}")
    if ok.any():
        print(f"f_sat : median {np.median(fsat[ok]):.3f}, range {fsat[ok].min():.3f} .. "
              f"{fsat[ok].max():.3f}  ({ok.sum()}/{len(fsat)} rows, from {fsat_src})")
    print(f"empty catalogs: {(ngal == 0).sum()}")
    print(f"wrote {D/'hod_summary.npz'} and {D/'hod_summary.csv'}")


if __name__ == "__main__":
    main()

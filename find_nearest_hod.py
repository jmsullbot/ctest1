#!/usr/bin/env python3
"""
Find the HOD runs in the parameter table closest to a target parameter set.

Distances are computed in each parameter's prior range (so a 0.1 shift in
logM_cut and a 0.1 shift in kappa count differently), with user-set weights.
By default the occupation parameters that control number density
(logM_cut, logM1, logsigma) dominate.

If HDF5 metadata shards are given (``--hdf5-dir``), the *measured* number
density n_gal / L_box^3 is included in the ranking -- that is the quantity
you usually care about most, and it is not in the parameter table (nbar
there is NaN by design).

Usage
-----
    # rank on HOD parameters only (works anywhere)
    python find_nearest_hod.py --logM_cut 13.571 --logM1 14.915 --logsigma -1.177

    # on Perlmutter, include measured nbar from the base-box shards
    python find_nearest_hod.py --logM_cut 13.571 --logM1 14.915 --logsigma -1.177 \
        --nbar 7.42e-5 --hdf5-dir $PSCRATCH/lrg_hods

Any of the 12 HOD parameters may be given; unspecified ones are ignored.
"""

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np

from make_lrg_param_table import COLUMNS, EQ74_BOUNDS

_RANGE = {name: hi - lo for name, lo, hi in EQ74_BOUNDS}

# Default weights: the three parameters that set nbar dominate.
_DEFAULT_W = {"logM_cut": 3.0, "logM1": 3.0, "logsigma": 2.0}


def load_measured(hdf5_dir: str, n_rows: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (nbar, f_sat) per global row plus the box size.

    nbar comes from n_gal in the metadata shards.  f_sat comes from the
    shards' ``f_sat`` dataset when present (runs made after it was added),
    otherwise from ``<hdf5_dir>/fsat_table.npz`` written by compute_fsat.py,
    otherwise NaN.
    """
    rows, ngal, fsat, L = [], [], [], None
    # sharded runs write <name>_rowsSTART-END.hdf5; a serial run writes <name>.hdf5
    for fn in sorted(glob.glob(str(Path(hdf5_dir) / "*.hdf5"))):
        with h5py.File(fn, "r") as f:
            rows.append(f["row_index"][:])
            ngal.append(f["n_gal"][:])
            fsat.append(f["f_sat"][:] if "f_sat" in f else np.full(len(rows[-1]), np.nan))
            L = float(f.attrs.get("box_size", 2000.0))
    if not rows:
        raise SystemExit(f"no metadata .hdf5 files found in {hdf5_dir}")
    rows, ngal, fsat = map(np.concatenate, (rows, ngal, fsat))
    n = max(n_rows, rows.max() + 1)
    nbar_out = np.full(n, np.nan); nbar_out[rows] = ngal / L**3
    fsat_out = np.full(n, np.nan); fsat_out[rows] = fsat

    tbl = Path(hdf5_dir) / "fsat_table.npz"
    if np.isnan(fsat_out).all() and tbl.exists():
        t = np.load(tbl)
        fsat_out[t["row"]] = t["f_sat"]
        print(f"f_sat loaded from {tbl}")
    return nbar_out, fsat_out, L


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, _, _ in EQ74_BOUNDS:
        p.add_argument(f"--{name}", type=float, default=None)
    p.add_argument("--nbar", type=float, default=None,
                   help="target number density [(Mpc/h)^-3]; needs --hdf5-dir")
    p.add_argument("--fsat", type=float, default=None,
                   help="target satellite fraction; needs --hdf5-dir (f_sat in the "
                        "shards, or fsat_table.npz from compute_fsat.py)")
    p.add_argument("--hdf5-dir", default=None,
                   help="directory with *_rows*.hdf5 shards (adds measured nbar, f_sat)")
    p.add_argument("--params", default="lrg_params_eq74.npy")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--nbar-weight", type=float, default=4.0,
                   help="weight on |log10 nbar| mismatch, in dex units")
    p.add_argument("--fsat-weight", type=float, default=4.0,
                   help="weight on |f_sat| mismatch, per 0.1 of f_sat")
    p.add_argument("--weight", action="append", default=[], metavar="NAME=W",
                   help="override a parameter weight, e.g. --weight kappa=1")
    args = p.parse_args()

    table = np.load(args.params)
    idx = {c: i for i, c in enumerate(COLUMNS)}

    weights = dict(_DEFAULT_W)
    for spec in args.weight:
        k, v = spec.split("=")
        weights[k] = float(v)

    target = {n: getattr(args, n) for n, _, _ in EQ74_BOUNDS if getattr(args, n) is not None}
    if not target and args.nbar is None and args.fsat is None:
        raise SystemExit("give at least one --<param>, --nbar or --fsat")

    d2 = np.zeros(len(table))
    terms = []
    for name, val in target.items():
        w = weights.get(name, 1.0)
        z = (table[:, idx[name]] - val) / _RANGE[name]
        d2 += w * z**2
        terms.append(f"{name}(w={w:g})")

    nbar = fsat = None
    if args.hdf5_dir:
        nbar, fsat, L = load_measured(args.hdf5_dir, len(table))
        nbar, fsat = nbar[: len(table)], fsat[: len(table)]
        print(f"measured nbar loaded from {args.hdf5_dir}  (box {L:g} Mpc/h)")
        if args.nbar is not None:
            dlog = np.log10(nbar) - np.log10(args.nbar)
            dlog = np.where(np.isfinite(dlog), dlog, 10.0)   # missing/empty -> far away
            d2 += args.nbar_weight * dlog**2
            terms.append(f"log10nbar(w={args.nbar_weight:g}, dex)")
        if args.fsat is not None:
            if np.isnan(fsat).all():
                raise SystemExit("no f_sat available: shards lack it and no fsat_table.npz -- "
                                 f"run  python compute_fsat.py {args.hdf5_dir}")
            df = (fsat - args.fsat) / 0.1
            df = np.where(np.isfinite(df), df, 100.0)
            d2 += args.fsat_weight * df**2
            terms.append(f"f_sat(w={args.fsat_weight:g} per 0.1)")
        if np.isnan(fsat).all():
            fsat = None
    elif args.nbar is not None or args.fsat is not None:
        print("NOTE: --nbar/--fsat given without --hdf5-dir; they are measured quantities "
              "and cannot be used. Ranking on HOD parameters only.\n")

    order = np.argsort(d2)[: args.top]

    show = ["logM_cut", "logM1", "logsigma", "alpha", "alpha_c", "alpha_s",
            "kappa", "s", "Acent", "Asat", "Bcent", "Bsat"]
    print("ranking on:", ", ".join(terms))
    print()
    def fmt_or(v, w, f):
        return f" {v:{f}}" if v is not None else f" {'-':>{w}}"

    hdr = f"{'rank':>4} {'row':>6} {'dist':>7}"
    hdr += f" {'nbar':>9}" if nbar is not None else ""
    hdr += f" {'f_sat':>6}" if fsat is not None else ""
    hdr += "".join(f" {c:>9}" for c in show)
    print(hdr)
    tl = f"{'':>4} {'TARGET':>6} {'':>7}"
    tl += fmt_or(args.nbar, 9, "9.3e") if nbar is not None else ""
    tl += fmt_or(args.fsat, 6, "6.3f") if fsat is not None else ""
    tl += "".join(f" {target[c]:9.4f}" if c in target else f" {'-':>9}" for c in show)
    print(tl)
    for r, i in enumerate(order, 1):
        line = f"{r:4d} {i:6d} {np.sqrt(d2[i]):7.4f}"
        if nbar is not None:
            line += f" {nbar[i]:9.3e}"
        if fsat is not None:
            line += f" {fsat[i]:6.3f}"
        line += "".join(f" {table[i, idx[c]]:9.4f}" for c in show)
        print(line)
    print()
    print("catalog files:", ", ".join(f"catalogs/{i:06d}.npy" for i in order[:5]), "...")


if __name__ == "__main__":
    main()

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


def load_measured_nbar(hdf5_dir: str) -> tuple[np.ndarray, float]:
    """Return (nbar per global row, box size) from the metadata shards."""
    rows, ngal, L = [], [], None
    for fn in sorted(glob.glob(str(Path(hdf5_dir) / "*_rows*.hdf5"))):
        with h5py.File(fn, "r") as f:
            rows.append(f["row_index"][:])
            ngal.append(f["n_gal"][:])
            L = float(f.attrs.get("box_size", 2000.0))
    if not rows:
        raise SystemExit(f"no *_rows*.hdf5 shards found in {hdf5_dir}")
    rows, ngal = np.concatenate(rows), np.concatenate(ngal)
    nbar = np.full(rows.max() + 1, np.nan)
    nbar[rows] = ngal / L**3
    return nbar, L


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, _, _ in EQ74_BOUNDS:
        p.add_argument(f"--{name}", type=float, default=None)
    p.add_argument("--nbar", type=float, default=None,
                   help="target number density [(Mpc/h)^-3]; needs --hdf5-dir")
    p.add_argument("--hdf5-dir", default=None,
                   help="directory with *_rows*.hdf5 shards (adds measured nbar)")
    p.add_argument("--params", default="lrg_params_eq74.npy")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--nbar-weight", type=float, default=4.0,
                   help="weight on |log10 nbar| mismatch, in dex units")
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
    if not target and args.nbar is None:
        raise SystemExit("give at least one --<param> or --nbar")

    d2 = np.zeros(len(table))
    terms = []
    for name, val in target.items():
        w = weights.get(name, 1.0)
        z = (table[:, idx[name]] - val) / _RANGE[name]
        d2 += w * z**2
        terms.append(f"{name}(w={w:g})")

    nbar = None
    if args.hdf5_dir:
        nbar, L = load_measured_nbar(args.hdf5_dir)
        if len(nbar) < len(table):
            nbar = np.concatenate([nbar, np.full(len(table) - len(nbar), np.nan)])
        if args.nbar is not None:
            dlog = np.log10(nbar) - np.log10(args.nbar)
            dlog = np.where(np.isfinite(dlog), dlog, 10.0)   # missing/empty -> far away
            d2 += args.nbar_weight * dlog**2
            terms.append(f"log10nbar(w={args.nbar_weight:g}, dex)")
        print(f"measured nbar loaded from {args.hdf5_dir}  (box {L:g} Mpc/h)")
    elif args.nbar is not None:
        print("NOTE: --nbar given without --hdf5-dir; nbar is NaN in the table and "
              "cannot be used. Ranking on HOD parameters only.\n")

    order = np.argsort(d2)[: args.top]

    show = ["logM_cut", "logM1", "logsigma", "alpha", "alpha_c", "alpha_s",
            "kappa", "s", "Acent", "Asat", "Bcent", "Bsat"]
    print("ranking on:", ", ".join(terms))
    print()
    hdr = f"{'rank':>4} {'row':>6} {'dist':>7}" + (f" {'nbar':>9}" if nbar is not None else "")
    hdr += "".join(f" {c:>9}" for c in show)
    print(hdr)
    tl = f"{'':>4} {'TARGET':>6} {'':>7}" + (f" {args.nbar:9.3e}" if (nbar is not None and args.nbar is not None) else (f" {'':>9}" if nbar is not None else ""))
    tl += "".join(f" {target[c]:9.4f}" if c in target else f" {'-':>9}" for c in show)
    print(tl)
    for r, i in enumerate(order, 1):
        line = f"{r:4d} {i:6d} {np.sqrt(d2[i]):7.4f}"
        if nbar is not None:
            line += f" {nbar[i]:9.3e}"
        line += "".join(f" {table[i, idx[c]]:9.4f}" for c in show)
        print(line)
    print()
    print("catalog files:", ", ".join(f"catalogs/{i:06d}.npy" for i in order[:5]), "...")


if __name__ == "__main__":
    main()

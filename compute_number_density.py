#!/usr/bin/env python3
"""
Compute the comoving number density n_bar for each HOD run in a QSO catalog.

    n_bar  [(Mpc/h)^{-3}]  =  N_gal / L_box^3

Can be used as a standalone script or imported as a module.

Usage
-----
    python compute_number_density.py output/qso_hods.hdf5 \\
        --box_size 2000 [--output output/nbar.npz]
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def compute_number_densities(hdf5_path: str, box_size_mpch: float) -> dict:
    """
    Compute the comoving number density for each HOD run in a catalog file.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 file produced by generate_qso_hods.py.
    box_size_mpch : float
        Comoving side length of the simulation box [Mpc/h].

    Returns
    -------
    dict with keys:
        n_bar        : ndarray [n_runs]        number density [(Mpc/h)^{-3}]
        n_gal        : ndarray [n_runs]        total galaxy count per run
        params       : ndarray [n_runs, n_p]   HOD parameter vectors
        param_names  : list[str]               column names for ``params``
        V_box        : float                   box volume [(Mpc/h)^3]
    """
    V_box = box_size_mpch ** 3
    with h5py.File(hdf5_path, "r") as hf:
        n_gal       = hf["n_gal"][:]
        params      = hf["params"][:]
        param_names = list(hf["params"].attrs["columns"])
    return {
        "n_bar":       n_gal / V_box,
        "n_gal":       n_gal,
        "params":      params,
        "param_names": param_names,
        "V_box":       V_box,
    }


def print_summary(result: dict) -> None:
    n_bar = result["n_bar"]
    print(f"Runs          : {len(n_bar)}")
    print(f"Box volume    : {result['V_box']:.3e}  (Mpc/h)^3")
    print(f"n_bar  min    : {n_bar.min():.4e}  (Mpc/h)^-3")
    print(f"n_bar  max    : {n_bar.max():.4e}  (Mpc/h)^-3")
    print(f"n_bar  median : {np.median(n_bar):.4e}  (Mpc/h)^-3")
    print(f"n_bar  mean   : {n_bar.mean():.4e}  (Mpc/h)^-3")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input HDF5 catalog file")
    parser.add_argument(
        "--box_size", type=float, required=True,
        help="Comoving box side length [Mpc/h]",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional: save n_bar and n_gal arrays to this .npz file",
    )
    return parser.parse_args()


def main() -> None:
    args   = _parse_args()
    result = compute_number_densities(args.input, args.box_size)
    print_summary(result)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            n_bar=result["n_bar"],
            n_gal=result["n_gal"],
            params=result["params"],
        )
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

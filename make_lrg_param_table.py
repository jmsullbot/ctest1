#!/usr/bin/env python3
"""
Generate the LRG HOD parameter table from the flat priors of eq. (74) of
Ivanov, Obuljen, Cuesta-Lazaro & Toomey 2024 (arXiv:2409.10609):

    log10 M_cut ∈ [12, 14]      log10 M1 ∈ [13, 15]
    log10 σ     ∈ [−3.5, 1.0]   α        ∈ [0.5, 1.5]
    α_c ∈ [0, 1]   α_s ∈ [0, 2]   s ∈ [0, 1]   κ ∈ [0, 1.5]
    A_cen, A_sat, B_cen, B_sat ∈ [−1, 1]

Output is a .npy array of shape (n, 14) in the column order expected by
generate_lrg_hods.py:

    logM_cut  logM1  logsigma  alpha  alpha_c  alpha_s  kappa
    s  Acent  Asat  Bcent  Bsat  sigma  nbar

where sigma = 10**logsigma and nbar = NaN (the number density is a
*measured output* of each HOD run, not an input — the paper's table had it
filled in after the fact).

Sampling is Latin Hypercube by default (best space coverage for emulator
training; deterministic for a given seed + n).  Pass --method uniform for
independent uniform draws, which is what the paper itself did.

Usage
-----
    python make_lrg_param_table.py                        # 10500 LHS, seed 42
    python make_lrg_param_table.py --n 500 --seed 1 --method uniform \\
        --output my_table.npy
"""

import argparse

import numpy as np

# (name, lo, hi) for the 12 varied parameters — eq. (74) of arXiv:2409.10609,
# in the column order of the output table.
EQ74_BOUNDS = [
    ("logM_cut", 12.0, 14.0),
    ("logM1",    13.0, 15.0),
    ("logsigma", -3.5,  1.0),
    ("alpha",     0.5,  1.5),
    ("alpha_c",   0.0,  1.0),
    ("alpha_s",   0.0,  2.0),
    ("kappa",     0.0,  1.5),
    ("s",         0.0,  1.0),
    ("Acent",    -1.0,  1.0),
    ("Asat",     -1.0,  1.0),
    ("Bcent",    -1.0,  1.0),
    ("Bsat",     -1.0,  1.0),
]

COLUMNS = [name for name, _, _ in EQ74_BOUNDS] + ["sigma", "nbar"]


def make_table(n: int, seed: int, method: str = "lhs") -> np.ndarray:
    """Sample n parameter vectors from the eq. (74) flat priors.

    Returns an (n, 14) float64 array; columns as in COLUMNS.
    """
    lo = np.array([b[1] for b in EQ74_BOUNDS])
    hi = np.array([b[2] for b in EQ74_BOUNDS])

    if method == "lhs":
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=len(EQ74_BOUNDS), seed=seed)
        varied = qmc.scale(sampler.random(n=n), lo, hi)
    elif method == "uniform":
        rng = np.random.default_rng(seed)
        varied = rng.uniform(lo, hi, size=(n, len(EQ74_BOUNDS)))
    else:
        raise ValueError(f"Unknown method {method!r}. Choose 'lhs' or 'uniform'.")

    logsigma = varied[:, COLUMNS.index("logsigma")]
    sigma    = 10.0 ** logsigma
    nbar     = np.full(n, np.nan)   # measured per run, unknown a priori

    return np.column_stack([varied, sigma, nbar])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n",      type=int, default=10500,
                        help="number of samples (default: 10500, as in the paper)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--method", choices=["lhs", "uniform"], default="lhs",
                        help="'lhs' (default) or 'uniform' (paper-style draws)")
    parser.add_argument("--output", default="lrg_params_eq74.npy",
                        help="output .npy path (default: lrg_params_eq74.npy)")
    args = parser.parse_args()

    table = make_table(args.n, args.seed, args.method)
    np.save(args.output, table)

    print(f"Wrote {args.output}: shape {table.shape}, method={args.method}, "
          f"seed={args.seed}")
    print(f"{'col':10s} {'min':>12s} {'max':>12s}")
    for j, name in enumerate(COLUMNS):
        if name == "nbar":
            print(f"{name:10s} {'NaN':>12s} {'NaN':>12s}   (measured per run, not an input)")
            continue
        print(f"{name:10s} {table[:, j].min():12.6f} {table[:, j].max():12.6f}")


if __name__ == "__main__":
    main()

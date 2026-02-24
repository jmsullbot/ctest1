#!/usr/bin/env python3
"""
Generate QSO HOD mock catalogs over a parameter space drawn from the
priors in Table II of Yuan et al. 2023 (arXiv:2306.06314), QSO column.

Seven parameters are varied:

    Parameter   Prior              Bounds       Description
    ---------   -----              ------       -----------
    logM_cut    N(12.7, 1.0)   [11.2, 14.0]   log10 min halo mass [log10 Msun/h]
    logM1       N(15.0, 1.0)   [12.0, 16.0]   log10 satellite characteristic mass
    sigma       N(0.5,  0.5)   [0.0,   3.0]   width of central occupation step
    alpha       N(1.0,  0.5)   [0.3,   2.0]   satellite power-law slope
    kappa       N(0.5,  0.5)   [0.3,   3.0]   satellite truncation parameter
    alpha_c     N(1.5,  1.0)   [0.0,   2.0]   central velocity bias
    alpha_s     N(0.2,  1.0)   [0.0,   2.0]   satellite velocity bias

p_max (max central occupation probability) is fixed at 0.33.
All assembly-bias and profile-rank parameters are fixed at 0.

Sampling strategies (set via param_space.sampling in the config):
    lhs    – Latin Hypercube Sampling, uniform within bounds  [default]
             Best coverage of the parameter space for a given N; recommended
             for building emulator training sets.
    prior  – Truncated-Gaussian draws matching the Table II Gaussian priors.
             Useful when you want samples weighted like the prior distribution.

Usage
-----
Prepare simulation subsamples (once per box/redshift):
    python generate_qso_hods.py --prepare_sim --path2config config/abacus_hod.yaml

Generate HOD catalogs:
    python generate_qso_hods.py --path2config config/abacus_hod.yaml \\
        --output output/qso_hods.hdf5 --Nthread 32

Output HDF5 layout
------------------
    /
    ├── attrs              n_runs, param_names, sampling, seed, want_rsd,
    |                      sim_name, z_mock
    ├── param_space/
    │   ├── logM_cut/      attrs: prior_mean, prior_sigma, bounds_lo, bounds_hi
    │   ├── logM1/         attrs: …
    │   └── …
    ├── params             [n_runs × 7]  one row per HOD run
    ├── n_gal              [n_runs]      total QSO count per run
    ├── fixed_params/
    │   └── attrs          p_max, s, s_v, …
    └── catalogs/
        ├── 000000/
        │   ├── attrs      logM_cut, logM1, sigma, alpha, kappa,
        │   |               alpha_c, alpha_s, n_gal
        │   ├── x          [n_gal]  comoving x  [Mpc/h]
        │   ├── y          [n_gal]  comoving y  [Mpc/h]
        │   ├── z          [n_gal]  comoving z  [Mpc/h]
        │   ├── vx         [n_gal]  peculiar vx [km/s]
        │   ├── vy         [n_gal]  peculiar vy [km/s]
        │   ├── vz         [n_gal]  peculiar vz [km/s]
        │   ├── mass       [n_gal]  host halo mass [Msun/h]
        │   └── id         [n_gal]  host halo id
        └── …
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Fixed QSO HOD parameters (held constant for every run)
# ---------------------------------------------------------------------------
FIXED_PARAMS: dict = {
    "p_max":  0.33,   # maximum central occupation probability
    # rank-based satellite profile flexibility (only active if want_ranks=True)
    "s":      0.0,
    "s_v":    0.0,
    "s_p":    0.0,
    "s_r":    0.0,
    # assembly bias amplitudes (only active if want_AB=True)
    "Acent":  0.0,
    "Asat":   0.0,
    "Bcent":  0.0,
    "Bsat":   0.0,
    # incompleteness correction
    "ic":     1.0,
}

_CATALOG_FIELDS = ("x", "y", "z", "vx", "vy", "vz", "mass", "id", "Ncent")


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def _draw_samples(config: dict) -> tuple[list[str], np.ndarray]:
    """
    Draw parameter samples according to the param_space section of the config.

    Returns
    -------
    param_names : list[str]
        Names of the varied parameters, in column order.
    samples : ndarray, shape (n_samples, n_params)
        One row per HOD run.
    """
    ps = config["param_space"]
    n_samples  = int(ps["n_samples"])
    method     = ps.get("sampling", "lhs")
    seed       = int(ps.get("seed", 42))
    params_cfg = ps["params"]
    param_names = list(params_cfg.keys())
    n_params    = len(param_names)

    lo = np.array([params_cfg[p]["bounds"][0] for p in param_names], dtype=float)
    hi = np.array([params_cfg[p]["bounds"][1] for p in param_names], dtype=float)

    if method == "lhs":
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_params, seed=seed)
        samples = qmc.scale(sampler.random(n=n_samples), lo, hi)

    elif method == "prior":
        from scipy.stats import truncnorm
        rng = np.random.default_rng(seed)
        samples = np.empty((n_samples, n_params))
        for j, name in enumerate(param_names):
            mu, sig = params_cfg[name]["prior"]
            a = (lo[j] - mu) / sig
            b = (hi[j] - mu) / sig
            samples[:, j] = truncnorm.rvs(
                a, b, loc=mu, scale=sig, size=n_samples, random_state=rng
            )

    else:
        raise ValueError(
            f"Unknown sampling method {method!r}. Choose 'lhs' or 'prior'."
        )

    return param_names, samples


# ---------------------------------------------------------------------------
# Core routines
# ---------------------------------------------------------------------------

def prepare_simulation(path2config: str) -> None:
    """
    Run AbacusHOD's prepare_sim step to subsample the halo catalog and
    compute environmental quantities.  Run once per simulation box/redshift;
    subsequent HOD runs reuse the saved data.
    """
    from abacusnbody.hod import prepare_sim  # noqa: PLC0415

    print("Preparing simulation subsamples (this may take several minutes)…")
    t0 = time.time()
    prepare_sim.main(path2config)
    print(f"Preparation complete in {time.time() - t0:.1f} s\n")


def generate_hod_samples(
    path2config: str,
    output_file: str,
    Nthread: int = 16,
    want_rsd: bool = True,
    verbose: bool = False,
) -> None:
    """
    Draw HOD parameter samples and generate a QSO mock catalog for each.

    Parameters
    ----------
    path2config : str
        Path to the AbacusHOD YAML configuration file.
    output_file : str
        Path for the output HDF5 file (parent directories are created).
    Nthread : int
        Number of threads for each HOD run.
    want_rsd : bool
        Whether to apply redshift-space distortions.
    verbose : bool
        Pass verbose=True to AbacusHOD (prints per-halo statistics).
    """
    from abacusnbody.hod.abacus_hod import AbacusHOD  # noqa: PLC0415

    with open(path2config) as fh:
        config = yaml.safe_load(fh)

    sim_params        = config["sim_params"]
    HOD_params        = config["HOD_params"]
    clustering_params = config.get("clustering_params", {})

    HOD_params["tracer_flags"]   = {"LRG": False, "ELG": False, "QSO": True}
    HOD_params["want_rsd"]       = want_rsd
    HOD_params["write_to_disk"]  = False

    # Draw parameter samples
    param_names, samples = _draw_samples(config)
    n_runs = len(samples)

    ps = config["param_space"]
    print(f"Sampling method  : {ps.get('sampling', 'lhs')}")
    print(f"Seed             : {ps.get('seed', 42)}")
    print(f"Varied params    : {param_names}")
    print(f"Total HOD runs   : {n_runs}\n")
    print("Parameter bounds (from Table II of Yuan+2023, QSO column):")
    for name in param_names:
        pcfg = ps["params"][name]
        print(f"  {name:10s}  prior N({pcfg['prior'][0]:.1f}, {pcfg['prior'][1]:.1f})"
              f"  bounds {pcfg['bounds']}")
    print()

    # Initialise AbacusHOD with the first sample as placeholder params
    first = dict(zip(param_names, samples[0]))
    HOD_params["QSO_params"] = {**first, **FIXED_PARAMS}

    print("Loading simulation subsamples…")
    t0 = time.time()
    ball = AbacusHOD(sim_params, HOD_params, clustering_params)
    print(f"Loaded in {time.time() - t0:.1f} s\n")

    # Warm-up run to trigger JIT compilation before timing
    print("Warm-up run (JIT compilation)…")
    _ = ball.run_hod(ball.tracers, want_rsd, write_to_disk=False, Nthread=Nthread)
    print("Done.\n")

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_gals       = np.zeros(n_runs, dtype=np.int64)
    log_interval = max(1, n_runs // 50)   # ~50 progress lines total

    # Column header for progress output
    col_w = 8
    hdr_params = "  ".join(f"{n:>{col_w}}" for n in param_names)
    print(f"{'Run':>6}  {hdr_params}  {'N_QSO':>8}  {'dt ms':>7}")
    print("-" * (6 + (col_w + 2) * len(param_names) + 20))

    t_total = time.time()

    with h5py.File(out_path, "w") as hf:
        # Root metadata
        hf.attrs["n_runs"]      = n_runs
        hf.attrs["param_names"] = param_names
        hf.attrs["sampling"]    = ps.get("sampling", "lhs")
        hf.attrs["seed"]        = int(ps.get("seed", 42))
        hf.attrs["want_rsd"]    = want_rsd
        hf.attrs["sim_name"]    = sim_params.get("sim_name", "unknown")
        hf.attrs["z_mock"]      = float(sim_params.get("z_mock", -1.0))

        # Prior / bounds metadata
        grp_ps = hf.create_group("param_space")
        for name in param_names:
            pcfg = ps["params"][name]
            g = grp_ps.create_group(name)
            g.attrs["prior_mean"]  = float(pcfg["prior"][0])
            g.attrs["prior_sigma"] = float(pcfg["prior"][1])
            g.attrs["bounds_lo"]   = float(pcfg["bounds"][0])
            g.attrs["bounds_hi"]   = float(pcfg["bounds"][1])

        # Sample table: row i → parameter vector for run i
        ds = hf.create_dataset("params", data=samples.astype(np.float64))
        ds.attrs["columns"] = param_names

        # Fixed parameters
        grp_fixed = hf.create_group("fixed_params")
        for k, v in FIXED_PARAMS.items():
            grp_fixed.attrs[k] = v

        ds_ngal  = hf.create_dataset("n_gal", shape=(n_runs,), dtype=np.int64)
        grp_cats = hf.create_group("catalogs")

        # ------------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------------
        for i, row in enumerate(samples):
            params_i = dict(zip(param_names, row))

            ball.tracers["QSO"] = {**params_i, **FIXED_PARAMS}

            t_run = time.time()
            mock_dict = ball.run_hod(
                ball.tracers,
                want_rsd,
                write_to_disk=False,
                Nthread=Nthread,
                verbose=verbose,
            )
            dt_ms = (time.time() - t_run) * 1e3

            qso_cat = mock_dict.get("QSO", {})
            n_gal   = int(len(qso_cat.get("x", [])))
            n_gals[i]  = n_gal
            ds_ngal[i] = n_gal

            # Write catalog group
            grp = grp_cats.create_group(f"{i:06d}")
            for name, val in params_i.items():
                grp.attrs[name] = float(val)
            grp.attrs["n_gal"] = n_gal

            for field in _CATALOG_FIELDS:
                arr = qso_cat.get(field)
                if arr is not None and len(arr) > 0:
                    grp.create_dataset(field, data=arr, compression="lzf")

            if i == 0 or (i + 1) % log_interval == 0 or i == n_runs - 1:
                vals_str = "  ".join(f"{v:{col_w}.4f}" for v in row)
                print(f"{i+1:6d}  {vals_str}  {n_gal:8d}  {dt_ms:7.1f}")

    elapsed = time.time() - t_total
    print(f"\nFinished {n_runs} runs in {elapsed:.1f} s  "
          f"({elapsed / n_runs * 1e3:.1f} ms/run)")
    print(f"N_QSO  min={n_gals.min()}  max={n_gals.max()}  "
          f"median={int(np.median(n_gals))}")
    print(f"Output : {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--path2config",
        default="config/abacus_hod.yaml",
        help="Path to the AbacusHOD YAML config  (default: config/abacus_hod.yaml)",
    )
    parser.add_argument(
        "--output",
        default="output/qso_hods.hdf5",
        help="Output HDF5 file path  (default: output/qso_hods.hdf5)",
    )
    parser.add_argument(
        "--Nthread",
        type=int,
        default=16,
        help="Threads per HOD run  (default: 16)",
    )
    parser.add_argument(
        "--no_rsd",
        action="store_true",
        help="Disable redshift-space distortions",
    )
    parser.add_argument(
        "--prepare_sim",
        action="store_true",
        help="Run the prepare_sim step before generating HODs "
             "(only needed once per simulation box)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-halo verbose output from AbacusHOD",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.prepare_sim:
        prepare_simulation(args.path2config)
    generate_hod_samples(
        path2config=args.path2config,
        output_file=args.output,
        Nthread=args.Nthread,
        want_rsd=not args.no_rsd,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

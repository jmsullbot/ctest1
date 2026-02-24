#!/usr/bin/env python3
"""
Generate QSO HOD mock catalogs over a grid of central-occupation parameters.

The satellite occupation parameters (logM1, kappa, alpha) are fixed to
physically reasonable values throughout; only the three parameters that
govern *central* QSO occupation are varied:

    logM_cut  – log10 of the minimum halo mass [log10 Msun/h]
    sigma     – width of the central occupation step function
    p_max     – maximum central occupation probability

Usage
-----
Step 1 – prepare the simulation subsamples (run once per simulation box):

    python generate_qso_hods.py --prepare_sim --path2config config/abacus_hod.yaml

Step 2 – generate the HOD catalog grid:

    python generate_qso_hods.py --path2config config/abacus_hod.yaml \\
        --output output/qso_hods.hdf5 --Nthread 32

Output HDF5 layout
------------------
    /                       root attrs: n_runs, param_names, want_rsd,
    |                                   sim_name, z_mock
    ├── param_grid/
    │   ├── logM_cut        [n_logMcut]  values of logM_cut in the grid
    │   ├── sigma           [n_sigma]    values of sigma in the grid
    │   └── p_max           [n_pmax]     values of p_max in the grid
    ├── params              [n_runs × 3] one row per HOD run;
    |                                    columns = (logM_cut, sigma, p_max)
    ├── n_gal               [n_runs]     total QSO count for each run
    ├── fixed_sat_params/
    │   └── attrs           logM1, kappa, alpha, alpha_c, alpha_s, …
    └── catalogs/
        ├── 000000/
        │   ├── attrs       logM_cut, sigma, p_max, n_gal
        │   ├── x           [n_gal]  comoving x  [Mpc/h]
        │   ├── y           [n_gal]  comoving y  [Mpc/h]
        │   ├── z           [n_gal]  comoving z  [Mpc/h]
        │   ├── vx          [n_gal]  peculiar vx [km/s]
        │   ├── vy          [n_gal]  peculiar vy [km/s]
        │   ├── vz          [n_gal]  peculiar vz [km/s]
        │   ├── mass        [n_gal]  host halo mass [Msun/h]
        │   └── id          [n_gal]  host halo id
        ├── 000001/
        │   └── …
        └── …
"""

import argparse
import itertools
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Satellite (and other non-varied) QSO HOD parameters
# These are held fixed for every run in the grid.
# ---------------------------------------------------------------------------
FIXED_SAT_PARAMS: dict = {
    # satellite occupation
    "logM1":   13.94,   # log10 characteristic satellite halo mass [Msun/h]
    "kappa":    1.0,    # satellite truncation parameter (M_min = kappa * M_cut)
    "alpha":    0.4,    # satellite power-law slope
    # velocity bias
    "alpha_c":  0.0,    # central velocity bias  (0 = no bias)
    "alpha_s":  1.0,    # satellite velocity bias (1 = DM particle speed)
    # rank-based satellite profile flexibility (only matters if want_ranks=True)
    "s":        0.0,
    "s_v":      0.0,
    "s_p":      0.0,
    "s_r":      0.0,
    # assembly bias (only matters if want_AB=True)
    "A_s":      1.0,
    "Acent":    0.0,
    "Asat":     0.0,
    "Bcent":    0.0,
    "Bsat":     0.0,
    # incompleteness correction
    "ic":       1.0,
}

# Fields copied from the AbacusHOD output dict into the HDF5 catalog groups
_CATALOG_FIELDS = ("x", "y", "z", "vx", "vy", "vz", "mass", "id", "Ncent")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path2config: str) -> tuple:
    """Load YAML config and return (sim_params, HOD_params, clustering_params)."""
    with open(path2config) as fh:
        config = yaml.safe_load(fh)
    sim_params = config["sim_params"]
    HOD_params = config["HOD_params"]
    clustering_params = config.get("clustering_params", {})
    return sim_params, HOD_params, clustering_params, config


def _build_param_grid(config: dict) -> tuple[list[str], list[np.ndarray], list[tuple]]:
    """
    Build the Cartesian parameter grid from the config.

    Returns
    -------
    param_names : list of str
    param_values : list of 1-D numpy arrays
    grid_points : list of tuples  (one per HOD run)
    """
    grid_cfg = config.get("param_grid", {})

    # Default fallback values (used if a key is absent from the config)
    defaults = {
        "logM_cut": np.linspace(11.5, 13.5, 20),
        "sigma":    np.linspace(0.10, 1.50, 10),
        "p_max":    np.linspace(0.05, 0.50,  5),
    }

    param_names = ["logM_cut", "sigma", "p_max"]
    param_values = []
    for name in param_names:
        raw = grid_cfg.get(name, defaults[name])
        param_values.append(np.asarray(raw, dtype=float))

    grid_points = list(itertools.product(*param_values))
    return param_names, param_values, grid_points


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def prepare_simulation(path2config: str) -> None:
    """
    Run the AbacusHOD prepare_sim step to subsample the halo catalog and
    compute environmental quantities.  This only needs to be done once per
    simulation box / redshift combination.
    """
    from abacusnbody.hod import prepare_sim  # noqa: PLC0415

    print("Preparing simulation subsamples (this may take several minutes)…")
    t0 = time.time()
    prepare_sim.main(path2config)
    print(f"Preparation complete in {time.time() - t0:.1f} s")


def generate_hod_grid(
    path2config: str,
    output_file: str,
    Nthread: int = 16,
    want_rsd: bool = True,
    verbose: bool = False,
) -> None:
    """
    Generate QSO HOD catalogs over the Cartesian parameter grid defined in
    the config file and write them to an HDF5 file.

    Parameters
    ----------
    path2config : str
        Path to the AbacusHOD YAML configuration file.
    output_file : str
        Path for the output HDF5 file (parent directories are created
        automatically).
    Nthread : int
        Number of threads to use for each HOD run.
    want_rsd : bool
        Whether to include redshift-space distortions.
    verbose : bool
        Pass verbose=True to AbacusHOD (prints per-run halo statistics).
    """
    from abacusnbody.hod.abacus_hod import AbacusHOD  # noqa: PLC0415

    sim_params, HOD_params, clustering_params, config = _load_config(path2config)

    # Force QSO-only mode
    HOD_params["tracer_flags"] = {"LRG": False, "ELG": False, "QSO": True}
    HOD_params["want_rsd"] = want_rsd
    HOD_params["write_to_disk"] = False

    # Build the parameter grid
    param_names, param_values, grid_points = _build_param_grid(config)
    n_runs = len(grid_points)

    print("Parameter grid:")
    for name, vals in zip(param_names, param_values):
        print(f"  {name:10s}: {len(vals):3d} values  "
              f"[{vals.min():.4f}, {vals.max():.4f}]")
    print(f"Total HOD runs : {n_runs}")
    print()

    # Seed AbacusHOD with the first grid point so the constructor has valid
    # QSO parameter values; the actual values are overridden each iteration.
    first_logMcut, first_sigma, first_pmax = grid_points[0]
    HOD_params["QSO_params"] = {
        "p_max":    float(first_pmax),
        "logM_cut": float(first_logMcut),
        "sigma":    float(first_sigma),
        **FIXED_SAT_PARAMS,
    }

    print("Loading simulation subsamples…")
    t0 = time.time()
    ball = AbacusHOD(sim_params, HOD_params, clustering_params)
    print(f"Loaded in {time.time() - t0:.1f} s\n")

    # Warm-up run so Numba/JIT code is compiled before timing starts
    print("Warm-up run (JIT compilation)…")
    _ = ball.run_hod(ball.tracers, want_rsd, write_to_disk=False, Nthread=Nthread)
    print("Warm-up done.\n")

    # Prepare output file
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    params_array = np.array(grid_points, dtype=np.float64)  # [n_runs × 3]
    n_gals = np.zeros(n_runs, dtype=np.int64)

    log_interval = max(1, n_runs // 50)   # print at most ~50 progress lines

    print(f"Writing output to: {out_path}")
    print(f"{'Run':>6}  {'logM_cut':>9}  {'sigma':>7}  {'p_max':>7}  "
          f"{'N_QSO':>8}  {'dt [ms]':>8}")
    print("-" * 62)

    t_total = time.time()

    with h5py.File(out_path, "w") as hf:
        # Root-level metadata
        hf.attrs["n_runs"]      = n_runs
        hf.attrs["param_names"] = param_names
        hf.attrs["want_rsd"]    = want_rsd
        hf.attrs["sim_name"]    = sim_params.get("sim_name", "unknown")
        hf.attrs["z_mock"]      = float(sim_params.get("z_mock", -1.0))

        # Parameter grid arrays
        grp_grid = hf.create_group("param_grid")
        for name, vals in zip(param_names, param_values):
            grp_grid.create_dataset(name, data=vals)

        # Parameter table: row i → (logM_cut, sigma, p_max) for run i
        ds_params = hf.create_dataset("params", data=params_array)
        ds_params.attrs["columns"] = param_names

        # Fixed satellite parameters stored as group attributes
        grp_sat = hf.create_group("fixed_sat_params")
        for k, v in FIXED_SAT_PARAMS.items():
            grp_sat.attrs[k] = v

        # Placeholder for per-run galaxy counts (filled in below)
        ds_ngal = hf.create_dataset("n_gal", shape=(n_runs,), dtype=np.int64)

        grp_cats = hf.create_group("catalogs")

        # ------------------------------------------------------------------
        # Main loop over the parameter grid
        # ------------------------------------------------------------------
        for i, (logM_cut, sigma, p_max) in enumerate(grid_points):
            # Update QSO parameters for this grid point
            ball.tracers["QSO"] = {
                "p_max":    float(p_max),
                "logM_cut": float(logM_cut),
                "sigma":    float(sigma),
                **FIXED_SAT_PARAMS,
            }

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
            n_gal = int(len(qso_cat.get("x", [])))
            n_gals[i] = n_gal
            ds_ngal[i] = n_gal

            # Write catalog to HDF5
            grp = grp_cats.create_group(f"{i:06d}")
            grp.attrs["logM_cut"] = float(logM_cut)
            grp.attrs["sigma"]    = float(sigma)
            grp.attrs["p_max"]    = float(p_max)
            grp.attrs["n_gal"]    = n_gal

            for field in _CATALOG_FIELDS:
                arr = qso_cat.get(field)
                if arr is not None and len(arr) > 0:
                    grp.create_dataset(field, data=arr, compression="lzf")

            if i == 0 or (i + 1) % log_interval == 0 or i == n_runs - 1:
                print(f"{i+1:6d}  {logM_cut:9.4f}  {sigma:7.4f}  {p_max:7.4f}  "
                      f"{n_gal:8d}  {dt_ms:8.1f}")

    elapsed = time.time() - t_total
    print()
    print(f"Finished {n_runs} runs in {elapsed:.1f} s  "
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
        help="Path to the AbacusHOD YAML configuration file "
             "(default: config/abacus_hod.yaml)",
    )
    parser.add_argument(
        "--output",
        default="output/qso_hods.hdf5",
        help="Path for the output HDF5 file (default: output/qso_hods.hdf5)",
    )
    parser.add_argument(
        "--Nthread",
        type=int,
        default=16,
        help="Number of threads for each HOD run (default: 16)",
    )
    parser.add_argument(
        "--no_rsd",
        action="store_true",
        help="Disable redshift-space distortions",
    )
    parser.add_argument(
        "--prepare_sim",
        action="store_true",
        help="Run the prepare_sim step before generating HODs. "
             "Only needed once per simulation box.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose per-halo output from AbacusHOD",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.prepare_sim:
        prepare_simulation(args.path2config)

    generate_hod_grid(
        path2config=args.path2config,
        output_file=args.output,
        Nthread=args.Nthread,
        want_rsd=not args.no_rsd,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

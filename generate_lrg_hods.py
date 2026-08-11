#!/usr/bin/env python3
"""
Generate LRG HOD mock catalogs for a pre-specified parameter table.

Input
-----
A NumPy .npy file (shape N × 14) with columns (in order):
    logM_cut  logM1  logsigma  alpha  alpha_c  alpha_s  kappa
    s  Acent  Asat  Bcent  Bsat  sigma  nbar

Column notes:
  - logsigma (col 2)  log10(σ); converted to  σ = 10**logsigma  before
                       passing to AbacusHOD
  - sigma    (col 12) 10**logsigma — pre-computed in the input file;
                       used directly (avoids repeating the exponentiation)
  - nbar     (col 13) target comoving number density [(Mpc/h)^{-3}];
                       stored in HDF5 for reference but NOT an AbacusHOD
                       parameter

HOD model (AbacusHOD LRG base model — Zheng+2007 with velocity bias,
assembly bias, and satellite-profile extensions):

    <N_cen>(M) = ic/2 · erfc[ (logM_cut − log10 M) / (√2 σ) ]
                 (σ in dex — AbacusHOD's LRG convention uses log10 mass)

    <N_sat>(M) = <N_cen>(M) · [(M − κ M_cut) / M1]^α   for M > κ M_cut

    Assembly bias:  want_AB    = True  →  Acent, Asat, Bcent, Bsat active
    Profile ranks:  want_ranks = True  →  s active (rank-order satellites
                                           by local Vmax)

The parameter table follows the flat priors of eq. (74) of Ivanov et al.
2024 (arXiv:2409.10609): 10500 samples at z=0.5 on the AbacusSummit
fiducial cosmology,

    log10 M_cut ∈ [12, 14]      log10 M1 ∈ [13, 15]
    log10 σ     ∈ [−3.5, 1.0]   α        ∈ [0.5, 1.5]
    α_c ∈ [0, 1]   α_s ∈ [0, 2]   s ∈ [0, 1]*   κ ∈ [0, 1.5]
    A_cen, A_sat, B_cen, B_sat ∈ [−1, 1]

    * the supplied table spans s ∈ [−1, 1], wider than printed in eq. (74);
      both lie within AbacusHOD's valid domain for s.

HOD code reference: Yuan et al. 2022 (arXiv:2110.11412, AbacusHOD)

Usage
-----
Prepare simulation subsamples (once per box/redshift):

    python generate_lrg_hods.py --prepare_sim \\
        --path2config config/lrg_hod.yaml \\
        --params      new_hods.npy

Generate HOD catalogs:

    python generate_lrg_hods.py \\
        --path2config config/lrg_hod.yaml \\
        --params      new_hods.npy \\
        --output      output/lrg_hods.hdf5 \\
        --Nthread     32

Output files
------------
    output/lrg_hods.hdf5       – HDF5 with all runs; random-access per run
    output/catalogs/000000.npy – per-run structured NumPy array

HDF5 layout
-----------
    /
    ├── attrs          n_runs, param_names, want_rsd, sim_name, z_mock,
    │                  params_file
    ├── params         [n_runs × 12]  AbacusHOD parameters per run
    │   └── attrs      columns = param_names
    ├── nbar           [n_runs]  target number density [(Mpc/h)^{-3}]
    ├── logsigma       [n_runs]  log10(σ) as supplied in the input file
    ├── n_gal          [n_runs]  realised galaxy count per run
    ├── fixed_params/
    │   └── attrs      s_v, s_p, s_r, ic
    └── catalogs/
        ├── 000000/
        │   ├── attrs  logM_cut … Bsat, nbar, logsigma, n_gal
        │   ├── x      [n_gal]  comoving x [Mpc/h]  real-space
        │   ├── y      [n_gal]  comoving y [Mpc/h]  real-space
        │   ├── z      [n_gal]  comoving z [Mpc/h]  real-space
        │   ├── z_rsd  [n_gal]  redshift-space z [Mpc/h]  (if want_rsd)
        │   ├── vx/vy/vz  [n_gal]  peculiar velocities [km/s]
        │   ├── mass   [n_gal]  host halo mass [Msun/h]
        │   └── id     [n_gal]  host halo id
        └── …
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# AbacusSummit Planck-2018 flat ΛCDM  (used for z_rsd conversion)
_OM_ABACUS = 0.315192

# Per-galaxy fields extracted from AbacusHOD output (order is preserved in .npy)
_CATALOG_FIELDS = ("x", "y", "z", "vx", "vy", "vz", "mass", "id")

# Column names in the input .npy file  (N × 14)
_INPUT_COLS = [
    "logM_cut", "logM1", "logsigma", "alpha", "alpha_c", "alpha_s",
    "kappa", "s", "Acent", "Asat", "Bcent", "Bsat", "sigma", "nbar",
]

# Parameters that are passed to AbacusHOD's LRG tracer dict.
# sigma = 10**logsigma; nbar and logsigma are stored separately.
_ABACUS_PARAM_NAMES = [
    "logM_cut", "logM1", "sigma", "alpha", "kappa",
    "alpha_c", "alpha_s", "s", "Acent", "Asat", "Bcent", "Bsat",
]

# Fixed LRG HOD parameters not present in the input file
_FIXED_PARAMS: dict = {
    "s_v": 0.0,   # velocity-rank satellite flexibility
    "s_p": 0.0,   # perihelion-rank satellite flexibility
    "s_r": 0.0,   # radial-rank satellite flexibility
    "ic":  1.0,   # incompleteness correction (1 = complete)
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_z_rsd(
    z: np.ndarray,
    vz: np.ndarray,
    z_mock: float,
    Om: float = _OM_ABACUS,
) -> np.ndarray:
    """Plane-parallel redshift-space z-coordinate (z-axis displacement only).

        z_rsd = z_real + v_z / (H(z_mock) / (1 + z_mock))

    H(z) [km/s/(Mpc/h)] = 100 · sqrt(Ω_m (1+z)³ + Ω_Λ)
    """
    E_z = np.sqrt(Om * (1.0 + z_mock) ** 3 + (1.0 - Om))
    H_over_1pz = 100.0 * E_z / (1.0 + z_mock)
    return z + vz / H_over_1pz


def _save_catalog_npy(
    npy_path: Path,
    lrg_cat: dict,
    z_rsd_arr: np.ndarray | None,
) -> None:
    """Save one HOD catalog to a structured NumPy .npy file.

    Fields follow the order in _CATALOG_FIELDS, with z_rsd inserted
    immediately after z when provided.

    Load with::

        cat = np.load("catalogs/000042.npy")
        x   = cat["x"]
    """
    dtype_fields: list[tuple[str, np.dtype]] = []
    for field in _CATALOG_FIELDS:
        arr = lrg_cat.get(field)
        if arr is not None:
            dtype_fields.append((field, np.asarray(arr).dtype))
        if field == "z" and z_rsd_arr is not None:
            dtype_fields.append(("z_rsd", z_rsd_arr.dtype))

    if not dtype_fields:
        np.save(npy_path, np.empty(0, dtype=np.float32))
        return

    n_gal = len(np.asarray(lrg_cat[dtype_fields[0][0]]))
    out   = np.empty(n_gal, dtype=np.dtype(dtype_fields))
    for name, _ in dtype_fields:
        out[name] = z_rsd_arr if name == "z_rsd" else lrg_cat[name]
    np.save(npy_path, out)


def _load_params(params_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the input parameter table.

    Parameters
    ----------
    params_file : str
        Path to the .npy file (shape N × 14).

    Returns
    -------
    abacus_params : ndarray, shape (N, 12)
        Parameters for AbacusHOD, columns = _ABACUS_PARAM_NAMES.
        sigma = 10**logsigma (column 12 of the input, not column 2).
    logsigma_arr : ndarray, shape (N,)
        log10(σ) as supplied.
    nbar_arr : ndarray, shape (N,)
        Target number density [(Mpc/h)^{-3}].
    """
    raw = np.load(params_file)
    if raw.ndim != 2 or raw.shape[1] != 14:
        raise ValueError(
            f"{params_file}: expected shape (N, 14), got {raw.shape}"
        )

    col = {name: raw[:, i] for i, name in enumerate(_INPUT_COLS)}

    # Build AbacusHOD parameter matrix:
    #   use sigma (col 12 = 10**logsigma) directly — avoids re-exponentiation
    abacus_params = np.column_stack([
        col["logM_cut"],
        col["logM1"],
        col["sigma"],      # 10**logsigma
        col["alpha"],
        col["kappa"],
        col["alpha_c"],
        col["alpha_s"],
        col["s"],
        col["Acent"],
        col["Asat"],
        col["Bcent"],
        col["Bsat"],
    ])

    return abacus_params, col["logsigma"], col["nbar"]


# ---------------------------------------------------------------------------
# Core routines
# ---------------------------------------------------------------------------

def prepare_simulation(path2config: str) -> None:
    """Run AbacusHOD's prepare_sim step (once per box/redshift)."""
    from abacusnbody.hod import prepare_sim  # noqa: PLC0415

    print("Preparing simulation subsamples (this may take several minutes)…")
    t0 = time.time()
    prepare_sim.main(path2config)
    print(f"Preparation complete in {time.time() - t0:.1f} s\n")


def generate_hod_samples(
    path2config: str,
    params_file: str,
    output_file: str,
    Nthread: int = 16,
    want_rsd: bool = True,
    verbose: bool = False,
) -> None:
    """
    Run AbacusHOD LRG for every row of the parameter table and save output.

    Parameters
    ----------
    path2config : str
        Path to the AbacusHOD YAML configuration file.
    params_file : str
        Path to the input .npy parameter table (N × 14).
    output_file : str
        Path for the output HDF5 file.
    Nthread : int
        Threads per HOD run.
    want_rsd : bool
        Whether to compute and store redshift-space distortions.
    verbose : bool
        Print per-halo verbose output from AbacusHOD.
    """
    from abacusnbody.hod.abacus_hod import AbacusHOD  # noqa: PLC0415

    with open(path2config) as fh:
        config = yaml.safe_load(fh)

    sim_params        = config["sim_params"]
    HOD_params        = config["HOD_params"]
    clustering_params = config.get("clustering_params", {})

    # Force LRG tracer; always fetch real-space positions from AbacusHOD
    HOD_params["tracer_flags"]  = {"LRG": True, "ELG": False, "QSO": False}
    HOD_params["want_rsd"]      = False   # z_rsd computed explicitly below
    HOD_params["write_to_disk"] = False
    HOD_params["want_AB"]       = True    # assembly bias: Acent/Asat/Bcent/Bsat
    HOD_params["want_ranks"]    = True    # profile rank: s

    z_mock = float(sim_params.get("z_mock", 0.5))

    # Load parameter table
    abacus_params, logsigma_arr, nbar_arr = _load_params(params_file)
    n_runs = len(abacus_params)

    print(f"Parameter file   : {params_file}")
    print(f"Total HOD runs   : {n_runs}")
    print(f"AbacusHOD params : {_ABACUS_PARAM_NAMES}")
    print(f"z_mock           : {z_mock}")
    print(f"want_rsd         : {want_rsd}")
    print()

    # Initialise AbacusHOD with the first row as placeholder params
    first = dict(zip(_ABACUS_PARAM_NAMES, abacus_params[0]))
    HOD_params["LRG_params"] = {**first, **_FIXED_PARAMS}

    print("Loading simulation subsamples…")
    t0 = time.time()
    ball = AbacusHOD(sim_params, HOD_params, clustering_params)
    print(f"Loaded in {time.time() - t0:.1f} s\n")

    # Warm-up run (triggers JIT compilation)
    print("Warm-up run (JIT compilation)…")
    _ = ball.run_hod(ball.tracers, False, write_to_disk=False, Nthread=Nthread)
    print("Done.\n")

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_gals       = np.zeros(n_runs, dtype=np.int64)
    npy_dir      = out_path.parent / "catalogs"
    npy_dir.mkdir(exist_ok=True)
    log_interval = max(1, n_runs // 50)

    # Column header for progress output
    col_w = 9
    hdr_params = "  ".join(f"{n:>{col_w}}" for n in _ABACUS_PARAM_NAMES)
    print(f"{'Run':>6}  {hdr_params}  {'N_LRG':>8}  {'dt ms':>7}")
    print("-" * (6 + (col_w + 2) * len(_ABACUS_PARAM_NAMES) + 20))

    t_total = time.time()

    with h5py.File(out_path, "w") as hf:
        hf.attrs["n_runs"]      = n_runs
        hf.attrs["param_names"] = _ABACUS_PARAM_NAMES
        hf.attrs["want_rsd"]    = want_rsd
        hf.attrs["sim_name"]    = sim_params.get("sim_name", "unknown")
        hf.attrs["z_mock"]      = z_mock
        hf.attrs["params_file"] = str(Path(params_file).name)

        ds = hf.create_dataset("params", data=abacus_params)
        ds.attrs["columns"] = _ABACUS_PARAM_NAMES

        hf.create_dataset("nbar",     data=nbar_arr)
        hf.create_dataset("logsigma", data=logsigma_arr)

        grp_fixed = hf.create_group("fixed_params")
        for k, v in _FIXED_PARAMS.items():
            grp_fixed.attrs[k] = v

        ds_ngal  = hf.create_dataset("n_gal", shape=(n_runs,), dtype=np.int64)
        grp_cats = hf.create_group("catalogs")

        # ------------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------------
        for i, row in enumerate(abacus_params):
            params_i = dict(zip(_ABACUS_PARAM_NAMES, row))

            ball.tracers["LRG"] = {**params_i, **_FIXED_PARAMS}

            t_run = time.time()
            mock_dict = ball.run_hod(
                ball.tracers,
                False,
                write_to_disk=False,
                Nthread=Nthread,
                verbose=verbose,
            )
            dt_ms = (time.time() - t_run) * 1e3

            lrg_cat = mock_dict.get("LRG", {})
            n_gal   = int(len(lrg_cat.get("x", [])))
            n_gals[i]  = n_gal
            ds_ngal[i] = n_gal

            z_rsd_arr: np.ndarray | None = None
            if want_rsd and n_gal > 0:
                _z  = lrg_cat.get("z")
                _vz = lrg_cat.get("vz")
                if _z is not None and _vz is not None:
                    z_rsd_arr = _compute_z_rsd(
                        np.asarray(_z), np.asarray(_vz), z_mock
                    )

            grp = grp_cats.create_group(f"{i:06d}")
            for name, val in params_i.items():
                grp.attrs[name] = float(val)
            grp.attrs["logsigma"] = float(logsigma_arr[i])
            grp.attrs["nbar"]     = float(nbar_arr[i])
            grp.attrs["n_gal"]    = n_gal

            for field in _CATALOG_FIELDS:
                arr = lrg_cat.get(field)
                if arr is not None and len(arr) > 0:
                    grp.create_dataset(field, data=arr, compression="lzf")

            if z_rsd_arr is not None:
                grp.create_dataset("z_rsd", data=z_rsd_arr, compression="lzf")

            _save_catalog_npy(npy_dir / f"{i:06d}.npy", lrg_cat, z_rsd_arr)

            if i == 0 or (i + 1) % log_interval == 0 or i == n_runs - 1:
                vals_str = "  ".join(f"{v:{col_w}.4f}" for v in row)
                print(f"{i+1:6d}  {vals_str}  {n_gal:8d}  {dt_ms:7.1f}")

    elapsed = time.time() - t_total
    print(f"\nFinished {n_runs} runs in {elapsed:.1f} s  "
          f"({elapsed / n_runs * 1e3:.1f} ms/run)")
    print(f"N_LRG  min={n_gals.min()}  max={n_gals.max()}  "
          f"median={int(np.median(n_gals))}")
    print(f"HDF5   : {out_path}  ({out_path.stat().st_size / 1e9:.3f} GB)")
    print(f"NPY    : {npy_dir}/  ({n_runs} files)")


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
        default="config/lrg_hod.yaml",
        help="Path to the AbacusHOD YAML config  (default: config/lrg_hod.yaml)",
    )
    parser.add_argument(
        "--params",
        default="new_hods.npy",
        help="Input .npy parameter table (N × 14)  (default: new_hods.npy)",
    )
    parser.add_argument(
        "--output",
        default="output/lrg_hods.hdf5",
        help="Output HDF5 file  (default: output/lrg_hods.hdf5)",
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
        help="Run the prepare_sim step before generating HODs",
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
        params_file=args.params,
        output_file=args.output,
        Nthread=args.Nthread,
        want_rsd=not args.no_rsd,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

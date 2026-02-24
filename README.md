# QSO HOD Grid Generation

Generate QSO Halo Occupation Distribution (HOD) mock catalogs across a dense
grid of central-occupation parameter values using
[AbacusHOD](https://abacusutils.readthedocs.io/en/latest/hod.html) and the
[AbacusSummit](https://abacussummit.readthedocs.io) N-body simulations.

The simulation data is **not** included here — point the config at wherever
you have the AbacusSummit boxes stored.

---

## HOD model

QSO central occupation follows the Zheng+2007 complementary-error-function form:

```
<N_cen>(M) = p_max/2 * erfc[ ln(M_cut / M) / (sqrt(2) * sigma) ]
```

Satellite occupation:

```
<N_sat>(M) = [(M - kappa * M_cut) / M1]^alpha * <N_cen>(M)
```

### Varied parameters (one grid point per combination)

| Parameter  | Meaning                                      | Default grid              |
|------------|----------------------------------------------|---------------------------|
| `logM_cut` | log₁₀ minimum halo mass [log₁₀ M☉/h]        | 20 pts in [11.5, 13.61]   |
| `sigma`    | width of the central occupation step         | 10 pts in [0.10, 1.50]    |
| `p_max`    | maximum central occupation probability       |  5 pts in [0.05, 0.50]    |

Total default runs: **1 000**

### Fixed satellite parameters

| Parameter | Value | Meaning                                    |
|-----------|-------|--------------------------------------------|
| `logM1`   | 13.94 | log₁₀ characteristic satellite mass        |
| `kappa`   |  1.0  | satellite truncation parameter             |
| `alpha`   |  0.4  | satellite power-law slope                  |

All assembly-bias and velocity-bias parameters are set to zero (no effect).

---

## Setup

```bash
git clone <this-repo>
cd <repo>
pip install -r requirements.txt
```

### Edit the config

Open `config/abacus_hod.yaml` and fill in your local paths:

```yaml
sim_params:
    sim_name: 'AbacusSummit_base_c000_ph000'
    sim_dir:       '/path/to/AbacusSummit/'
    output_dir:    '/path/to/output/mocks/'
    subsample_dir: '/path/to/output/subsample/'
    z_mock: 1.1
```

You can also edit `param_grid` in the same file to change the number of grid
points or their ranges.

---

## Running

### Step 1 — prepare the simulation (once per box/redshift)

This subsamples the halo catalog and computes environmental quantities.
It only needs to be run once; subsequent HOD runs reuse the saved data.

```bash
python generate_qso_hods.py --prepare_sim --path2config config/abacus_hod.yaml
```

### Step 2 — generate the HOD grid

```bash
python generate_qso_hods.py \
    --path2config config/abacus_hod.yaml \
    --output      output/qso_hods.hdf5 \
    --Nthread     32
```

Both steps can be combined by passing `--prepare_sim` and the other flags
together on the same command.

### All options

```
--path2config   Path to the YAML config  (default: config/abacus_hod.yaml)
--output        Output HDF5 file         (default: output/qso_hods.hdf5)
--Nthread       Threads per HOD run      (default: 16)
--no_rsd        Disable redshift-space distortions
--prepare_sim   Run prepare_sim before generating HODs
--verbose       Print per-halo AbacusHOD output
```

---

## Output format

The output is a single HDF5 file with the following layout:

```
qso_hods.hdf5
├── attrs              n_runs, param_names, want_rsd, sim_name, z_mock
├── param_grid/
│   ├── logM_cut       [n_logMcut]  grid values
│   ├── sigma          [n_sigma]    grid values
│   └── p_max          [n_pmax]     grid values
├── params             [n_runs × 3] columns = (logM_cut, sigma, p_max)
├── n_gal              [n_runs]     total QSO count per run
├── fixed_sat_params/
│   └── attrs          logM1, kappa, alpha, alpha_c, alpha_s, …
└── catalogs/
    ├── 000000/
    │   ├── attrs      logM_cut, sigma, p_max, n_gal
    │   ├── x          [n_gal]  comoving x  [Mpc/h]
    │   ├── y          [n_gal]  comoving y  [Mpc/h]
    │   ├── z          [n_gal]  comoving z  [Mpc/h]
    │   ├── vx         [n_gal]  peculiar vx [km/s]
    │   ├── vy         [n_gal]  peculiar vy [km/s]
    │   ├── vz         [n_gal]  peculiar vz [km/s]
    │   ├── mass       [n_gal]  host halo mass [M☉/h]
    │   └── id         [n_gal]  host halo id
    ├── 000001/
    │   └── …
    └── …
```

### Reading the output

```python
import h5py
import numpy as np

with h5py.File("output/qso_hods.hdf5", "r") as f:
    # Parameter table — row i corresponds to catalogs/000i
    params = f["params"][:]          # shape (n_runs, 3)
    col    = list(f["params"].attrs["columns"])
    n_gal  = f["n_gal"][:]

    # Find a specific run
    idx = np.argmin(np.abs(params[:, col.index("logM_cut")] - 12.5))
    print(f"Run {idx}: logM_cut={params[idx,0]:.3f}, N_QSO={n_gal[idx]}")

    # Load its catalog
    cat = f[f"catalogs/{idx:06d}"]
    x, y, z = cat["x"][:], cat["y"][:], cat["z"][:]
```

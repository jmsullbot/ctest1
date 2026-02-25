# QSO HOD Sample Generation

Generate QSO Halo Occupation Distribution (HOD) mock catalogs across a
parameter space drawn from the priors in **Table II of
Yuan et al. 2023** ([arXiv:2306.06314](https://arxiv.org/abs/2306.06314))
using [AbacusHOD](https://abacusutils.readthedocs.io/en/latest/hod.html)
and the [AbacusSummit](https://abacussummit.readthedocs.io) N-body simulations.

The simulation data is **not** included in this repository — point the config
at wherever you have the AbacusSummit boxes stored.

---

## HOD model

QSO central occupation (Zheng+2007 form):

```
<N_cen>(M) = p_max/2 · erfc[ ln(M_cut / M) / (√2 σ) ]
```

Satellite occupation:

```
<N_sat>(M) = [(M − κ M_cut) / M1]^α   for M > κ M_cut, else 0
```

Note: the satellite occupation is **not** modulated by `<N_cen>`. QSO satellites
are treated independently of the central (unlike the LRG form in Zheng+2007).

### Varied parameters — priors from Table II of Yuan+2023 (QSO column)

| Parameter     | Prior              | Bounds        | Description                                      |
|---------------|--------------------|---------------|--------------------------------------------------|
| `logM_cut`    | N(12.7, 1.0)       | [11.2, 14.0]  | log₁₀ min halo mass [log₁₀ M☉/h]                |
| `logM1`       | N(15.0, 1.0)       | [12.0, 16.0]  | log₁₀ satellite characteristic mass              |
| `sigma`       | N(0.5,  0.5)       | [0.0,  3.0]   | width of the central occupation step             |
| `alpha`       | N(1.0,  0.5)       | [0.3,  2.0]   | satellite power-law slope                        |
| `kappa`       | N(0.5,  0.5)       | [0.3,  3.0]   | satellite truncation parameter                   |
| `alpha_c`     | N(1.5,  1.0)       | [0.0,  2.0]   | central velocity bias                            |
| `alpha_s`     | N(0.2,  1.0)       | [0.0,  2.0]   | satellite velocity bias                          |
| `log10_f_ic`  | N(−1.35, 0.5)      | [−2.1, −0.6]  | log₁₀ incompleteness / max central occupation; `p_max = 10^log10_f_ic` (called *f_ic* in Yuan+23 Fig 12) |

### Fixed parameters

All assembly-bias and satellite profile-rank parameters are set to zero.

---

## Sampling strategy

With 8 parameters, a Cartesian grid is impractical. Instead the script draws
`n_samples` points using one of two strategies (set via `param_space.sampling`
in the config):

| Strategy | Description                                                              |
|----------|--------------------------------------------------------------------------|
| `lhs`    | **Latin Hypercube Sampling**, uniform within bounds **(default)**. Gives the best parameter-space coverage for a given N; recommended for building emulator training sets. |
| `prior`  | Truncated-Gaussian draws matching the Table II priors. Useful when you want samples weighted like the prior distribution. |

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

Optionally adjust the number of samples, sampling method, or parameter bounds:

```yaml
param_space:
    n_samples: 1000    # total HOD runs
    sampling: 'lhs'    # 'lhs' or 'prior'
    seed: 42
```

---

## Running

### Step 1 — prepare the simulation (once per box/redshift)

Subsamples the halo catalog and computes environmental quantities.
Only needed once; subsequent runs reuse the saved data.

```bash
python generate_qso_hods.py --prepare_sim --path2config config/abacus_hod.yaml
```

### Step 2 — generate the HOD catalogs

```bash
python generate_qso_hods.py \
    --path2config config/abacus_hod.yaml \
    --output      output/qso_hods.hdf5 \
    --Nthread     32
```

Both steps can be combined by passing `--prepare_sim` together with the other flags.

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

## Helper scripts

### `compute_number_density.py` — comoving number density per run

```bash
python compute_number_density.py output/qso_hods.hdf5 --box_size 2000
```

Prints summary statistics (`n_bar` min / max / median) and optionally saves
an `.npz` file with `n_bar` and `n_gal` arrays:

```bash
python compute_number_density.py output/qso_hods.hdf5 --box_size 2000 \
    --output output/nbar.npz
```

Can also be imported as a module:

```python
from compute_number_density import compute_number_densities
result = compute_number_densities("output/qso_hods.hdf5", box_size_mpch=2000.0)
print(result["n_bar"])   # ndarray, shape (n_runs,), units (Mpc/h)^{-3}
```

### `plot_occupation.py` — occupation functions vs halo mass

```bash
python plot_occupation.py \
    --logM_cut 12.7 --logM1 14.5 --sigma 0.5 \
    --alpha 1.0 --kappa 0.5 --log10_f_ic -1.35 \
    --output occupation.png
```

Produces a log–log plot of `<N_cen>(M)`, `<N_sat>(M)`, and `<N_tot>(M)`.
Can also be used as a module (see `plot_occupation` and `n_cen` / `n_sat`
functions).

---

## Output format

Two outputs are written automatically:

| Output | Best for |
|--------|----------|
| `qso_hods.hdf5` | Random access to individual runs; preserves all metadata |
| `catalogs/NNNNNN.npy` | Fast per-run structured NumPy arrays; no h5py dependency |

### HDF5 layout

```
qso_hods.hdf5
├── attrs              n_runs, param_names, sampling, seed, want_rsd,
│                      sim_name, z_mock
├── param_space/
│   ├── logM_cut/      attrs: prior_mean, prior_sigma, bounds_lo, bounds_hi
│   ├── logM1/         attrs: …
│   └── …              (one group per varied parameter)
├── params             [n_runs × 8]  parameter vectors; columns = param_names
├── n_gal              [n_runs]      total QSO count per run
├── fixed_params/
│   └── attrs          s, s_v, s_p, s_r, Acent, Asat, Bcent, Bsat, ic
└── catalogs/
    ├── 000000/
    │   ├── attrs      logM_cut, logM1, sigma, alpha, kappa, alpha_c,
    │   │               alpha_s, log10_f_ic, n_gal
    │   ├── x          [n_gal]  comoving x     [Mpc/h]  (real-space)
    │   ├── y          [n_gal]  comoving y     [Mpc/h]  (real-space)
    │   ├── z          [n_gal]  comoving z     [Mpc/h]  (real-space)
    │   ├── z_rsd      [n_gal]  redshift-space z [Mpc/h] (only if want_rsd=True)
    │   ├── vx         [n_gal]  peculiar vx    [km/s]
    │   ├── vy         [n_gal]  peculiar vy    [km/s]
    │   ├── vz         [n_gal]  peculiar vz    [km/s]
    │   ├── mass       [n_gal]  host halo mass [M☉/h]
    │   └── id         [n_gal]  host halo id
    └── …
```

### Per-catalog `.npy` layout

One file per HOD run, written to `output/catalogs/` immediately after each run
completes.  Each file is a **1-D structured NumPy array** (one element per galaxy):

```
catalogs/000042.npy   (load with cat = np.load("catalogs/000042.npy"))
└── structured array, fields:
    x       [n_gal]   comoving x        [Mpc/h]  real-space
    y       [n_gal]   comoving y        [Mpc/h]  real-space
    z       [n_gal]   comoving z        [Mpc/h]  real-space
    z_rsd   [n_gal]   redshift-space z  [Mpc/h]  only if want_rsd=True
    vx      [n_gal]   peculiar vx       [km/s]
    vy      [n_gal]   peculiar vy       [km/s]
    vz      [n_gal]   peculiar vz       [km/s]
    mass    [n_gal]   host halo mass    [M☉/h]
    id      [n_gal]   host halo id
```

### Reading the output

**NumPy (per-run `.npy`):**

```python
import numpy as np

cat = np.load("output/catalogs/000042.npy")

x, y, z = cat["x"], cat["y"], cat["z"]
z_rsd   = cat["z_rsd"]   # redshift-space z (if want_rsd=True)
mass    = cat["mass"]
```

**HDF5:**

```python
import h5py
import numpy as np

with h5py.File("output/qso_hods.hdf5", "r") as f:
    params = f["params"][:]        # shape (n_runs, 8)
    cols   = list(f["params"].attrs["columns"])
    n_gal  = f["n_gal"][:]

    # Find the run closest to a target parameter value
    idx = np.argmin(np.abs(params[:, cols.index("logM_cut")] - 12.5))
    print(f"Run {idx}: logM_cut={params[idx, 0]:.3f}, N_QSO={n_gal[idx]}")

    cat = f[f"catalogs/{idx:06d}"]
    x, y, z = cat["x"][:], cat["y"][:], cat["z"][:]
```

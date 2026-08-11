# QSO HOD Sample Generation

> **This branch (`old_LRG`) also contains an LRG pipeline** —
> `generate_lrg_hods.py` + `config/lrg_hod.yaml` — which runs AbacusHOD's
> LRG tracer over a pre-supplied parameter table following
> Ivanov et al. 2024 ([arXiv:2409.10609](https://arxiv.org/abs/2409.10609)).
> See [LRG HOD Generation](#lrg-hod-generation-this-branch) below.

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

---

# LRG HOD Generation (this branch)

`generate_lrg_hods.py` runs AbacusHOD's **LRG** tracer (with velocity bias,
assembly bias, and the satellite-profile parameter `s`) once per row of a
pre-supplied parameter table — no sampling happens at runtime.

The repo ships **`lrg_params_eq74.npy`** — 10500 parameter vectors drawn
(Latin Hypercube, seed 42) from the flat priors of **eq. (74) of Ivanov,
Obuljen, Cuesta-Lazaro & Toomey 2024**
([arXiv:2409.10609](https://arxiv.org/abs/2409.10609)), who generated
10500 LRG mocks at z = 0.5 on the AbacusSummit fiducial cosmology with
AbacusHOD (Yuan et al. 2022,
[arXiv:2110.11412](https://arxiv.org/abs/2110.11412)). Regenerate or
resample the table with `make_lrg_param_table.py` (`--method uniform`
gives paper-style independent uniform draws instead of LHS).

## Input parameter table

A `.npy` array of shape (N, 14). Columns, in order:

| # | Column | eq. (74) prior | Passed to AbacusHOD? |
|---|--------|----------------|----------------------|
| 0 | `logM_cut` | [12, 14] | yes |
| 1 | `logM1` | [13, 15] | yes |
| 2 | `logsigma` | [−3.5, 1.0] | no — stored for reference |
| 3 | `alpha` | [0.5, 1.5] | yes |
| 4 | `alpha_c` | [0, 1] | yes |
| 5 | `alpha_s` | [0, 2] | yes |
| 6 | `kappa` | [0, 1.5] | yes |
| 7 | `s` | [0, 1] | yes |
| 8 | `Acent` | [−1, 1] | yes |
| 9 | `Asat` | [−1, 1] | yes |
| 10 | `Bcent` | [−1, 1] | yes |
| 11 | `Bsat` | [−1, 1] | yes |
| 12 | `sigma` | = 10^logsigma | yes (used directly) |
| 13 | `nbar` | NaN — measured output, not an input | no — stored for reference |

All twelve varied parameters, including `s`, use exactly the eq. (74)
ranges.

Fixed parameters (not in the table): `s_v = s_p = s_r = 0`, `ic = 1`.

## Running on NERSC Perlmutter

AbacusSummit is hosted at NERSC under the DESI CFS space
(`/global/cfs/cdirs/desi/cosmosim/Abacus/`, requires `desi` group
membership — otherwise pull the box into your own space with Globus).
`config/lrg_hod.yaml` already points there.

### One-time setup

No GitHub keys needed on Perlmutter — copy the repo over as a zip:

```bash
# On your laptop (or grab a zip that was already made for you):
git archive --format=zip --prefix=ctest1/ -o ctest1_old_LRG.zip old_LRG
scp ctest1_old_LRG.zip <user>@perlmutter.nersc.gov:~/

# On Perlmutter:
unzip ctest1_old_LRG.zip && cd ctest1

# tell the scripts where your conda lives (once):
cp slurm/env.local.sh.example slurm/env.local.sh
conda info --base                 # <- put this path in CONDA_BASE
${EDITOR:-nano} slurm/env.local.sh

bash slurm/setup_env.sh           # creates the env + installs deps
```

Run `setup_env.sh` on a **login node**, not via `sbatch`. It is idempotent
— safe to re-run if it fails partway.

`slurm/env.local.sh` is gitignored, so your paths survive re-unzipping the
repo. It works with **your own conda installation** (the default), NERSC's
conda module (`USE_CONDA_MODULE=1`), or your own activation script
(`ENV_READY=1`) — see `slurm/env.local.sh.example` for all three.

> **Why this is needed:** batch jobs inherit your `PATH` but not shell
> functions, so a bare `conda activate` fails inside a job script even
> when it works in your login shell. The scripts source
> `$CONDA_BASE/etc/profile.d/conda.sh` first to define it.

(`git clone` + `git checkout old_LRG` works too if you have keys set up.)

Edit `config/lrg_hod.yaml`: replace `<u>/<user>` in the `$PSCRATCH` paths
with your username, and confirm `sim_name` / `z_mock` match the snapshot
you want (z = 0.5 matches Ivanov+2024).

### Submit

The generation is embarrassingly parallel across table rows and runs as a
SLURM **job array** (16 shards × ~20–40 min, 1 h walltime request; short
jobs also backfill much faster in the queue than one long job):

```bash
# once per box + redshift — builds halo subsamples, wait for it to finish:
sbatch slurm/perlmutter_prepare_sim.sbatch

# then the array (finishes in well under an hour of wall-clock):
sbatch slurm/perlmutter_lrg_hods_array.sbatch
```

The scripts work whether you submit from the repo root or from `slurm/`.
From anywhere else, point them at the repo:

```bash
sbatch --export=ALL,REPO_DIR=/path/to/ctest1 slurm/perlmutter_prepare_sim.sbatch
```

Other overrides, passed the same way via `--export=ALL,...` or set once in
`slurm/env.local.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `CONDA_BASE` | auto-detected | conda root (dir containing `etc/profile.d/conda.sh`) |
| `CONDA_ENV` | `abacus` | env name, or a full prefix path |
| `USE_CONDA_MODULE` | `0` | set `1` to `module load conda` (NERSC's conda) |
| `ENV_READY` | `0` | set `1` if `env.local.sh` activates the env itself |
| `REPO_DIR` | submit dir | repo location |
| `OUTDIR` | `$PSCRATCH/lrg_hods` | where catalogs are written |

Each array task writes per-run `.npy` catalogs named by **global row
index** (no collisions) and its own small metadata-only
`lrg_hods_rowsSTART-END.hdf5` shard; the `row_index` dataset in each
shard maps entries back to table rows. To change the shard count, edit
both `--array=0-15` and `NSHARDS=16`. A serial fallback
(`slurm/perlmutter_lrg_hods.sbatch`) runs all rows in one job (~5–8 h).

### Storage warning

With the base (2 Gpc/h)³ box and the Ivanov+2024 number densities
(median n̄ ≈ 1.1×10⁻³ (Mpc/h)⁻³ → ~9M galaxies/run, up to 66M), the
per-run `.npy` catalogs for all 10500 runs total roughly **8 TB** —
a large fraction of the default 20 TB `$PSCRATCH` quota, and `$PSCRATCH`
is purged (~8 weeks). Consider computing your summary statistics inside
the loop instead of keeping every catalog, or keeping catalogs for a
subset of rows only.

## Output

All galaxy data lives in the per-run structured `.npy` catalogs
(`catalogs/NNNNNN.npy`, named by global table row) with fields
`x, y, z, z_rsd, vx, vy, vz, mass, id` (real-space positions; `z_rsd`
computed plane-parallel along z at `z_mock`). The HDF5 file is a small
**metadata-only index** (a few MB): the parameter table, `row_index`,
`n_gal`, `nbar`, `logsigma`, and the run configuration — no per-galaxy
datasets.

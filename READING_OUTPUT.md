# Reading the LRG HOD pipeline output

Output lives in `$PSCRATCH/lrg_hods/` and comes in two parts:

| What | Where | Size |
|---|---|---|
| Galaxy catalogs | `catalogs/NNNNNN.npy` — one per HOD run, named by **global** table row | ~0.4 GB median, up to ~3 GB |
| Metadata index | `lrg_hods_rowsSTART-END.hdf5` — one per job-array shard | a few MB total |

The join between them is the run number: run `N` in the parameter table is
`catalogs/{N:06d}.npy`, and appears in whichever HDF5 shard covers row `N`.

There may be two such directories, produced from the **same parameter
table** so catalogs correspond row-for-row:

| Run | Box | Directory | Shard files | Box side |
|---|---|---|---|---|
| base | `AbacusSummit_base_c000_ph000` | `$PSCRATCH/lrg_hods/` | `lrg_hods_rows*.hdf5` | 2000 Mpc/h |
| small | `AbacusSummit_small_c000_ph3000` | `$PSCRATCH/lrg_hods_small/` | `lrg_hods_small_rows*.hdf5` | 500 Mpc/h |

Read the box side from the `box_size` attribute rather than hardcoding it
(see §3). Small-box catalogs have ~64× fewer galaxies for the same HOD
parameters, and rows with high `logM_cut` may legitimately be empty.

---

## 1. A single catalog

Each `.npy` is a **structured array** — one element per galaxy, fields
accessed by name.

```python
import numpy as np

cat = np.load("catalogs/000042.npy")
print(len(cat))              # number of galaxies
print(cat.dtype.names)
# ('x', 'y', 'z', 'z_rsd', 'vx', 'vy', 'vz', 'mass', 'id')

x, y, z    = cat["x"], cat["y"], cat["z"]       # real-space comoving [Mpc/h]
z_rsd      = cat["z_rsd"]                        # redshift-space z   [Mpc/h]
vx, vy, vz = cat["vx"], cat["vy"], cat["vz"]     # peculiar velocity  [km/s]
mass       = cat["mass"]                         # host halo mass  [Msun/h]
halo_id    = cat["id"]                           # host halo id
```

### Field reference

| Field | Units | Notes |
|---|---|---|
| `x`, `y`, `z` | Mpc/h | comoving, **real space** |
| `z_rsd` | Mpc/h | redshift-space z; present only when RSD is enabled |
| `vx`, `vy`, `vz` | km/s | peculiar velocity |
| `mass` | M☉/h | host halo mass |
| `id` | — | host halo id |

### Positions for clustering codes

Most clustering codes want an `(N, 3)` array:

```python
pos_real = np.column_stack([x, y, z])
pos_rsd  = np.column_stack([x, y, z_rsd])       # RSD applied along z only
```

`x` and `y` are identical in both. The RSD displacement is plane-parallel
along the z axis, which is why there is a separate `z_rsd` column rather
than a modified `z`:

```
z_rsd = z + v_z / (H(z_mock) / (1 + z_mock))
H(z) [km/s/(Mpc/h)] = 100 * sqrt(Om (1+z)^3 + OL),  Om = 0.315192
```

No periodic wrapping is applied, so `z_rsd` can fall slightly outside
`[0, L_box)`. Wrap it yourself if your estimator needs that:

```python
z_rsd_wrapped = np.mod(z_rsd, 2000.0)           # L_box = 2000 Mpc/h
```

### Large files

At high number density a single run reaches ~3 GB. Memory-map rather than
loading it whole:

```python
cat = np.load("catalogs/000042.npy", mmap_mode="r")
heavy = cat["mass"][:] > 1e14        # only the pages you touch are read
```

---

## 2. The metadata HDF5

One shard per job-array task. Contents:

```python
import h5py

with h5py.File("lrg_hods_rows000000-000657.hdf5", "r") as f:
    cols      = list(f["params"].attrs["columns"])
    params    = f["params"][:]       # (n_runs, 12) HOD parameters
    row_index = f["row_index"][:]    # global table row of each entry
    n_gal     = f["n_gal"][:]        # realised galaxy count per run
    logsigma  = f["logsigma"][:]     # log10(sigma) as supplied
    z_mock    = f.attrs["z_mock"]
```

| Dataset / attr | Shape | Meaning |
|---|---|---|
| `params` | (n_runs, 12) | HOD parameters; column names in `.attrs["columns"]` |
| `row_index` | (n_runs,) | global table row — the link to `catalogs/NNNNNN.npy` |
| `n_gal` | (n_runs,) | galaxies actually produced |
| `nbar` | (n_runs,) | **NaN by design** — see below |
| `logsigma` | (n_runs,) | log10(σ) from the input table |
| attrs | — | `n_runs`, `n_runs_total`, `row_start`, `row_end`, `param_names`, `want_rsd`, `sim_name`, `z_mock`, `params_file` |

The 12 parameter columns, in order:

```
logM_cut  logM1  sigma  alpha  kappa  alpha_c  alpha_s  s  Acent  Asat  Bcent  Bsat
```

Note `sigma` here is already `10**logsigma` — that is what was passed to
AbacusHOD. The `logsigma` dataset keeps the original log-space value.

### Parameters for one run

```python
k = np.where(row_index == 42)[0][0]      # position within this shard
print(dict(zip(cols, params[k])))
print("n_gal:", n_gal[k])
```

---

## 3. Combining all shards

Shards finish out of order, so sort by `row_index`:

```python
import glob
import h5py
import numpy as np

rows, pars, ngals = [], [], []
for fn in sorted(glob.glob("*_rows*.hdf5")):          # base: lrg_hods_rows*, small: lrg_hods_small_rows*
    with h5py.File(fn, "r") as f:
        rows.append(f["row_index"][:])
        pars.append(f["params"][:])
        ngals.append(f["n_gal"][:])
        # Box side [Mpc/h]. Shards written before this attribute existed
        # (the first base-box run) lack it -> fall back to the base box.
        L_BOX = float(f.attrs.get("box_size", 2000.0))

rows, pars, ngals = map(np.concatenate, (rows, pars, ngals))
order = np.argsort(rows)
rows, pars, ngals = rows[order], pars[order], ngals[order]

assert np.array_equal(rows, np.arange(len(rows)))    # no missing runs
```

That assertion is a cheap completeness check: if a job-array task failed,
its rows are absent and this trips.

### Number density

**`nbar` in the input table and the HDF5 is NaN on purpose.** Number
density is a *measured output* of each HOD run, not an input. Compute it:

```python
nbar = ngals / L_BOX ** 3            # (Mpc/h)^-3; L_BOX read from the HDF5 above
                                     # (2000 for the base box, 500 for a small box)
print(f"median {np.median(nbar):.3e}, range {nbar.min():.3e} .. {nbar.max():.3e}")
```

For reference, the eq. (74) priors of Ivanov et al. 2024
([arXiv:2409.10609](https://arxiv.org/abs/2409.10609)) give a median around
1.1e-3 (Mpc/h)^-3, spanning roughly 1.1e-5 to 8.3e-3.

---

## 4. Full example: iterate over runs

```python
import glob
import h5py
import numpy as np

# Build the run table once
rows, pars, ngals = [], [], []
for fn in sorted(glob.glob("*_rows*.hdf5")):
    with h5py.File(fn, "r") as f:
        cols  = list(f["params"].attrs["columns"])
        L_BOX = float(f.attrs.get("box_size", 2000.0))   # Mpc/h (fallback: base box)
        rows.append(f["row_index"][:])
        pars.append(f["params"][:])
        ngals.append(f["n_gal"][:])

rows, pars, ngals = map(np.concatenate, (rows, pars, ngals))
order = np.argsort(rows)
rows, pars, ngals = rows[order], pars[order], ngals[order]

# Select the runs you care about — e.g. near the DESI LRG density
nbar = ngals / L_BOX ** 3
sel  = np.where(np.abs(nbar - 5e-4) / 5e-4 < 0.1)[0]
print(f"{len(sel)} runs within 10% of nbar = 5e-4")

for i in sel[:5]:
    run = rows[i]
    cat = np.load(f"catalogs/{run:06d}.npy", mmap_mode="r")
    pos = np.column_stack([cat["x"], cat["y"], cat["z_rsd"]])
    print(f"run {run:5d}  n_gal={ngals[i]:>9,}  nbar={nbar[i]:.3e}  "
          f"logM_cut={pars[i, cols.index('logM_cut')]:.3f}")
    # ... your estimator here, e.g. P(k) or xi(r) from `pos`
```

---

## Gotchas

- **`nbar` is NaN** in the parameter table and HDF5 — derive it from
  `n_gal / L_box**3`.
- **Filenames use the global row**, not a per-shard index, so
  `catalogs/009999.npy` is table row 9999 regardless of which array task
  produced it.
- **`z_rsd` is not wrapped** into the box; wrap it yourself if needed.
- **`sigma` vs `logsigma`**: the `params` column is `10**logsigma`.
- **`$PSCRATCH` is purged** (~8 weeks) and is not backed up. Move anything
  you intend to keep to CFS.

---

## 5. Cosmology and growth factors at z = 0.5

Both boxes (`base_c000_ph000`, `small_c000_ph3000`) share the fiducial
`c000` cosmology and an identical growth history:

| Quantity | Value | Notes |
|---|---|---|
| Ω_m, Ω_DE | 0.315192, 0.684808 | flat; ω_b = 0.02237, ω_cdm = 0.12, ω_ncdm = 0.0006442 (one 0.06 eV ν), n_s = 0.9649, w = −1 |
| z_init | 99 | IC density field is δ at this redshift |
| **D(z=0.5) / D(z=0)** | **0.769424** | linear growth relative to today |
| **D(z=0.5) / D(z_init)** | **59.911485** | multiply the `ic_dens` field by this to get δ_lin at z = 0.5 |
| `Growth` (D with D→a at early times) | 0.606627 | Abacus's own normalisation; D(z=0) ≈ 0.7884 in this convention |
| **f(z=0.5) = dlnD/dlna** | **0.759070** | for RSD / Kaiser; Ω_m(z=0.5)^0.55 = 0.761 as a check |
| E(z=0.5) = H/H₀ | 1.322339 | `HubbleNow` in the header |

These come from the simulation headers, which abacusutils bundles — no file
access needed:

```python
from abacusnbody.metadata import get_meta
m = get_meta("AbacusSummit_base_c000_ph000", redshift=0.5)
m["Growth"], m["f_growth"], m["GrowthTable"]     # GrowthTable: {z: D(z)}, D(z_init)=1
```

The same header (with `GrowthTable`) sits inside every `halo_info_*.asdf`
and in the IC file, as `af["header"]`.

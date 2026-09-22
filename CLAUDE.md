# Project notes for Claude

Standing constraints for this repo. Read before proposing setup, config,
or job-submission changes.

## Environment constraints (do not violate)

1. **Python comes from a locally installed conda, not a module.**
   Never emit `module load python` / `module load conda` as the default
   path. The user runs their own conda installation. Batch jobs must
   source `$CONDA_BASE/etc/profile.d/conda.sh` before `conda activate`,
   because job scripts inherit `PATH` but not shell functions.
   Site-local settings live in `slurm/env.local.sh` (gitignored; see
   `slurm/env.local.sh.example`). `USE_CONDA_MODULE=1` exists only as an
   opt-in escape hatch — never the default.

2. **Never assume DESI access — neither data nor allocation.**
   The user is *not* in the `desi` unix group, and `-A desi` is *not* a
   usable SLURM allocation for them. Never emit either as a default.
   - Unreadable: `/global/cfs/cdirs/desi/cosmosim/Abacus/`
     (collaboration-only).
   - The sbatch scripts leave `#SBATCH -A` commented out so SLURM uses
     the user's own default allocation.

3. **Use only public resources.**
   AbacusSummit is already on disk at NERSC, world-readable, at

       /global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/

   The `/public/` component is what makes it open — `desi` in the path is
   just where it is hosted, and does **not** imply membership is needed.
   Read-only mount for compute nodes:
   `/dvs_ro/cfs/cdirs/desi/public/cosmosim/AbacusSummit/`.
   Layout: `<sim_name>/halos/z0.500/halo_info/halo_info_000.asdf`, …
   No Globus download is required. Not every box is staged on disk (some
   are HPSS-tape only), so check before submitting. Same principle for any
   other data or software: no collaboration-gated sources.

## Dependency gotchas

- **Corrfunc is mandatory even though we never compute clustering.**
  `abacusnbody/hod/abacus_hod.py` line ~23 imports
  `..analysis.tpcf_corrfunc` at *module level*, and that module raises
  `ImportError` if Corrfunc is absent. So `from abacusnbody.hod.abacus_hod
  import AbacusHOD` fails outright; no config flag avoids it. Only
  `calc_xirppi_fast` / `calc_wp_fast` / `calc_multipole_fast` actually use
  it, and those live in `compute_*` methods this pipeline never calls.
  Install with `conda install -c conda-forge corrfunc` — **not** pip,
  which builds from source and needs GSL ≥ 2.4 plus a compiler. Corrfunc
  is deliberately absent from `requirements.txt` so a build failure can't
  abort the whole pip step.
- `prepare_sim` does **not** import Corrfunc, so the prep job can succeed
  while every generation task fails. Preflight checks in
  `slurm/_common.sh` must cover what the *generation* step imports.

## Platform

- NERSC **Perlmutter** (not Harvard Cannon — an earlier version of the
  docs targeted Cannon; that was reverted).
- Scratch: `/pscratch/sd/<u>/<user>/`, purged ~8 weeks, ~20 TB quota.
- Submit jobs from the repo root; the sbatch scripts also tolerate being
  submitted from `slurm/` and accept `REPO_DIR=` for anywhere else.

## Science context

- **LRG pipeline** (`generate_lrg_hods.py`, branch `old_LRG`): runs
  AbacusHOD's LRG tracer over `lrg_params_eq74.npy` — 10500 rows drawn by
  `make_lrg_param_table.py` from the flat priors of **eq. (74) of Ivanov,
  Obuljen, Cuesta-Lazaro & Toomey 2024 (arXiv:2409.10609)**, at z = 0.5 on
  the AbacusSummit fiducial cosmology. `want_AB=True`, `want_ranks=True`.
  The AbacusHOD code paper is Yuan et al. 2022 (arXiv:2110.11412) — do not
  attribute 2409.10609 to Yuan.
- **QSO pipeline** (`generate_qso_hods.py`, branch `main`): samples 8
  parameters via LHS from Table II of Yuan et al. 2023 (arXiv:2306.06314).
  QSO satellites are *not* modulated by `<N_cen>`.
- Parameter-table columns are `logM_cut logM1 logsigma alpha alpha_c
  alpha_s kappa s Acent Asat Bcent Bsat sigma nbar`. `sigma = 10**logsigma`
  is passed to AbacusHOD; `logsigma` and `nbar` are metadata only. `nbar`
  is a *measured output*, NaN in the generated table.

## Output policy

- All galaxy data goes to per-run `.npy` files named by **global** table
  row (`catalogs/NNNNNN.npy`), so job-array shards never collide.
- The HDF5 is **metadata only** (params, `row_index`, `n_gal`, `nbar`,
  `logsigma`, attrs). Do not reintroduce per-galaxy datasets into it —
  that was removed deliberately; it doubled runtime and storage.
- Full run at base-box densities is ~8 TB of `.npy`. Flag this before
  suggesting anything that increases it.
- The HDF5 attrs carry `box_size` (Mpc/h, read from `ball.lbox`). Shards
  from the *first* base-box run predate this attribute — readers must fall
  back to 2000 when it is absent. Never hardcode the box size.

## Small-box companion run

- `config/lrg_hod_small.yaml` + `slurm/perlmutter_prepare_sim_small.sbatch`
  + `slurm/perlmutter_lrg_hods_small_array.sbatch` regenerate the **same
  10500 rows** on `AbacusSummit_small_c000_ph3000` (500 Mpc/h, 1728³
  particles, same particle mass as base, 64× less volume). Output goes to
  `$PSCRATCH/lrg_hods_small/`, never the base directory. Catalog `NNNNNN`
  corresponds row-for-row across the two runs.
- Small boxes have particle subsamples at z = 1.4, 1.1, 0.8, 0.5, 0.2.
  Phases ph3000–ph4999 are irregularly numbered (some absent) — `ls` the
  public path before assuming a phase exists.
- **Small boxes live under `<root>/small/`** (and their cleaning under
  `<root>/cleaning/small/<sim>/`), per abacusutils'
  `compaso_halo_catalog._setup_file_paths`. `sim_dir` for a small run is
  therefore `<root>/small/`, not `<root>/`. A bare
  `ls <root>/AbacusSummit_small_…` fails with "No such file" for this
  reason, not because the box is absent.
- `prepare_sim` reads only `halo_info`, `halo_rv_A`, `field_rv_A` (no B
  subsamples, no PIDs) plus `cleaning/…/{cleaned_halo_info,cleaned_rvpid}`.
  `slurm/globus_fetch_small.sh` transfers exactly that set from the public
  AbacusSummit Globus collection (found by display name; UUIDs are not
  hardcoded — docs.nersc.gov and abacusnbody.org are egress-blocked from
  the sandbox, so they were never verified here).
- Same cell size across boxes means N_small = N_base / 4 for any mesh
  (576³↔144³, 1152³↔288³, 1024³↔256³). Match cell size, not N.
- The sbatch scripts take `CONFIG=` and `OUTDIR=` overrides via
  `--export=ALL,...`; AbacusHOD writes subsamples under
  `subsample_dir/<sim_name>/z0.500/`, so different sims never collide.

## Working style

- The user runs everything on Perlmutter and often hand-edits files there.
  When changing scripts, offer the inline diff, not only a zip.
- Deliver the repo as a zip (`git archive`) — the user does not use
  GitHub keys on Perlmutter.
- Development happens on branch `old_LRG`; push there.

## Centrals / satellites

- AbacusHOD returns the cen/sat split only as a scalar `Ncent`: **the
  first `Ncent` galaxies of a catalog are centrals**, the rest satellites.
  There is no per-galaxy flag. `generate_lrg_hods.py` now records
  `n_cent` and `f_sat` in the HDF5; the first base and small runs predate
  this and need `compute_fsat.py`, which recovers `Ncent` from the single
  drop in the halo-id sequence at the cen→sat boundary (validated per row
  by counting drops). Do not recover it by counting unique halo ids — LRG
  satellites are not conditioned on a central existing (only ELG are), so
  that undercounts.
- `nbar` depends on *all* occupation parameters (`alpha`, `kappa`, `logM1`
  set the satellite count), not only `logM_cut`/`logM1`/`logsigma`.
  Matching on the latter three alone does not reproduce `nbar`; rank on
  measured `nbar` and `f_sat` via `find_nearest_hod.py --hdf5-dir`.

## Initial conditions

- The public disk copy has **no `ic/` directory** for `base_c000_ph000`,
  and the HPSS copy (`/nersc/projects/desi/cosmosim/Abacus/`) returns
  `HPSS_EACCES` for this user. The ICs were obtained via the Globus web
  interface and live at (user-chosen, not the official layout):
      `$PSCRATCH/abacus_ics_000/`        base: `ic_{dens,disp}_N{576,1152,2048}.asdf`
      `$PSCRATCH/small_abacus_ics_000/`  small: `ic_{dens,disp}_N576.asdf`
  plus `checksums.crc32` in each. `check_ics.py` verifies and inspects them.
- **N means cells across the box, so N576 is 3.47 Mpc/h on the base box
  but 0.87 Mpc/h on the small box.** The small N576 has no cell-matched
  base partner (that would be N144, not shipped); closest is base N2048
  (0.98 Mpc/h). Coarsen a GRF by k-space truncation, never by real-space
  averaging.
- The bundled header (abacusutils metadata) carries the full IC recipe:
  `ZD_Seed = 12321`, `ZD_Version = 2`, `ZD_NumBlock = 384`, z_init = 99,
  PLT on, and the CLASS input spectrum — so the field can in principle be
  regenerated with the public zeldovich-PLT code if a box's ICs are ever
  unavailable (unverified whether reduced-N output is phase-identical).

## Field-level P_err results (hod 2155, z=0.5)

- `fit_perr.py` / `compare_perr.py` plot n̄P_err(k,μ) and fit
  1 − D·W(kR)·exp(−k²μ²ℓ_F²), W = exp(−k²R²/2) by default (top-hat
  optional); μ-bin factor averaged analytically over the bin; bins
  weighted σ ∝ 1/k (no per-bin errors are stored). Result files are
  pickled object arrays: real `[k, P_err, P_true, P_model, r, …, nbar]`,
  RSD `[k, mu_centers, P_err(k,μ), P_true, P_model, r, …, nbar]`.
- **N_mesh = 256 on the base box puts k_Nyq = 0.402 exactly at
  k_max = 0.4.** The apparent n̄P_err > 1 in the highest-μ bin there is a
  mass-assignment/Nyquist artifact, confirmed by the small box
  (k_Nyq = 1.61, same HOD row): at k ∈ [0.3, 0.4] the small box gives
  real/μ1/μ2/μ3 = 0.70/0.71/0.79/0.91 vs base 0.79/0.80/0.90/1.07 — all
  bins inflated on base, the LOS bin most. Base ℓ_F drifts with k_max
  (4.4→6.1); small is stable at ≈3.7 Mpc/h. Use N_mesh ≥ 512 (ideally
  1024, cell-matched to small 256) on the base box for k_max = 0.4, or
  cut base to k_max ≈ 0.2. R is unconstrained at these resolutions.

## Running lrgs.py (field-level bias) over all HODs

- `lrgs.py` (the user's transfer-function script, lives with their Hi-Fi_mocks
  setup) calls `CurrentMPIComm.get()`, so `srun -n N python lrgs.py` makes
  nbodykit treat the N ranks as ONE distributed computation. **Never launch it
  under srun/mpirun for throughput.** Instead run N *separate* single-rank
  python processes — each gets its own singleton MPI_COMM_WORLD. The work is
  embarrassingly parallel across catalogs.
- Corollary: a single catalog cannot span nodes, but the ENSEMBLE can —
  `slurm/perlmutter_lrgs_array.sbatch` runs one pool per node via a job array.
- `run_lrgs_parallel.sh` is the pool runner (xargs -P). Set OMP/MKL/OPENBLAS
  threads to 1 — N processes already fill the node. Default workers =
  min(physical cores, mem/8GB) ≈ 64 on a 512 GB CPU node; only 4 outside a
  SLURM allocation, since login nodes are shared.
- lrgs.py skips entries whose output already exists, so the whole thing is
  resumable and safe to re-run; `--status`, `--dry-run`, `--retry-failed`.
- `lrgs_hoist.diff` moves 9 of 25 per-catalog `FFTPower` calls out of the two
  loops (they depend only on the shifted fields, not the catalog): real-space
  pd1/pd2ort/pdG2ort/pd3ort, RSD pz/pd1ort/pd2ort/pG2ort/pd3ort, plus the
  constant `dz - 3/7 f dG2par`. ~36% fewer FFTs, numerically identical. Also
  clamps the batch range to the catalogs that exist (a batch_size not dividing
  10500 otherwise crashes the last batch) and fixes `start` being reused for
  both the timer and the batch offset, which made every "elapsed time" print
  meaningless.
- Bash gotcha hit while writing the runner: `local a=$1 b=...${a}...` expands
  all arguments before assigning any, so `${a}` is empty. Use separate `local`
  statements — otherwise every worker shares one log file.

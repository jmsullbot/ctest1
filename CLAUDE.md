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

## Working style

- The user runs everything on Perlmutter and often hand-edits files there.
  When changing scripts, offer the inline diff, not only a zip.
- Deliver the repo as a zip (`git archive`) — the user does not use
  GitHub keys on Perlmutter.
- Development happens on branch `old_LRG`; push there.

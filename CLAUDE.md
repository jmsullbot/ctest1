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
  - **Since 2026-11-23, deliver changed files individually instead of
    whole-repo zips.** The user has added their own files to repo
    directories on Perlmutter, and plans to fix git later.
  - Send each new file with its repo path, plus one patch per modified file
    (`git diff <their version> HEAD -- <file>`, applied with `patch -p1`
    from the repo root). Check that each patch applies to their version.
  - Optionally add a tar holding only the changed files at their paths.
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

- **Correction (2026-11-23): the ICs ARE on the public CFS disk, in a
  separate IC tree.**
  - The user found it with `ls <root> | grep -i ic`; there is no per-box
    `<sim>/ic/`, which is all the earlier note checked.
  - Layout: `<root>/ic/<sim>/ic_dens_N576.asdf`, e.g.
    `/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/ic/AbacusSummit_base_c000_ph000/`.
    The hugebase boxes are there too.
  - Name patterns must include the leading underscore or slash
    (`*/AbacusSummit_base_c000_ph*`): `*base_c000_ph0*` also matches
    `hugebase_c000`, which bit us once.
  - So Globus is NOT needed for the multi-phase run. Symlink each phase's
    file into `$PSCRATCH/AbacusSummit/<sim>/ic/ic_dens_N576.asdf`, which is
    what run_phases_perr.sh and check_phases.sh expect.
  - The HPSS copy (`/nersc/projects/desi/cosmosim/Abacus/`) still returns
    `HPSS_EACCES` for this user.
- Earlier, wrong belief: "the public disk copy has **no `ic/` directory** for
  `base_c000_ph000`". The ICs were then obtained via the Globus web
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
- 2026-11-15: the user wants to download the raw base ICs to recompute the
  shifted fields with a different code.
  - Which N my_generate_shifted_fields.py reads is still unknown (grep it
    for `ic_dens`). N576 is the smallest file that supports a 512 grid
    (k_Nyq 0.905 > 0.804); truncate in k-space.
  - Reading needs asdf + abacusutils (Blosc). The header GrowthTable gives
    D(0.5)/D(z_init=99) = 59.911485.

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
- **Confirmed with base Nmesh=512 (2026-11-13, hod 2155).** Base-512 matches
  the small box. Mean n̄P_err for k ∈ [0.3, 0.4), real/μ1/μ2/μ3:
  - base-256: 0.79/0.80/0.90/1.07
  - **base-512: 0.72/0.73/0.80/0.91**
  - small-256: 0.70/0.71/0.79/0.91
- Fits at k_max 0.4: base-512 D 0.288, ℓ_F 3.65, R → 0; small D 0.313,
  ℓ_F 3.73, R 0.68 ± 0.33. Base-512 ℓ_F is stable vs k_max (3.56–3.68 for
  k_max 0.2–0.7); base-256 drifted 4.4 → 6.1 (artifact).
- Base-512 agrees with small up to k ≈ 0.6. At 0.6–0.8 its real space
  rises (0.763 vs 0.735, 0.805 vs 0.757): the 512 grid's own Nyquist
  effect. So use k_max ≲ 0.5–0.6 at 512.
- Attribution of the old base-256 excess:
  - base-256 is already 0.02–0.04 above base-512 at k 0.05–0.2, where the
    256 grid is harmless. That part comes from the model fields (the weak
    base-256 d3 is the prime suspect; coarser d2/dG2 may add).
  - The rest grows toward k_Nyq(256), strongest along the LOS (up to
    +0.15): that is the grid.
- Unresolved: base-512 sits ~0.015–0.02 above small in real space at all
  k (1 − D: 0.712 vs 0.687). It could be box-to-box halo differences or
  the remaining resolution mismatch (base 512 is not cell-matched; that
  would be 1024).
- `compare_perr.py` now takes any number of panels (`--run REAL RSD LABEL
  BOX NMESH`), draws a per-grid Nyquist marker, and prints k-window means
  (`--kedges`) and a k_max refit scan (`--kscan`). `--left/--right` still
  work, with `--left-nmesh/--right-nmesh`.
  The plot shows every run's FULL k range on a shared x-axis (fit solid,
  extrapolation dashed, k_max marked); `--kplot` cuts it. Don't crop
  panels at k_max: the user wants the full output.

## perr_from_ic.py: IC grid + catalog -> transfer functions, P_err, plots (2026-11-19)

- Standalone numpy/scipy (no nbodykit; vermin: Python 3.2+), written because
  the full lrgs.py was no longer available (the transcript was compacted).
  - Inputs: `--ic` N³ .npy at `--z-ic` (the grid IS the mesh; `--nmesh`
    truncates in Fourier space first), `--catalog`, `--z-out` (required).
  - **Defaults = the user's new, NON-Abacus sim** (header given 2026-11-19):
    `--box 338.3` Mpc/h (= 500 Mpc at h 0.6766), `--z-ic 49`, Nmesh 256,
    Omega0 0.30964144, OmegaLambda 0.6902672, HubbleParam 0.6766,
    OmegaBaryon 0.04897468 (h and Ω_b don't enter).
    - Ω_m + Ω_Λ = 1 − 9.136e-5, exactly photons + massless ν, so
      `--omega-r` defaults to 9.136e-5 (flat, Ω_Λ = 1 − Ω_m − Ω_r).
    - Which background the sim's code used is UNKNOWN. `--omega-r 0` (Gadget,
      N-GenIC: no radiation in H(a)) makes D(z_out)/D(49) 0.94 % higher; f
      barely moves (2e-4).
  - D(z_out)/D(z_ic) and f(z_out) come from the flat growth ODE: RK4 in
    ln a, starting on the growing mode; massive ν stay smooth via
    `--omega-nu`. `--growth`/`--f` override them.
    - Default cosmology at z_out 0.5: D = 29.945553, f = 0.756735.
    - Abacus c000 check (`--z-ic 99 --omega-m 0.315192 --omega-nu 0.0014197
      --omega-r 0`) gives 59.911626 / 0.759070 vs Abacus 59.911485 /
      0.759070.
  - Catalog handling:
    - A centred catalog is auto-detected (any coordinate < −0.01 L);
      `--no-centre-shift` forces [0, L).
    - Without z_rsd, vz (km/s) → z + vz(1+z)/(100 E(z)).
    - Plain arrays are (N, 3|4|6).
    - The extent is checked against --box: error if outside, warning if it
      spans < 75 % of the box.
  - It builds the shifted operators (Zel'dovich, CIC, compensation
    "shotnoise" = nbodykit's non-interlaced default) and the library cubic
    d3 = (δ1, |k| ≤ 0.5)³.
  - It orthogonalizes per k / (k, μ) bin and fits β per bin.
  - Outputs are in the lrgs.py results layout: `<out>_real.npy`,
    `<out>_rsd.npy` (Nmu = 3 default), `<out>_transfer.png` (Fig. 4 style
    of the user's arXiv:2505.03626: β1, β2, βG2, β3, P_err; real space
    black, μ shades, P_err in h⁻³Mpc³ with 1/n̄) and `<out>_perr.png`
    (P_err blow-up: one panel per curve, k_Nyq marks, n̄P_err axis).
    `--replot PREFIX` redraws; `--save-fields DIR` writes the generator's
    field layout.
- Model = Eq. 16 of 2505.03626 (the user is first author: James M.
  Sullivan): δ_g = δ_Z − 3/7 f G2par + β1 d1 + β2 d2⊥ + βG2 G2⊥ + β3 d3⊥.
  - Real space is its f=0 limit, so δ_Z is fixed in real space too and
    β1 → b1 − 1 in both spaces (black matches μ bins at low k, as in the
    paper's Fig. 4).
  - `--real-no-dz` gives the lrgs.py-like real-space fit (its real field
    files have no dz).
  - G2par is the library's Lagrangian-projected version (lrgs.py's
    `dz − 3/7 f dG2par`), not μ²·(shifted G2).
- Validated in the sandbox: CIC = naive loop; truncation keeps shared
  modes; Poisson samples of a truth in the model span give n̄P_err ≈
  1.01–1.07 in real and redshift space; P_true = P_model + P_err (real)
  to 2e-8. A 256³ run takes 41 s on 4 cores, 2.7 GiB. 512³ is ~8× (use a
  compute node). NOT yet compared with lrgs.py on real data: run both on
  hod 2155 at 256 and compare with compare_perr.py.
  - Growth/f/vz/centring rework: explicit `--growth 1 --f …` is
    bit-identical to before. The computed-D, [0, L) and vz paths agree to
    float32 rounding (≤ 3e-5 of each array's scale).

## LAE/LBG runs on the new 338.3 Mpc/h sim (2026-11-20)

- run1, run1b = LAE; run1i, run1ii = LBG (two stellar-mass cuts).
  - n̄ = 1.3e-3 (three runs) and 6.7e-4: the targets of arXiv:2505.03626,
    whose samples are at z = 3 in MTNG740 and Astrid (ODIN/S5 LAE, S5/CARS
    LBG).
  - perr_from_ic.py at 256³ (k_Nyq 2.38), f(3) = 0.981, D(3)/D(49) = 12.31.
- Real space looks right.
  - b1 = β1 + 1 ≈ 2.2 (LAE) and 3.9 / 4.25 (LBG); the paper has 1.9–2.7
    and 3.9–4.2.
  - n̄P_err ≈ 1.00–1.06 (LAE) and 0.68–0.72 (LBG); the paper's α0 is
    +0.02 to +0.07 (LAE) and −0.16 to −0.32 (LBG).
- **The "redshift-space" catalogs carry no RSD.**
  - The galaxies' own P(k,μ)/P_real is 1.00 ± 0.03 in every μ bin for
    k 0.1–2. Linear Kaiser at f = 0.98 predicts 1.35–1.74 in the top bin,
    and FoG would pull it below 1 at high k.
  - f_eff ≈ 0.0 ± 0.1 for all four runs, both from that ratio
    (0.05 ≤ k < 0.2) and from the β1 slope vs ⟨μ²⟩ at k < 0.12 (−0.81 to
    −1.01, i.e. ≈ −f).
  - That mismatch (the model RSD-displaces by (1+f)Ψ_z; the galaxies were
    not displaced) explains β1 falling with μ at low k and the large top-μ
    P_err (LBG n̄P_err up to 1.8).
  - Below k ~ 0.1 single (k, μ) bins have few modes and scatter by tens of
    percent. All four samples share one realization, so read the ratio at
    k ≳ 0.1.
- Zel'dovich mocks in the same setup (scratch, not in the repo) validate the
  diagnostic.
  - A catalog with RSD lands on the Kaiser lines with flat β1 (f_eff
    0.9–1.3).
  - A catalog with z_rsd = z gives ratio ≈ 1 and slope ≈ −f (f_eff
    −0.04 to 0.32). Its top-μ P_err excess grows with bias but is only
    5–10 %, smaller than the data's.
- Open questions:
  - Why is z_rsd ≈ z? Check rms(z_rsd − z), expected ~1.5–2.5 Mpc/h at
    z = 3 (linear σ_v ≈ 180 km/s × 0.0088 Mpc/h per km/s), and the
    perr_from_ic.py log line about vz.
  - Confirm with `--f 0`: the RSD results should then reproduce real space.
  - **RESOLVED 2026-11-23: our bug, not the user's conversion factor.**
    - The user's catalogs are (N, 6) arrays `x y z xrsd yrsd zrsd`, with RSD
      along each axis; use `zrsd` for LOS z.
    - perr_from_ic.py read every 6-column array as `x y z vx vy vz`, so it
      built z_rsd = z + 0.00883 × zrsd: a 0.9 % stretch along z with
      mean shift L/2 × 0.00883 = 1.5 Mpc/h.
    - That gives no Kaiser boost, a large top-μ P_err, and r_cc → 0 near
      k_z ≈ π/(0.00883 L) ≈ 1 h/Mpc.
    - Fix: the 6-column layout is detected from the data (median |col 4–6
      − col 1–3| below min(0.05 L, 20) → RSD positions; above L/8 →
      velocities; otherwise error), with `--cols6 rsd|vel` to override.
      Structured arrays may name the field `zrsd`.
    - New guard: z_rsd − z is logged (rms, mean, correlation with z). A
      WARNING fires if rms < 0.1 Mpc/h, |mean| > 0.2 rms or |corr| > 0.9,
      and is repeated at the end of the run.
      - The correlation threshold was 0.2 at first. It was raised because
        genuine shifts correlate with position through box-scale modes:
        −0.11 / +0.17 in the 338 Mpc/h mock, up to ±0.78 for a single
        fundamental mode.
      - A shift built from the position itself (the bug) gives ≈ 0.98.
    - Checked on the LBG-like mock:
      - The 6-column RSD, 6-column velocity and structured `zrsd` layouts
        give results bit-identical to the structured `z_rsd` reference.
      - Forcing the old reading reproduces the user's symptoms: shift mean
        +1.50, rms 0.86, corr 0.98; top-μ n̄P_err 1.18–1.26 at k < 0.3;
        top-μ r_cc 0.01 at k 0.7–1. The guard fires.
    - The user must rerun the four LAE/LBG samples. Their real-space
      results are unaffected.
- `plot_fig4.py --run PREFIX LABEL ... --z-out 3 --out X` draws the paper's
  Fig. 4 (one row per run) and Fig. 5 (r_cc), plus `X_rsdcheck.png` with
  both diagnostics. It imports perr_from_ic.py from its own directory.
- `perr_from_ic.py --los z|x|y|several|all` (2026-11-23):
  - For x or y, the IC grid is transposed cyclically (x → (1,2,0),
    y → (2,0,1)) and the catalog columns are permuted, so the LOS is the
    last axis. The operators are rebuilt there.
  - The LOS z path reuses the real-space operators and is bit-identical to
    before. Real space is computed once in the original frame, so it is
    bit-identical for any --los.
  - With several LOS, each goes to `<out>_rsd_los<a>.npy`, and
    `<out>_rsd.npy` is their mean (r = mean cross / √(mean P_true · mean
    P_model)).
  - A default output name gets `_los<axes>` unless the LOS is z.
  - `--save-fields` writes `..._axisrsd_<a>_...` in the original
    orientation.
  - Tests on the mock:
    - LOS x / y on the original inputs equal LOS z on pre-rotated inputs
      to 2e-6.
    - Saved fields land in the right orientation (5e-4 of std, against
      ~400× for the wrong one).
    - In the 3-LOS mean, P_err equals the average of the three.
    - Averaging two LOS gives f_eff 0.99–1.00 (true 0.98), against
      0.91 / 1.10 for one LOS.
  - Caveat: the same physical problem computed in two orientations differs
    by ~1e-3 at k ≳ 1.5. The odd-derivative operators (Ψ, off-diagonal G2)
    are ambiguous on the Nyquist planes, and rfft treats the last axis
    differently. This is harmless, but it means rotated runs are not
    bit-identical.
- `run_perr_samples.sh`: paste IC, ZOUT, LOS and "label catalog" lines at
  the top.
  - Run it with `bash` (login node) or `sbatch` (cpu, debug, 30 min); FFT
    threads are 8 on a login node and 32 in a job.
  - It runs perr_from_ic.py on each catalog (logs to `$OUTDIR/<label>.log`),
    then plot_fig4.py on all of them.
  - At the end it greps the layout and RSD-shift lines and any WARNING.
  - It needs perr_from_ic.py and plot_fig4.py in the submit directory, or
    `CODE=`.
  - Tested in the sandbox on two mocks with `LOS="x z"`.
  - It activates `CONDA_ENV` itself (default nbodykit_env_25; a setting at
    the top), like the repo's other job scripts. The first version forgot
    this, and the user had to point it out.
    - CONDA_BASE comes from $CONDA_EXE, `conda info --base` or
      `slurm/env.local.sh`. That file is read for CONDA_BASE only, so its
      `CONDA_ENV=abacus` cannot override the setting.
    - `ENV_READY=1` skips activation.
    - Tested with a stand-in conda (a conda.sh function) on a bare PATH, and
      in sbatch-like conditions with no CONDA_EXE.
  - 2026-11-23: the user reported "your script didn't work" without the
    error and ran the samples by hand on an interactive node.
    - A likely cause: inside salloc, SLURM_SUBMIT_DIR is set (to where
      salloc started), and CODE used it first.
    - CODE now resolves in this order: the script's own directory (bash,
      also inside salloc), then SLURM_SUBMIT_DIR (sbatch runs a spooled
      copy), then $PWD. All three cases were tested.
    - The real error is still unknown; ask for it if it recurs.
- All figures (perr_from_ic.py `_transfer`/`_perr`, plot_fig4.py
  fig4/fig5/rsdcheck) are also written as vector PDF next to the PNG, via
  `perr_from_ic.save_figure` (user request, 2026-11-23).
- **Rerun after the column fix (2026-11-23):** the user's files are
  LAE_ODIN, LAE_S5 (= run1, run1b), LBG_CARS (6.7e-4, = run1ii) and LBG_S5
  (= run1i). Their real-space files are identical to the earlier runs.
  - RSD is now present. Top-μ n̄P_err at k 0.3–0.5:
    - LAE: 1.08–1.09 (was 1.27);
    - CARS LBG: 0.85 (was 1.47);
    - S5 LBG: 0.88 (was 1.78).
  - The LBG redshift-space r_cc tracks real space to k ~ 0.5; β1 is
    μ-flat at low k for the LBGs; the top-μ P_err rises past 1/n̄ only at
    k ≳ 0.5. Qualitatively this is like the paper's MTNG Fig. 4.
  - **But the RSD amplitude is ~half of linear theory.**
    - The top-μ P(k,μ)/P_real plateaus, flat over k 0.1–0.7, at 1.3–1.4
      for the LAEs (Kaiser 1.73) and 1.2–1.25 for the LBGs (Kaiser
      1.35–1.39). FoG would make it fall with k instead.
    - f_eff = 0.48–0.50 from the ratio for all four samples, and
      0.45–0.77 from the β1 slope (−0.21 to −0.53).
    - The same f_eff in LAEs and LBGs points to a common cause, i.e. the
      conversion, not LAE physics.
    - A factor of exactly 2 at z = 3 is √a (or 1/√(1+z)). Example: Gadget /
      AREPO snapshot velocities are u = v_pec/√a, and using v = u·a
      instead of u·√a gives exactly 0.5; so does √(1+z) in place of (1+z).
    - Check: the logs' "RSD shift z_rsd - z: rms" should be ~1.5–2.5
      Mpc/h; ~0.8–1.2 would confirm a factor 2.
    - The LAE β1 still drops with μ at low k (top bin ≈ 0.75 vs 1.0): the
      same f mismatch (slope ≈ f_eff − f).

## Bias parameters from the transfer functions: fit_bias.py (2026-11-23)

- The request: even-in-k polynomial fits to the low-k limits, as in
  2505.03626, but weighted with the transfer-function covariance of
  1811.10640, compared with Table III of 2505.03626.
- 2505.03626 defers the method to 2412.01888, which defers to 2409.10609
  §4.7.
  - Real-space templates: β_X = a0 + a1 k² + a2 k⁴ (their eq. 58), k_max 0.4.
  - Low-k limits with δ_Z kept (their eq. 45; our default real space is the
    same): b1 = 1 + β1, b2 = 2β2, b_G2 = β_G2 − 2/7 b1 + 1/2, b3 = 6β3.
  - Without δ_Z (their eq. 20; `--real-no-dz`): b1 = β1, b_G2 = β_G2 − 2/7 b1.
  - Verified by matching second-order kernels:
    - δ̃1 → 1 + shift;
    - δ_Z → 1 + shift + ½F_G2;
    - Eulerian F2 = 1 + shift + 2/7 F_G2 (G2 basis; 17/21 is the K² basis).
  - The papers fit β1 with the one-loop shape (their eq. 59), not a
    polynomial. So b_Γ3, Table III's b_∇²δ, c_μ² and b4 are NOT obtainable
    from polynomials.
- 1811.10640 covariance (its eqs. 86–88):
  - Noise: C_noise = P_err/(N_k P_11), diagonal. Generalized here to
    P_err/(N_k P_O⊥) per operator, with N_k = all grid modes (both k and
    −k; `Mesh.wsum1/2`).
  - Theory error: δβ(k_i)δβ(k_j) exp(−Δk²/2/0.2²), with δβ = b1(D/D0)⁴
    (k/0.45)^3.3 for β1 (two-loop) and b1(D/D0)²(k/0.3)^1.8 for the others
    (one-loop). Negligible at z = 3.
  - Validated on 16 Zel'dovich mock realizations (128³, 338.3 Mpc/h): the
    scatter / prediction is 0.87–1.15 for every β, in real space and per μ
    bin. Pulls of the fitted a0 have std 0.82–1.15.
- perr_from_ic.py now writes `<out>_real_ops.npy` [k, N_k, P_O1..4] and
  `<out>_rsd_ops.npy` [k, μ, N(k,μ), P_O1..4] (per LOS, plus the mean).
  - These depend only on the IC, not on the catalog, so one rerun serves
    every sample (`fit_bias.py --ops PREFIX`).
  - The user's LAE/LBG files predate them. The first fit used stand-in ops:
    two 256³ ICs with an Eisenstein–Hu no-wiggle P(k) and σ8 0.81. The
    user's δ̃1 power is 9–19 % above the stand-in's, so that only changes
    the errors and weights slightly.
- Results, real space, k ≤ 0.4, a0 + a2k² + a4k⁴, stand-in ops. The table
  shows ours vs Table III MTNG / Astrid:

  | sample | b1 | b2 | b_G2 | b3 | α0 |
  |---|---|---|---|---|---|
  | ODIN LAE | 2.16 ± 0.03 (2.33 / 2.15) | −0.32 ± 0.19 (0.17 / 1.05) | 0.39 ± 0.27 (−0.80 / −0.98) | −6.5 ± 1.1 (−1.2 / 6.7) | 0.026 (0.046 / 0.074) |
  | S5 LAE | 2.13 (2.68 / 1.89) | −0.53 (0.61 / −0.05) | 0.42 (−0.84 / −0.52) | −6.2 (−2.7 / −1.05) | 0.029 (0.041 / 0.023) |
  | CARS LBG | 4.23 (4.24 / 3.96) | 8.33 ± 0.24 (6.21 / 6.02) | −0.66 ± 0.31 (−3.03 / −1.44) | 14.6 ± 1.3 (13.1 / 9.2) | −0.327 (−0.16 / −0.32) |
  | S5 LBG | 3.87 (3.92 / 4.05) | 6.33 (4.85 / 7.76) | −0.49 (−2.28 / −1.19) | 7.9 (8.75 / 12.5) | −0.344 (−0.17 / −0.307) |

  - Across k_max 0.2–0.5 and orders 2 and 4: b1 moves ±0.07 (more than its
    formal ±0.03), b2 and b3 move less than their errors for the LBGs, and
    b_G2 moves ±0.3–0.4.
  - β1's χ²/dof is ~1.8 (LAE) and ~3 (LBG) for every choice, because the
    polynomial cannot follow β1; this is why the papers fit the one-loop
    shape.
  - Our b_G2 is less negative than Table III everywhere. For the LBGs it is
    near the local-Lagrangian value −2/7(b1−1) ≈ −0.85. α0 matches Astrid
    for the LBGs.
  - α2 and the redshift-space a0's are affected by the factor-2 RSD
    amplitude, so do not trust them yet.

## Multi-phase AbacusBase run: c000 ph000–ph024, HOD row 1003 (2026-11-23)

- The request: reproduce the ph000 pipeline on all 25 c000 phase boxes, for
  table row 1003 only (not all 10500).
- Workflow (README "All 25 base phases for one HOD row"):
  1. `slurm/check_phases.sh`: status table (halos, cleaning, subs, IC, cat,
     perr). Paths come from the config; it also follows progress.
  2. `PHASES=0-24 bash slurm/globus_fetch_ic.sh`: one Globus batch for all
     boxes (`SIMS=` / `PHASES=` / `SIM=`). Missing source files stop it
     unless `ALLOW_MISSING=1`, and `--sync-level size` makes reruns cheap.
     Tested with a stub `globus` CLI.
  3. `sbatch slurm/perlmutter_phases_hod_array.sbatch` (array 0–24, 3 h, HOD
     env via `_common.sh`).
     - `generate_lrg_hods.py --sim_name SIM` writes `config_<sim>.yaml` next
       to `--output`; prepare_sim only accepts a config path.
     - prepare_sim is skipped when `<subsample_dir>/<sim>/z0.500/` is
       non-empty (`FORCE_PREPARE=1` redoes it).
     - `ROWS="1003 ..."` adds rows.
     - `CLEAN_SUBSAMPLES=1` deletes only the subsamples that task made (marker
       `.prepared_by_phases_job`), never pre-existing ones such as ph000's.
     - Outputs: `$PSCRATCH/lrg_hods_phases/phNNN/{catalogs/001003.npy,
       lrg_hods_rows001003-001004.hdf5}`.
  4. `sbatch run_phases_perr.sh` runs perr_from_ic.py per phase, then
     phase_ensemble.py.
     - Defaults: NMESH 512 (k-space truncation of N576), LOS all, 8 phases at
       once × 32 FFT threads. Resumable: finished phases are skipped.
     - `COSMO_ARGS` default `--omega-m 0.315192 --omega-nu 0.0014197
       --omega-r 0`, which reproduces Abacus c000's own growth. Set
       `COSMO_ARGS=""` for perr_from_ic's defaults.
     - Env: CONDA_ENV default abacus (needs asdf and abacusutils for the
       Blosc ICs).
- perr_from_ic.py reads `.asdf` ICs directly.
  - The header's BoxSize and InitialRedshift become the --box/--z-ic
    defaults (now None), and a contradicting flag is an error.
  - The header's GrowthTable at z_out is printed, with a WARNING if the
    computed D is > 0.1 % off.
  - `.npy` behaviour is unchanged. Results from a `.asdf` IC are
    bit-identical to the same grid as `.npy`.
- phase_ensemble.py: ensemble mean in the results layout
  (`_mean_real/_mean_rsd.npy`), ±1σ phase-scatter figure, per-phase
  fit_bias fits with "ratio = scatter / mean quoted error", P_err window
  table.
  - `--lrgs REAL RSD` compares one phase with lrgs.py. It uses our
    `_rsd_losz.npy` when it exists, and shifts our real-space β1 / β_G2 by
    +1 / +½ unless `--real-no-dz`.
  - On 16 mock realizations the ratios are 0.75–1.02.
- **Reversed by the user (2026-11-25): for the Abacus phases the field-level
  step is THEIR pipeline, my_generate_shifted_fields.py then lrgs.py, exactly
  as for ph000.** perr_from_ic.py and its runners belong to the TNG-like
  338.3 Mpc/h sim.
  - This session had mixed the two workflows, and the user had to point it
    out. Don't propose perr_from_ic.py, run_phases_perr.sh or
    phase_ensemble.py for the Abacus phases unless asked.
  - The HOD-array catalogs (`$PSCRATCH/lrg_hods_phases/phNNN/catalogs/`) and
    the IC links (`$PSCRATCH/AbacusSummit/<sim>/ic/`) serve that pipeline
    unchanged.
  - Cost per phase at Nmesh 512: the generator took 169 min and 306 GiB on a
    CPU node (job 60856968). So 24 phases cost ≈ 70 node-hours and
    24 × 5.4 GB of fields (real 2.15 + RSD axis 2 3.22 GB). lrgs.py on one
    catalog takes minutes (72 s real at 512; RSD unmeasured).
  - Both scripts hardcoded the IC path, output folder, `pathD` and the
    `ph=000` tag. The user's current copies arrived on 2026-11-25. They import
    `lib.tng_lib` from an absolute Hi-Fi_mocks path, so they run from any
    directory.
    - The generator reads `$PSCRATCH/abacus_ics_000/ic_dens_N576.asdf`, uses a
      hardcoded c000 prefactor 59.911485 and f 0.759070, and loops RSD over
      `range(3)`. So job 60856968's 169 min covered real space plus all three
      axes.
    - lrgs.py takes `--batch_size/--batch_num` and reads RSD axis 2 only. The
      uploaded copy does NOT have lrgs_hoist.diff applied.
  - `patches/my_generate_shifted_fields_phases.diff` adds `--ic`, `--out` and
    `--axes`. `--ic` without `--out` refuses to run, so it can never overwrite
    the ph000 fields.
  - `patches/lrgs_phases.diff` adds `--ph`, `--path` and `--pathD` (all three
    or none), and puts `ph=%03d` in both result names.
  - Without the new options both patched scripts write byte-identical files to
    the originals. This was checked with stand-in nbodykit, tng_lib and asdf
    modules on 8³ grids (scratch harness, not in the repo).
  - `slurm/perlmutter_phases_lrgs_array.sbatch` (array 1–24, 3 h, whole CPU
    node, nbodykit_env_25), per phase:
    - the generator with `--axes 2` into `phNNN/shifted_fields/`, skipped
      when the files exist at full size; it logs [mem] lines;
    - then lrgs.py per row into `phNNN/results/`.
    - It refuses an unpatched generator, an Nmesh mismatch, AXES without 2
      and missing IC/catalog. ph000 exits at once (ALLOW_PH000=1 overrides).
    - `DELETE_FIELDS=1` and `FORCE_FIELDS=1` exist.
    - Every path was tested on a fake tree with the stand-ins.
  - Estimated cost with axis 2 only: ~1–1.5 h per phase, ~35 node-hours in
    all. Not measured yet.
  - check_phases.sh now shows `fields` and `lrgs` columns (ph000 via
    `PH000_DIR` = Heavy_files_Andrej/AbacusBase) instead of `perr`.
  - `phase_perr.py DIRS|FILES --row 1003 --nmesh 512` (2026-11-27) plots the
    lrgs.py results over the phases with fit_perr.py's model. It writes
    `_mean` (ph000 alone | mean ± scatter), `_bins`, `_phases` (grid) and
    `_fits` (per-phase D, ℓ_F, R and window means) as PNG + PDF, plus
    `_summary.txt`.
    - It finds files recursively by name, accepting names with or without
      `=` and with an upload prefix. It pairs real and RSD per phase and
      skips other rows, other Nmesh and unpaired files.
    - Per-phase fits use fit_perr.fit (σ ∝ 1/k). The fit to the mean uses
      σ/√N per bin with absolute_sigma, so its χ² is meaningful (real and RSD
      bins treated as independent).
    - Tested on 25 synthetic phases in the exact layout. χ²/dof ≈ 0.9 at
      k_max ≤ 0.3 shows the errors are right.
  - **Results, row 1003, 25 phases at Nmesh 512 (2026-11-27)**, uploaded as
    `lrgs_hod1003_N512_25phases.tar.gz`. The pickles only reference
    numpy.ndarray/dtype/_reconstruct (checked before loading).
    - **Single-bin spikes at k = 34, 68, 102, 136 k_f (k_f = 2π/2000)**, in
      real space and the μ ∈ [0, 1/3] bin only. Excess n̄P_err = +1.02, +0.19,
      +0.05, +0.014 in real space (μ1: +3.0, +0.57, +0.15, +0.04).
      - The galaxy auto power has the same excess; the model power has none.
      - It is identical in all 25 phases (n̄P_gal at 34 k_f = 2.33–2.46, against
        ~1.3 around it).
      - So it is a pattern in the catalogs with period L/34 = 58.8 Mpc/h along
        an axis ⊥ LOS. L/34 matches the 34 superslab files of a base box.
      - It equals A1 ≈ 0.19 along one axis (0.13 each if along two). The 34 k_f
        shell holds 14754 modes, and N_gal = 870k.
    - Row 1003 has **Bcent = 0.988 (99th percentile of the table)** and σ =
      0.017 (near-step centrals). The suspect was the environment AbacusHOD's
      assembly bias uses, near superslab boundaries.
    - **CONFIRMED in the abacusutils source (2026-11-28, v2.1.2 sdist from
      PyPI; the user's installed version is unknown):**
      - `prepare_sim.prepare_slab(i)` loads ONE `halo_info_NNN.asdf` and sets
        `allpos = halos['x_L2com']`. `menv.do_Menv_from_tree` builds the
        KDTree from those halos only, with `boxsize=Lbox`. Its own comment
        says "periodicity exists only in y and z directions".
      - So a halo within `rad_outer` (default 10 Mpc/h in 2.1.2) of a
        superslab x-boundary has no neighbours from the next file, and its
        Menv is too low. `calc_fenv_opt` then ranks Menv per mass bin WITHIN
        the slab.
      - GRAND_HOD.py, LRG centrals: `logM_cut + Ac*deltac + Bc*fenv`, with
        fenv in [-0.5, 0.5]. Satellites use Bs in M1 the same way.
      - Hence a pattern of period L/34 along x with amplitude ∝ Bcent, in
        EVERY base-box catalog made with want_AB and B ≠ 0. It was there from
        the first base run on.
    - **Not caused by any change in this work.** The user asked whether the
      agent had broken something (2026-11-28).
      - ph000's result file in the tar is dated 2026-11-09 (original
        unpatched lrgs.py, original catalog) and has the same spike: n̄P_err
        1.766 at 34 k_f against 0.73 beside it.
      - The Nov 13 hod-2155 compare_perr plot (recovered from the session
        transcript) already shows it at both Nmesh 256 and 512: μ1 ≈ 1.01 and
        real ≈ 0.81 at k = 0.107, against ≈ 0.71. The small-box panel has no
        such point. Nobody flagged that single dot, the agent included.
      - Row 2155 has Bcent = −0.369. Its spike power is ~0.10 against 1.02;
        corrected for N_gal that is 0.148 in A1², and (0.369/0.988)² = 0.139.
        So the power goes as Bcent².
      - What is new is only the detector in phase_perr.py, a row with ~10×
        the spike power, and 25 phases (252σ instead of one dot).
      - The detector flags nothing on spike-free synthetic data. The masked
        per-phase fit equals fit_perr.fit with those 4 bins removed (D
        0.2783, ℓ_F 4.903 for ph007).
    - **Correction to an earlier claim** ("only those four bins are affected,
      the rest of P_err is clean"): that is true only to first order.
      - The same modulation multiplies the galaxy field, so it moves
        (A1/2)² ≈ 0.9 % of the clustering power by ±34 k_f along x. That is a
        smooth extra n̄P_err of order 0.01 for row 1003 and ~0.002 for row
        2155 (estimate, not measured).
      - More basic: the environment rank is wrong for every halo within
        rad_outer of a boundary. rad_outer is 10 Mpc/h in every 2.x release,
        so that is 20 of every 59 Mpc/h, a third of the volume. Part of the
        "environment" these mocks respond to is artificial, and masking bins
        does not undo that.
      - A real fix means a patched prepare_sim (tree from the slab plus its
        two neighbours, ideally box-wide ranks), then regenerating subsamples,
        catalogs and lrgs.py results. Not started; it is the user's call.
    - **Known upstream and already fixed on abacusutils `main`, but in no
      release** (checked 2026-11-28 in a clone of abacusorg/abacusutils and on
      the PR pages).
      - PR #207 "fix local env B related artificial spikes" by Hanyu Zhang
        (ahnyu): opened 2026-04-08, merged 2026-04-14 as a9b42ac by Lehman
        Garrison. No separate issue was filed.
      - Its description matches ours: Menv was estimated independently within
        each superslab, which gave an x-dependent striping of fenv, and its
        plots show spikes near k = 0.11, 0.22, 0.33.
      - The fix: prepare_sim pads each slab with neighbour-slab halos within
        rad_outer and writes raw Menv to
        `env_xcom_<i>_abacushod_localenv_new.h5`. abacus_hod.py then ranks
        fenv over the WHOLE box and maps it back by halo id. Periodic boxes
        only; light cones are unchanged.
      - PR #208 (5f1acb1, same day) makes it safe for boxes with one or two
        slabs.
      - The latest release is v2.1.2 (2025-10-13), so every pip/conda install
        still has the bug. The fix needs an install from GitHub main.
      - Test for the fix: `hasattr(abacusnbody.hod.prepare_sim,
        'load_env_halos')`.
      - With the fixed version prepare_sim must be rerun: abacus_hod raises
        FileNotFoundError without the env sidecars. Then regenerate catalogs
        and lrgs.py results. The shifted fields depend only on the ICs, so
        they stay valid.
    - **Decision (user, 2026-11-28): install abacusutils from GitHub main and
      rerun ALL catalogs, the 25 phases first, then the 10500 rows.**
      - `slurm/setup_env_envfix.sh` creates conda env `abacus_envfix`: python
        3.11 with corrfunc, numba, scipy, h5py, pyyaml and astropy from
        conda-forge in one solve, then `pip install "abacusutils @
        git+https://github.com/abacusorg/abacusutils@b4e33ed7..."` (main of
        2026-11-02, version 2.1.3.dev49+gb4e33ed72).
        - It reads env.local.sh for CONDA_BASE only, verifies the fix and the
          AbacusHOD import, and prints the old env's version.
        - abacusutils main is pure Python (setuptools_scm, no LFS) and needs
          python ≥ 3.10. The pip install was run for real in the sandbox;
          conda itself was a stand-in.
      - **The user's first run of it failed (2026-11-28, login node)**: numba
        died at import with `module 'coverage.types' has no attribute 'Tracer'.
        Did you mean: 'TTracer'?`.
        - numba uses `coverage` whenever one is importable and needs
          coverage ≥ 7.4.
        - A conda env is a regular, non-venv Python, so it puts the USER site
          (`~/.local/lib/python3.11/site-packages`) BEFORE its own
          site-packages. A stale package there shadows the env, and upgrading
          inside the env cannot help.
        - pip in that run also counted user-site packages as installed, so
          dependencies "already satisfied" from `~/.local` never reached the env.
        - Fix: the script writes `etc/conda/activate.d/zz_envfix_isolate.sh`
          (`PYTHONNOUSERSITE=1`; with `CLEAR_PYTHONPATH=1` it also unsets
          PYTHONPATH) and a deactivate.d counterpart. It reruns pip with the
          user site hidden, upgrades an in-env coverage < 7.4, lists every
          imported package that comes from outside the env (then exits 1), and
          checks the hook on a fresh activation. Re-running it repairs the
          existing env.
        - Where the user's stale coverage lives is not confirmed (the user
          site is assumed; the script prints the path if it is PYTHONPATH).
        - **Sandbox lesson:** a venv puts its own site-packages BEFORE the user
          site, the opposite of conda. One test in a system-site venv passed
          for the wrong reason (the venv had gained its own coverage). Test
          user-site leaks in a regular prefix: `uv python install 3.11` into
          the scratchpad, copy it per env, delete `EXTERNALLY-MANAGED`.
      - Other changes on main since v2.1.2 were reviewed. GRAND_HOD.py is
        cosmetic only. The LRG path of subsample_halos / submask_particles is
        unchanged (BGS got its own branch). The AbacusHOD, run_hod and
        prepare_sim.main signatures still fit the user's generate_lrg_hods.py.
        So the environment fix is the only functional change for LRGs.
      - main logs through `logging` with no handler, so prepare_sim is silent
        in a job log. The HOD job prints its own "[prepare_sim N min] n of 34
        slabs written" heartbeat (PROGRESS_SEC, default 300).
      - New folders: subsamples in `$PSCRATCH/lrg_hods_envfix/subsamples/`,
        phases in `$PSCRATCH/lrg_hods_phases_envfix/phNNN/`. The unfixed run
        stays in `$PSCRATCH/lrg_hods_phases/` and is not touched. These are
        now the DEFAULTS of the three phase scripts.
        - prepare_sim builds its path by string concatenation, so
          subsample_dir must end in "/". The job writes it that way into
          `config_base.yaml`.
      - `perlmutter_phases_hod_array.sbatch` (array now 0–24, ph000 included):
        - It activates `HOD_ENV` (abacus_envfix) after _common.sh. The
          user's env.local.sh has a plain `CONDA_ENV=abacus`, which would
          override an exported CONDA_ENV.
        - It exports `PYTHONNOUSERSITE=1` itself (`KEEP_USER_SITE=1` turns
          that off). Its start-up check imports AbacusHOD in HOD_ENV, i.e. what
          the generation step imports: exit 6 if that fails, before
          prepare_sim runs.
        - It exits 5 if `prepare_sim.load_env_halos` is missing
          (`ALLOW_UNFIXED=1` overrides).
        - prepare_sim is skipped only when halo, particle AND env files exist
          for every halo_info slab. It refuses subsamples without env files
          (made by the unfixed code) unless FORCE_PREPARE=1.
      - `perlmutter_phases_lrgs_array.sbatch` (array 0–24): shifted fields
        made earlier are symlinked from `FIELDS_FROM/phNNN/shifted_fields`
        (default the old root) or `PH000_FIELDS` (AbacusBase). The generator
        runs only if they are missing. DELETE_FIELDS on a link removes the
        link only.
      - check_phases.sh shows subs = ok / old / part, and fields = ok / link.
      - Tested on fake trees with the stand-in conda and the REAL abacusutils
        main (Corrfunc was a stand-in): fresh run, rerun, old env refused,
        unfixed subsamples refused, missing env, CLEAN_SUBSAMPLES, missing
        box, prepare failure, linked / generated fields.
      - The test caught a real bug: under `set -e` + pipefail, `find` on the
        not-yet-existing subsample folder killed the task. `count()` now
        returns 0 for a missing folder.
      - **Still to do after the phases:**
        - the 10500-row rerun with the fixed env (prepare_sim on ph000 into
          the new subsample dir, new OUTDIR `$PSCRATCH/lrg_hods_envfix`,
          another ~8 TB next to the old 8 TB, so space must be freed first),
          then lrgs.py at 512 on all rows;
        - check whether the small box is affected: count its halo_info
          files. With a single file the old code was already periodic in x.
      - Tests: `slab_profile.py CATALOG...` gives the folded density and
        first-harmonic amplitude A1 along x, y and z, for all galaxies,
        centrals and satellites.
      - `spike_scan.py RESULTS_DIR --ph 0 --nmesh 512` gives the 34/68 k_f
        excess for every row, against Acent/Asat/Bcent/Bsat (Spearman).
      - Both were tested on synthetic inputs: planted A1 = 0.20 (centrals, x)
        was recovered, and so was a Bcent² scaling (ρ = 0.996 with |Bcent|).
      - User runs pending.
    - phase_perr.py now finds such bins itself (> 8σ of the mean and > 3 %
      above the median of ±4 neighbours). It leaves them out of every fit,
      mean and window, marks them ▼, and adds a `_spikes` figure (`--mask
      none` keeps them).
      - Without masking, the μ1 window mean at k 0.1–0.2 read 0.836 instead of
        0.741.
    - After masking:
      - the window means rise monotonically with μ;
      - nbar scatter is 0.07 %;
      - per phase D = 0.2783 ± 0.0007 and ℓ_F = 4.90 ± 0.02, with R → 0 in
        every phase;
      - [0.3, 0.4) real/μ1/μ2/μ3 = 0.714/0.755/0.864/0.959 (scatter 0.001);
      - ph000 is typical (+0.8σ real).
    - The fit to the mean has χ²/dof = 31572/492. The model is inadequate at
      25-box precision.
      - Real-space n̄P_err FALLS from 0.799 ± 0.002 (k 0.02–0.05) to 0.709
        (0.2–0.3), then rises to 0.784 (0.6–0.8, the 512 grid).
      - 1 − D·W(kR) can only rise with k, hence R → 0.
      - An EFT-style 1 + α0 + α1 k² fit (α1 < 0 at low k) would follow it.
        Offered, not done.
  - Superseded plan: perr_from_ic.py per phase, validated against lrgs.py's
    ph000 row-1003 results at 512.
- Unknown and to be flagged: the subsample size per base box (`du -sh` of
  ph000's) × 25 against the 20 TB quota, on top of the ~8 TB of 10500-row
  catalogs. ICs are 25 × ~0.8 GB.
- ph000's IC lives at `$PSCRATCH/abacus_ics_000/` (user layout). The runner
  expects `$IC_ROOT/<sim>/ic/ic_dens_N576.asdf`: fetch ph000 again or symlink.
- Not runnable in the sandbox: real AbacusHOD/prepare_sim, real Globus, and
  Blosc-compressed asdf (no abacusutils). The bash flows were tested with
  stubs and fake trees.
- Where things are on Perlmutter (2026-11-24):
  - The base-box repo the jobs run from is
    `/pscratch/sd/j/jsull/qso_sbp/old_lrgs/ctest1`. The small-box copy is
    `.../old_lrgs/new_small_files/ctest1`.
  - The phases zip is unpacked at `/pscratch/sd/j/jsull/qso_sbp/abacus_phases`
    (`repo/`, `patches/`), not the README's `$PSCRATCH/tmp_abacus_phases`.
  - Until 2026-11-24 this repo had no perr_from_ic.py or fit_bias.py, so the
    LAE/LBG runs used copies in some other directory. A perr_from_ic.py fix
    has to reach both copies.
  - The first array submission failed in every task with `unrecognized
    arguments: --sim_name`. Only the zip's three slurm/ files had been
    copied in.
    - Nothing is written before that failure except empty `phNNN/` dirs.
    - A job log's `Repo dir` line shows which checkout the job used.
  - **This repo's generate_lrg_hods.py predates Sep 11 and has unknown hand
    edits** (most likely the August version the first base run used). The
    zip's patch is against Sep 21 and fails on every version before a08bc96.
    - ph000's catalog came from that code, so the file is kept. Only a
      minimal `--sim_name` patch goes in: the argument, `_config_for_sim` and
      the two-line hook in main, copied verbatim from HEAD and diffed against
      c348f5c.
    - That patch applies to c348f5c, 7ad79db and fd29f8b. It also applies
      with edits next to the hunk (fuzz) and with trailing whitespace lost in
      copy-paste.
    - With the job's exact arguments it writes `config_<sim>.yaml` and leaves
      the original config alone. Without `--sim_name` nothing changes.
    - So the phase HDF5 shards lack box_size, n_cent and f_sat, like ph000's.
      Use compute_fsat.py; readers fall back to box 2000.
    - Applied on Perlmutter (none of the user's lines changed). The
      `_config_for_sim` check passed in their abacus env, and the array was
      resubmitted as `--array=1-24`, with ph000's catalog copied from
      `$PSCRATCH/lrg_hods/catalogs/`.
  - The base catalogs are structured `x y z z_rsd vx vy vz mass id`, centred
    on 0. `--los all` uses the stored `z_rsd` for z, as lrgs.py does, and
    `x + vx·(1+z)/(100E)` (0.011343 Mpc/h per km/s at z 0.5) for x and y.

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
- **lrgs.py never saves painted galaxy meshes.** Per catalog it reads
  `pathD + "{:06d}.npy"` and shifts positions by +L/2 (catalogs are
  centred on 0). It uses (x, y, z) in real space and (x, y, z_rsd) in RSD,
  paints `ArrayCatalog(...).to_mesh(compensated=True).paint() - 1.0` in
  memory, and saves only `results`:
  - real: (k, P_err, P_true, P_model, r, β1, β2, βG2, β3, n̄);
  - RSD: the same per μ bin.
  The small copy has `pathD = /pscratch/sd/j/jsull/lrg_hods_small/catalogs/`;
  the base lrgs.py presumably uses `.../lrg_hods/catalogs/` (not seen
  here). One 512³ float32 mesh is 0.5 GiB, so all 10500 × (real + RSD) is
  ~10 TB. Repaint on demand rather than store.
- `pack_hods.sh [rows...]` (default: the user's 27-row list) bundles, per
  row, the base-256, base-512 and small-256 results (real + RSD) plus the
  rows of each `hod_summary.csv` and a README into
  `$PSCRATCH/downloads/hod_results_<tag>.tar.gz`. The base and small
  catalogs go into a separate `hod_catalogs_<tag>.tar` (sizes printed first;
  ~80 B/galaxy). `CATALOGS=0` / `BASE_CATALOGS=0` drop catalogs. Missing
  files are listed in MANIFEST.txt.
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
- **Small box:** `small_lrgs.py` is byte-identical to `lrgs.py` apart from
  `BoxSize=500`, the two paths, and `AbacusBase/ph=000` → `AbacusSmall/ph=3000`
  in the output names — so `lrgs_hoist.diff` applies to it unchanged
  (`small_lrgs_hoist.diff` is the same patch pre-rebased). Run it with
  `slurm/perlmutter_lrgs_small_array.sbatch`, which sets SCRIPT/CATDIR/
  RESULTSDIR and `RESULT_TAG=AbacusSmall` (the runner's `--status` globs
  `results_${RESULT_TAG}_*`, so the wrong tag silently reports 0 done).
- **Batch size must match nodes × workers.** Wall time is set by the busiest
  worker: ceil(batches_on_node / NWORKERS) rounds of BATCH_SIZE catalogs. On
  16 nodes, BATCH_SIZE=50 leaves ~14 batches per node, so only ~14 workers run
  (4.7× slow on base, 9.5× on small, where ~125 workers fit); BATCH_SIZE=10 on
  base gives 66 batches for 62 workers — a second round, 1.9× slow. Good
  values are ≈ NCAT/(nodes × workers), rounded up to a divisor of 10500: 12 for
  base (16×62 or 64×15), 6 for small (16×125). Both sbatch files set these as
  defaults, and the runner warns with a suggestion when the critical path
  exceeds 1.5× a balanced split. An earlier claim that batch size "only buys
  visibility" was wrong — it is the dominant throughput knob on many nodes.
  Keep BATCH_SIZE explicit (never auto-computed per node): all shards must use
  the same value, or the stride partition leaves catalogs unprocessed.
- The small box is NOT 64× cheaper here: same `Nmesh=256` grid, so identical
  FFT work per catalog; only painting and I/O shrink (~60-70% of base-box
  time). It is much lighter on memory though, so `MEM_PER_WORKER_GB=4`
  lets the runner use roughly twice as many workers per node.

## DONE (2026-11-13): base-box rerun of lrgs.py at Nmesh = 512

**Complete.** `NMESH=512 bash run_lrgs_parallel.sh --status` reports real
10500/10500 and RSD 10500/10500. `grep -l FAILED lrgs_*.out` is empty. Run
with the recommended array settings (below); exact job IDs and RSD-phase
MaxRSS not recorded. (`--status` prints "batches: 210 of 50" because it uses
the runner's default BATCH_SIZE for display only.)
- Still to do:
  - Compare base-512 to small-256 P_err for hod 2155 (compare_perr.py).
    That settles how much of the old base-256 excess was the mesh vs the
    weak base-256 cubic.
  - Back up `AbacusBase/results` (scratch purge).
  - **Re-htar `AbacusBase/shifted_fields/`.** The 2026-11-04 HPSS copy
    predates job 60856968 (Nov 5), so it lacks
    `shifted_fields_redshift_axisrsd_2_N_512_c_0.npy` and has the old real
    512 file. Use a new archive name to keep the old one.
- pack_hods.sh bundles hold results, params and catalogs only. The inputs
  behind the plots are separate downloads:
  - shifted fields: base real/RSD-axis-2 at 256 (0.27/0.40 GB) and 512
    (2.15/3.22 GB) in `$PSCRATCH/qso_sbp/Heavy_files_Andrej/AbacusBase/shifted_fields/`;
    small at 256 in `.../AbacusSmall/shifted_fields/`;
  - raw ICs: `$PSCRATCH/abacus_ics_000/` (Blosc asdf; public via Globus).

History of the item follows.

**Update 2026-11-05: the 512 shifted fields now exist.** Batch job 60856968
(perlmutter_shifted_fields.sbatch, CPU node, unedited
my_generate_shifted_fields.py in nbodykit_env_25 + asdf 2.12 + abacusutils
2.0.1) exited 0 after 169 min.
- Measured peak: generator RSS 306 GiB, node 327 GiB (60 s samples). It
  needs a CPU node (476 GiB) and could never fit the 224 GiB GPU nodes. The
  ×8 extrapolation (~365 GiB) was the right order.
- The run also REWROTE `shifted_fields_real_N_512_c_0.npy`.
- **Neither file has been checked yet.** check_shifted_fields.py wasn't in
  the submit dir (`$PSCRATCH/qso_sbp/Abacus_elg_runs`), so both checks were
  skipped, yet the sbatch printed "passed the checks". That was a bug, now
  fixed: it warns at start and ends "NOT verified" when checks are skipped.
- **Checks run 2026-11-07** (512 vs 256, both files):
  - d1 and dz: r = 1.000, power ratio 1.000 at low k. Same ICs, phases,
    axes and growth. d2, dG2, dG2par: r 0.91–0.99, consistent.
  - **d3 differs:** r 0.26–0.30 at low k (→0.03–0.07 at k_Nyq), power
    ratio ~11 (→35–53). Real-space std 0.25 (256) vs 4.24 (512), while the
    other fields grew only ~2×.
  - It is NOT just resolution. A synthetic test with the library's
    definitions (sharp k≤0.5 cubic, per-k-bin orthogonalization, 128³ vs
    64³ on 500 Mpc/h) reproduces the user's d1/d2/dG2 numbers but gives
    d3 r ≈ 0.88, ratio ≈ 1.0.
    - Basis of that test: the PUBLIC `d3_smooth(km=0.5)`, i.e. sharp
      |k| ≤ 0.5 then cube. It used UNSHIFTED operators on a Gaussian
      BBKS field, real space only, and sequential per-k-bin Gram–Schmidt.
    - The user's own tng_lib differs from the public one (131,896 vs
      101,926 B; it has generate_fields_rsd_new_growth). Their d3 may
      differ; check `grep -n -A6 "def d3_smooth"` in it.
    - Definitive test: regenerate base 256 with the current script into a
      separate folder and compare with check_shifted_fields.py.
  - The Sep 22 real 512 file already had the large d3 (slab std 4.3). So
    the d3 definition changed between the Sep 21 256 run and Sep 22, or
    the script's cubic smoothing depends on Nmesh.
  - tng_lib has two cubic definitions: `d3_smooth` (sharp cut k ≤ 0.5)
    and `generate_fields_new_smooth_cubic` (Gaussian R = 20 Mpc/h, 8
    fields).
  - The generator (grep, 2026-11-07) does the following:
    - `Nmesh = 512` (line 28). `dlin = ArrayMesh(IC).paint(Nmesh=Nmesh)`
      resamples the IC.
    - Real space: `generate_fields_new_growth` then `orthogonalize`.
    - Redshift space: a loop over `axis_rsd` calling
      `generate_fields_rsd_new_growth` (NOT in the public lib, so the
      user's tng_lib differs) then `orthogonalize_rsd(..., Nmufid)`.
    - Saves `[d1,d2,dG2,d3]` / `[dz,d1,d2,dG2,dG2par,d3]` with d2/dG2/d3
      orthogonalized.
    - Public `generate_fields_new_growth` builds d3 as
      `d3_smooth(km=0.5)`, a sharp k ≤ 0.5 cut that doesn't depend on
      Nmesh.
  - **Conclusion: the 256 files' d3 is a different cubic definition, not a
    resolution effect.** The synthetic test with a fixed k ≤ 0.5 cubic
    gives 512/256 std ratios of d1 1.49, d2 2.22, dG2 2.25, d3 1.16. The
    user's files give 1.66, 2.16, 2.21 and **16.8**. The 512 d3 (std ≈ 2×
    d1) looks like the k ≤ 0.5 cubic; the 256 d3 (std 0.19× d1) is ~10×
    too small for it.
  - So the Sep 21 256 run used another cubic (heavier smoothing?) than the
    code produces now.
  - **Resolved (2026-11-07): base-256 is the odd one out.**
    - The user's tng_lib.py was last modified 2026-06-05, so the library
      didn't change.
    - my_generate_shifted_fields.py was edited 2026-11-04, so its Sep 21
      state is unknowable.
    - The small-box real 256 field stds are d1 3.06, d2 5.92, dG2 1.80,
      d3 5.86. d3/d1 is ≈1.9 for small-256 and ≈2.0 for base-512, but
      0.19 for base-256.
    - So base-512 and small-256 share the current cubic, and the
      base-512 vs small comparison (the point of the rerun) is like for
      like. Only the old base-256 fields (and all 10500 base-256 lrgs.py
      results) used a different, much weaker cubic.
  - Caveat: the earlier "base-256 P_err inflated ~0.09 in every μ bin vs
    small = Nyquist artifact" conclusion compared different cubics.
    Part of that inflation may come from base-256's weak d3. The base-512
    results will separate the two.
- **lrgs.py 512 test, job 60988941_0** (1 node, 30 min, 16 workers at
  MEM_PER_WORKER_GB=30):
  - TIMEOUT as planned. Batch-step MaxRSS 392 GiB = **24.5 GiB/worker**,
    82 % of the node.
  - 324 real-space outputs in 24.2 min = **72 s/catalog/worker** (31 s at
    256). 0 RSD outputs.
  - 20 real outputs per worker with 0 RSD implies BATCH_SIZE was ~50 (the
    runner default; the user's sbatch copy apparently doesn't export 12).
    So workers never left the real-space loop, and **RSD-phase memory and
    speed are unmeasured.** RSD loads 6 fields instead of 4, so expect
    somewhat more memory.
  - Full-run recommendation: `--export=ALL,NMESH=512,MEM_PER_WORKER_GB=36,
    BATCH_SIZE=14 --array=0-63 -t 01:30:00`.
    - 13 workers/node, which tolerates ~36 GiB/worker in the RSD phase.
    - 750 batches, 11–12 per node, so one round.
    - About 14×(72 s + RSD) + startup ≈ 40–60 min wall, ~35–50
      node-hours.
    - Pass BATCH_SIZE explicitly. Resubmitting the same command resumes.
  - Bullets below that predate this update are history.

- Why: base-box results at Nmesh=256 have k_Nyq = 0.402 = k_max, so k ≳ 0.2
  is contaminated by the mesh (see "Field-level P_err results"). The small
  box at 256 is fine (k_Nyq = 1.61) and needs no rerun.
- Status of `$PSCRATCH/qso_sbp/Heavy_files_Andrej/AbacusBase/shifted_fields/`
  (checked 2026-11-04):
  - `shifted_fields_real_N_512_c_0.npy` is complete and sane. Written
    Sep 22 09:31, 2,147,483,776 B = exactly (4, 512³) float32 + 128 B
    header. All four fields are non-zero in the first and last 4 slabs, with
    matching stds (2.19/2.18, 3.27/3.26, 0.94/0.94, 4.31/4.37). Don't
    regenerate it.
  - `shifted_fields_redshift_axisrsd_2_N_512_c_0.npy` is MISSING (a complete
    file is 3,221,225,600 B = (6, 512³) float32 + 128 B).
  - lrgs.py only reads axis 2 (`axis_rsd=2` is hardcoded), so axes 0/1 at
    512 are not needed.
  - Integrity is not correctness. The real-space 512 file is verified only
    to be complete and filled. Whether it is the RIGHT field (same IC phases,
    operator definitions and normalisation as the 256 file every result
    used) was unverified as of 2026-11-05.
  - `check_shifted_fields.py FILE REF` tests that: FFT both files, then
    compare the shared modes per k-bin via r(k) and power ratio, the low-k
    r matrix, and per-field verdicts. 22 s and 3.1 GiB at 512 vs 256 (4
    cores), so it is fine on a login node.
  - Field order is lrgs.py's unpacking: real `d1, d2, dG2, d3`; redshift
    `dz, d1, d2, dG2, dG2par, d3`.
  - Amplitude is tested only for the fields linear in δ1 at low k (d1; dz
    and d1), in the lowest k-bin with a ±10 % tolerance. A synthetic test
    (cutoffs 0.8 vs 0.4 h/Mpc) showed why. The nonlinear operators'
    amplitudes legitimately depend on the grid cutoff: δ² ratio 1.1–1.7 and
    δ³ ≈ 5, because δ³ ≈ 3σ²δ. Their r stayed 0.93–0.98.
  - Wrong seed or transposed axes give r ≈ 0. A 0.769 growth error gives
    ratio 0.59. Permuted fields trip the order check.
- **Why it is missing: OUT_OF_MEMORY, not a timeout** (sacct, Sep 21–23).
  Interactive sessions 58752877 / 58753239 / 58754046 were OOM-killed after
  20 / 33 / 52 min, each at MaxRSS 224.35 GiB = 99.97 % of their 229802M
  limit. 58752551 was Ctrl-C'd at 15 min and was already at 135 GiB.
  - Every interactive session had ReqMem 229802M, the memory of a whole
    256 GB node. **Confirmed GPU node:** 58754046 ran on nid001885 with
    AllocCPUS 128 and `gres/gpu:a100=4`. A CPU node gives 487802M = 476 GiB
    (the lrgs batch job's ReqMem). Always request `-C cpu` for this step.
  - The Sep 21 sessions (58699492/58699511/58702726/58702740, presumably
    the 256 generation, since their timing matches the files) peaked at
    45.65 GiB. Scaling ×8 for 512³ gives ~365 GiB, so it should fit a full
    CPU node. This is an extrapolation, not a measurement.
  - An earlier estimate in this session (generation needs only 30–60 GB and
    is "comfortable on one node") was wrong. The user doubted it, correctly.
- Next step: rerun ONLY the redshift-space step, axis 2 only, on a CPU node,
  as a **batch job**. The user does not want to babysit interactive
  sessions, so don't suggest salloc for long runs.
  `slurm/perlmutter_shifted_fields.sbatch` does this. Pass the generator
  command after the script name; it runs unchanged in the submit
  directory:
  `sbatch perlmutter_shifted_fields.sbatch python <generator>.py ...`
  - It requests `-C cpu -q regular -t 03:00:00`; use `-t 05:00:00` if the
    generator redoes real space plus axes 0, 1, 2.
  - It runs check_shifted_fields.py on the real-space file before
    generating and on the new axis-2 file after.
  - It logs `[mem]` lines (node used + generator RSS) every minute,
    interleaved with the generator's output (PYTHONUNBUFFERED=1), and
    writes the same numbers to `mem_<jobid>.tsv`.
  - It exits non-zero if the generator fails, the file is missing, or a
    check flags a problem.
  - Tested locally with a stand-in generator: the normal, OOM (SIGKILL ->
    137), real-file-rewrite and no-command paths. It has not been run on
    Perlmutter yet.
- The generator is the user's **`my_generate_shifted_fields.py`**. It takes
  no arguments; all settings, including Nmesh, are inside the file. It
  hasn't been seen here yet (the 256 run wrote real, then axes 0, 1, 2 in
  one process), so axis-2-only edits and its memory hog are unknown. Ask
  for the file and don't say "your generator" without naming it: the user
  didn't know what was meant.
  - It presumably calls `generate_fields_new` / `generate_fields_rsd_new`
    from `lib/tng_lib.py`. The Hi-Fi_mocks checkout is
    `$PSCRATCH/Quijote_png/hifimocks_code/Hi-Fi_mocks`; the public repo
    is andrejobuljen/Hi-Fi_mocks.
  - The public routines put one particle per cell of `dlin`'s own grid and
    paint back onto it, so at 512³ they need only ~20–30 GiB (catalog
    8 GiB + displacements 3 GiB + position-update temporaries). The 224+
    GiB is therefore elsewhere in the script (IC loading/resampling?). The
    256 run's 45.65 GiB likewise can't be the routines alone (~3–4 GiB at
    256), so the ×8 → 365 GiB extrapolation is shaky.
  - The public `d3_smooth` applies a sharp cutoff at k = 0.5 h/Mpc, and
    painting uses `compensated=True`.
  - **Env:** the script imports `asdf` (line 14), which `nbodykit_env_25`
    lacked. The first batch job (2026-11-05) died on
    `ModuleNotFoundError: asdf`. The user then installed asdf 2.12.0 there.
    The env is Python **3.7.12**, numpy **1.21.6**, at
    `/global/cfs/projectdirs/m4031/jsull/env/envs/nbodykit_env_25`.
  - **The IC files ARE Blosc-compressed.** check_generator_env.py on
    `ic_dens_N576.asdf` gave "not 'blsc'". The decoder comes from
    abacusutils, whose `[asdf.extensions]` entry point registers it with
    asdf automatically, so the script itself needs no
    `import abacusnbody`.
  - Pin `abacusutils==2.0.1`, the newest release for Python 3.7 (2.1+
    needs 3.9). Unpinned, pip can quietly fall back to ancient releases
    (0.3.0 in a sandbox test) that predate the entry point. All 2.0.1
    deps have cp37 wheels compatible with numpy 1.21.6: numba 0.56.4,
    astropy 4.3.1, blosc 1.10.6, scipy 1.7.3, h5py 3.8.0, msgpack 1.0.5,
    parallel-numpy-rng 0.1.2. This was resolved by `pip download`, not
    tested on a real 3.7 interpreter (uv no longer ships 3.7).
  - Fix: install into `nbodykit_env_25` with pip, keeping numpy pinned so
    nbodykit's compiled extensions (and lrgs.py in the same env) are
    untouched:
    `python -m pip install asdf "numpy==$(python -c 'import numpy; print(numpy.__version__)')"`
  - `check_generator_env.py SCRIPT [--ic FILE]` verifies the env on a login
    node. It runs only the script's sys.path edits and imports (all
    failures listed at once), then reads values from `ic_dens_N576.asdf` to
    catch 'blsc'. The sbatch also runs it at job start and exits 4 if it
    fails.
  - The sbatch no longer lets `slurm/env.local.sh` (repo copy:
    `CONDA_ENV=abacus`) override `CONDA_ENV`. Before, an explicit
    `--export=ALL,CONDA_ENV=...` could be silently replaced.
- If it still OOMs at 476 GiB, there are two options, and both need the
  generator script:
  - write each field straight to disk with `np.lib.format.open_memmap`
    (float32) and free intermediates;
  - run the generator MPI-distributed over 2 nodes. srun IS the right tool
    here, unlike for lrgs.py, because this is one computation.
- The 256 shifted fields were generated by the user's own job (owner jsull,
  Sep 21 11:18 → 11:24 → 11:31 → 11:37, ~6–7 min per file), not precomputed
  by someone else as an earlier note claimed. "Heavy_files_Andrej" is only a
  directory name. At 512 expect ~1 h per redshift-space file.
- Shifted fields are stored as **float32** (file sizes prove it), so the
  loaded arrays are half the float64 figures used in the memory estimate.
  Keep `MEM_PER_WORKER_GB=30` until a real 512 run's MaxRSS is measured.
- lrgs.py memory, measured at 256: the base calibration job (58748346_0, 62
  workers) peaked at MaxRSS 441 GiB = 7.1 GiB/worker = 93 % of the node.
  That is more than the earlier per-worker estimate (median 2.9 GiB), so 30
  at 512 is unproven. The mesh-dependent part (the part that scales ×8) has
  not been separated from the catalog part yet. The small-box run measures
  it, since it uses the same Nmesh with 64× smaller catalogs:
  `sacct --name=lrgs_small` MaxRSS of the `.batch` steps ÷ that run's worker
  count.
- ICs on hand are N576/N1152/N2048 (no 512); 512 = k-space truncation of a
  finer IC, or use 576 to match an IC grid exactly (~47% more cost).
- Then: `Nmesh = 256` -> `512` in lrgs.py (one line), and submit with
  `--export=ALL,NMESH=512,MEM_PER_WORKER_GB=30` (~16 workers/node). Either
  64 nodes with the default BATCH_SIZE=12 and `-t 03:00:00` (~1.9-2.3 h), or
  `--array=0-95` with BATCH_SIZE=7 and `-t 02:00:00` (~1.1-1.4 h). An earlier
  suggested command used BATCH_SIZE=10 on 64 nodes, which the runner flags as
  ~2x slow -- don't reuse it.
- Outputs are named `..._Nmesh_512...` and coexist with the 256 results; pass
  NMESH=512 to `--status` to count them.
- Purge risk is resolved. The user htar'd `shifted_fields/` to their HPSS
  home on 2026-11-04. If scratch purges the files (~8 weeks without access),
  restore them with `htar -xvf`.

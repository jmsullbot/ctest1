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

## OPEN ITEM: base-box rerun of lrgs.py at Nmesh = 512 (not done as of 2026-11-04)

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

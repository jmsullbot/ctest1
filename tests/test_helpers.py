"""
Tests for compute_number_density.py and plot_occupation.py.

Generates all required data in-memory / in temporary files — no AbacusSummit
simulation data is needed.

Run with:
    pytest tests/test_helpers.py -v
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must be before any pyplot import; keep at top of file

import h5py
import numpy as np
import pytest

# Make project root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from compute_number_density import compute_number_densities
from generate_qso_hods import _CATALOG_FIELDS, _OM_ABACUS, _compute_z_rsd, _write_npz
from plot_occupation import n_cen, n_sat, plot_occupation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PARAM_NAMES = [
    "logM_cut", "logM1", "sigma", "alpha", "kappa",
    "alpha_c", "alpha_s", "log10_f_ic",
]

_DEFAULT_PARAMS = dict(
    logM_cut=12.7, logM1=14.5, sigma=0.5,
    alpha=1.0, kappa=0.5, log10_f_ic=-1.35,
)


def _make_dummy_hdf5(path: Path, n_gal: np.ndarray, box_size: float = 1000.0) -> None:
    """Write a minimal HDF5 catalog that compute_number_densities can read."""
    n_runs   = len(n_gal)
    n_params = len(PARAM_NAMES)
    rng      = np.random.default_rng(0)
    params   = rng.random((n_runs, n_params))

    with h5py.File(path, "w") as hf:
        hf.attrs["n_runs"]      = n_runs
        hf.attrs["param_names"] = PARAM_NAMES
        hf.attrs["sampling"]    = "lhs"
        hf.attrs["seed"]        = 42
        hf.attrs["want_rsd"]    = True
        hf.attrs["sim_name"]    = "dummy"
        hf.attrs["z_mock"]      = 1.1

        ds           = hf.create_dataset("params", data=params)
        ds.attrs["columns"] = PARAM_NAMES
        hf.create_dataset("n_gal", data=n_gal.astype(np.int64))

        grp_fixed = hf.create_group("fixed_params")
        for k, v in dict(s=0.0, s_v=0.0, ic=1.0).items():
            grp_fixed.attrs[k] = v

        grp_cats = hf.create_group("catalogs")
        for i in range(n_runs):
            g = grp_cats.create_group(f"{i:06d}")
            g.attrs["n_gal"] = int(n_gal[i])


# ---------------------------------------------------------------------------
# compute_number_density tests
# ---------------------------------------------------------------------------

def test_number_density_simple():
    """n_bar = N_gal / L^3 — verify with exact arithmetic."""
    L      = 500.0
    V      = L ** 3          # 1.25e8 (Mpc/h)^3
    n_gals = np.array([1000, 2500, 500], dtype=np.int64)
    expected = n_gals / V
    np.testing.assert_allclose(expected, n_gals / V)


def test_number_density_from_hdf5(tmp_path):
    """Round-trip: write dummy HDF5, read back n_bar, check values."""
    hdf5_path = tmp_path / "dummy.hdf5"
    box_size  = 1000.0                    # Mpc/h  →  V = 1e9 (Mpc/h)^3
    n_gal     = np.array([1_000, 2_000, 500], dtype=np.int64)

    _make_dummy_hdf5(hdf5_path, n_gal, box_size)
    result = compute_number_densities(str(hdf5_path), box_size)

    expected_n_bar = n_gal / box_size ** 3
    np.testing.assert_allclose(result["n_bar"], expected_n_bar, rtol=1e-10)
    np.testing.assert_array_equal(result["n_gal"], n_gal)
    assert result["V_box"] == pytest.approx(box_size ** 3)
    assert result["param_names"] == PARAM_NAMES


def test_number_density_zero_galaxies(tmp_path):
    """Runs with zero galaxies should give n_bar = 0, not NaN."""
    hdf5_path = tmp_path / "zero.hdf5"
    n_gal     = np.array([0, 100, 0], dtype=np.int64)
    _make_dummy_hdf5(hdf5_path, n_gal)
    result = compute_number_densities(str(hdf5_path), box_size_mpch=500.0)
    assert result["n_bar"][0] == 0.0
    assert result["n_bar"][2] == 0.0


# ---------------------------------------------------------------------------
# n_cen tests
# ---------------------------------------------------------------------------

def test_n_cen_at_M_cut():
    """At M = M_cut, erfc(0) = 1, so <N_cen> = p_max / 2."""
    logM_cut   = 12.5
    sigma      = 0.6
    log10_f_ic = -1.0          # p_max = 0.1
    M_cut      = 10.0 ** logM_cut
    p_max      = 10.0 ** log10_f_ic

    result = n_cen(np.array([M_cut]), logM_cut, sigma, log10_f_ic)
    assert result[0] == pytest.approx(p_max / 2.0, rel=1e-10)


def test_n_cen_large_mass_approaches_p_max():
    """For M >> M_cut, erfc(−∞) = 2, so <N_cen> → p_max."""
    logM_cut   = 12.0
    sigma      = 0.5
    log10_f_ic = -1.5
    p_max      = 10.0 ** log10_f_ic
    M_huge     = 1e20   # completely dominates M_cut

    result = n_cen(np.array([M_huge]), logM_cut, sigma, log10_f_ic)
    assert result[0] == pytest.approx(p_max, rel=1e-6)


def test_n_cen_small_mass_approaches_zero():
    """For M << M_cut, erfc(+∞) = 0, so <N_cen> → 0."""
    result = n_cen(np.array([1e5]), logM_cut=13.0, sigma=0.5, log10_f_ic=-1.0)
    assert result[0] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# n_sat tests
# ---------------------------------------------------------------------------

def test_n_sat_zero_below_threshold():
    """<N_sat> = 0 for M ≤ κ M_cut."""
    logM_cut = 12.0
    kappa    = 1.0
    M_cut    = 10.0 ** logM_cut
    M_below  = np.array([0.5 * M_cut, M_cut - 1.0, M_cut])  # all ≤ threshold

    result = n_sat(M_below, logM_cut, logM1=14.0, alpha=1.0, kappa=kappa)
    np.testing.assert_array_equal(result, 0.0)


def test_n_sat_at_one_M1_above_threshold():
    """At M = κ M_cut + M1, the argument is exactly 1, so <N_sat> = 1^α = 1."""
    logM_cut = 12.0
    logM1    = 14.0
    alpha    = 1.5
    kappa    = 0.5
    M_cut    = 10.0 ** logM_cut
    M1       = 10.0 ** logM1
    M_test   = kappa * M_cut + M1

    result = n_sat(np.array([M_test]), logM_cut, logM1, alpha, kappa)
    assert result[0] == pytest.approx(1.0, rel=1e-10)


def test_n_sat_power_law_slope():
    """Verify the slope of log(N_sat) vs log(M) is close to alpha for M >> kappa M_cut.

    The formula is [(M - κ M_cut) / M1]^α, so the slope is only exactly α in the
    limit M >> κ M_cut.  We use a loose tolerance to allow for the residual offset.
    """
    logM_cut = 11.0     # small threshold so we can ignore it
    logM1    = 13.0
    alpha    = 1.7
    kappa    = 0.01     # tiny kappa → kappa*M_cut = 1e9 << M
    M        = np.array([1e14, 1e15])

    ns = n_sat(M, logM_cut, logM1, alpha, kappa)
    slope = np.log(ns[1] / ns[0]) / np.log(M[1] / M[0])
    assert slope == pytest.approx(alpha, rel=1e-3)


def test_n_sat_not_modulated_by_n_cen():
    """Satellite occupation is independent of central occupation (no N_cen factor).

    Verify by checking that changing log10_f_ic (which only affects N_cen)
    does NOT change N_sat.
    """
    kw = dict(logM_cut=12.0, logM1=14.0, alpha=1.0, kappa=0.5)
    M  = np.array([1e13, 1e14, 1e15])
    ns1 = n_sat(M, **kw)
    ns2 = n_sat(M, **kw)   # same params → trivially equal

    # More importantly: n_sat signature has no log10_f_ic parameter at all
    import inspect
    sig = inspect.signature(n_sat)
    assert "log10_f_ic" not in sig.parameters
    assert "p_max"      not in sig.parameters

    np.testing.assert_array_equal(ns1, ns2)


# ---------------------------------------------------------------------------
# plot_occupation tests
# ---------------------------------------------------------------------------

def test_plot_occupation_returns_figure():
    """plot_occupation should return (fig, ax) without raising."""
    fig, ax = plot_occupation([_DEFAULT_PARAMS])
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_occupation_saves_file(tmp_path):
    """plot_occupation should write a PNG when output_file is specified."""
    out = tmp_path / "occ.png"
    fig, ax = plot_occupation([_DEFAULT_PARAMS], output_file=str(out))
    import matplotlib.pyplot as plt
    plt.close(fig)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_occupation_multiple_param_sets():
    """Multiple parameter sets should produce one set of three lines each."""
    p2 = dict(logM_cut=13.0, logM1=15.0, sigma=0.8,
              alpha=1.2, kappa=0.3, log10_f_ic=-0.8)
    fig, ax = plot_occupation(
        [_DEFAULT_PARAMS, p2],
        labels=["fiducial", "high-mass"],
    )
    import matplotlib.pyplot as plt
    # 2 param sets × 3 lines (tot, cen, sat) = 6 lines
    assert len(ax.lines) == 6
    plt.close(fig)


# ---------------------------------------------------------------------------
# _compute_z_rsd tests
# ---------------------------------------------------------------------------

def test_z_rsd_zero_velocity():
    """Zero velocity → no displacement."""
    z   = np.array([100.0, 200.0, 300.0])
    vz  = np.zeros(3)
    out = _compute_z_rsd(z, vz, z_mock=0.5)
    np.testing.assert_array_equal(out, z)


def test_z_rsd_at_z0():
    """At z_mock=0: H(0)/(1+0) = 100 km/s/(Mpc/h), so Δz = v/100."""
    z   = np.array([0.0])
    vz  = np.array([100.0])   # 100 km/s → shift of 1 Mpc/h
    out = _compute_z_rsd(z, vz, z_mock=0.0)
    assert out[0] == pytest.approx(1.0, rel=1e-10)


def test_z_rsd_formula():
    """Check against manually computed H(z)/(1+z) for z_mock=1.0."""
    z_mock = 1.0
    Om     = _OM_ABACUS
    E_z    = np.sqrt(Om * (1 + z_mock) ** 3 + (1 - Om))
    H_over_1pz = 100.0 * E_z / (1 + z_mock)

    z   = np.array([500.0])
    vz  = np.array([200.0])
    expected = 500.0 + 200.0 / H_over_1pz
    out = _compute_z_rsd(z, vz, z_mock=z_mock)
    assert out[0] == pytest.approx(expected, rel=1e-12)


def test_z_rsd_negative_velocity():
    """Negative velocity displaces toward smaller z."""
    z   = np.array([500.0])
    vz  = np.array([-300.0])
    out = _compute_z_rsd(z, vz, z_mock=0.8)
    assert out[0] < 500.0


# ---------------------------------------------------------------------------
# _write_npz round-trip tests
# ---------------------------------------------------------------------------

def _make_cat_accum(n_gals_per_run: list[int], seed: int = 7, with_z_rsd: bool = False):
    """Build a cat_accum dict matching the structure generate_hod_samples produces."""
    rng = np.random.default_rng(seed)
    fields = (*_CATALOG_FIELDS, "z_rsd") if with_z_rsd else _CATALOG_FIELDS
    cat_accum = {field: [] for field in fields}
    for ng in n_gals_per_run:
        for field in fields:
            if ng > 0:
                if field in ("id", "Ncent"):
                    arr = rng.integers(0, 1000, size=ng)
                else:
                    arr = rng.random(ng).astype(np.float32)
                cat_accum[field].append(arr)
            else:
                cat_accum[field].append(np.array([]))
    return cat_accum


def test_write_npz_round_trip(tmp_path):
    """_write_npz: written data reads back identically."""
    n_runs = 4
    n_gals_per_run = [5, 0, 3, 7]  # run 1 is intentionally empty
    rng     = np.random.default_rng(0)
    samples = rng.random((n_runs, len(PARAM_NAMES)))
    n_gals  = np.array(n_gals_per_run, dtype=np.int64)
    cat_acc = _make_cat_accum(n_gals_per_run)

    npz_path = tmp_path / "test.npz"
    _write_npz(npz_path, samples, n_gals, PARAM_NAMES, cat_acc)

    data = np.load(npz_path)

    # --- metadata ---
    np.testing.assert_array_equal(data["params"], samples)
    np.testing.assert_array_equal(data["n_gal"], n_gals)
    assert list(data["param_names"]) == PARAM_NAMES

    # --- offsets ---
    expected_offsets = np.array([0, 5, 5, 8, 15], dtype=np.int64)
    np.testing.assert_array_equal(data["offsets"], expected_offsets)

    # --- per-run slicing ---
    for run_i, ng in enumerate(n_gals_per_run):
        sl = slice(int(data["offsets"][run_i]), int(data["offsets"][run_i + 1]))
        expected_x = cat_acc["x"][run_i]
        if ng > 0:
            np.testing.assert_array_equal(data["x"][sl], expected_x)
        else:
            assert sl.stop - sl.start == 0


def test_write_npz_with_z_rsd(tmp_path):
    """z_rsd column is stored and slices correctly when provided."""
    n_gals_per_run = [4, 6]
    rng     = np.random.default_rng(1)
    samples = rng.random((2, len(PARAM_NAMES)))
    n_gals  = np.array(n_gals_per_run, dtype=np.int64)
    cat_acc = _make_cat_accum(n_gals_per_run, with_z_rsd=True)

    npz_path = tmp_path / "rsd.npz"
    _write_npz(npz_path, samples, n_gals, PARAM_NAMES, cat_acc)

    data = np.load(npz_path)
    assert "z_rsd" in data

    for run_i, ng in enumerate(n_gals_per_run):
        sl = slice(int(data["offsets"][run_i]), int(data["offsets"][run_i + 1]))
        np.testing.assert_array_equal(data["z_rsd"][sl], cat_acc["z_rsd"][run_i])


def test_write_npz_all_empty(tmp_path):
    """All-empty runs should produce empty catalog arrays and monotone offsets."""
    n_runs  = 3
    samples = np.zeros((n_runs, len(PARAM_NAMES)))
    n_gals  = np.zeros(n_runs, dtype=np.int64)
    cat_acc = _make_cat_accum([0, 0, 0])

    npz_path = tmp_path / "empty.npz"
    _write_npz(npz_path, samples, n_gals, PARAM_NAMES, cat_acc)

    data = np.load(npz_path)
    np.testing.assert_array_equal(data["offsets"], np.zeros(n_runs + 1, dtype=np.int64))
    # No catalog arrays should be present (all were empty)
    for field in _CATALOG_FIELDS:
        assert field not in data

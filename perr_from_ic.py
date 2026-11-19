#!/usr/bin/env python
"""Field-level P_err from an initial-condition grid and a galaxy catalog.

    python perr_from_ic.py --ic ic_delta_256.npy --catalog catalogs/002155.npy \\
        --growth 59.911485 --out hod2155_N256

Inputs
  --ic       .npy, an N^3 grid of the linear density contrast. The grid IS the
             mesh: everything (operators, galaxies, P(k)) is done at this N.
             --nmesh M (< N) first truncates it in Fourier space to M^3.
  --growth   factor taking the IC to the output redshift, D(z_out)/D(z_IC).
             AbacusSummit ICs (z = 99) to z = 0.5 on c000: 59.911485. Use 1 if
             the grid is already at z_out.
  --catalog  .npy galaxy catalog: structured array with x, y, z (and z_rsd for
             redshift space), or an (Ngal, 3 or 4) array in that column order.
             Positions centred on 0 (our catalogs) are shifted by +L/2, as
             lrgs.py does; pass --no-centre-shift if yours are in [0, L).

Outputs (<out> from --out)
  <out>_real.npy  [k, P_err, P_true, P_model, r, beta1, beta2, betaG2, beta3, nbar]
  <out>_rsd.npy   [k, mu_centers, P_err, P_true, P_model, r, beta1, beta2, betaG2,
                   beta3, nbar], each (k, mu) quantity an (nk, Nmu) array
  <out>_transfer.png  Fig. 4 of arXiv:2505.03626 style: one row of panels
                  beta_1, beta_2, beta_G2, beta_3, P_err; real space black,
                  mu bins in darkening shades; P_err in h^-3 Mpc^3 with 1/nbar
  <out>_perr.png  P_err blown up, one panel each for real space and every
                  mu bin, with 1/nbar, k_Nyq and k_Nyq/2 marked
The .npy layout is lrgs.py's, so fit_perr.py and compare_perr.py read these
directly, e.g. to compare with the lrgs.py results for the same HOD.
--replot <out> redraws both figures from saved results without recomputing.
--save-fields DIR also writes the shifted fields in the generator's layout
(shifted_fields_real_N_<N>_c_0.npy, shifted_fields_redshift_axisrsd_2_N_<N>_c_0.npy).

Method (Schmittfull et al. 2019, 2021; the public Hi-Fi_mocks library)
  1. delta_1(q) = growth * IC (mean removed) at the N^3 grid points q.
  2. Lagrangian operators at q: d1 = delta_1; d2 = delta_1^2 - <>;
     G2 = sum_ij (k_i k_j / k^2 delta_1)^2 - delta_1^2;
     d3 = (delta_1 with |k| <= --d3-kmax, default 0.5 as in the library's
     d3_smooth)^3 - <>; in redshift space also G2par = (k_z^2 / k^2) G2.
  3. Zel'dovich displacement Psi = i k delta_1 / k^2 (redshift space: the z
     component times 1 + f). One particle per grid point carries each operator
     to q + Psi and is CIC-painted onto the same grid, compensated
     (--compensation shotnoise = nbodykit's default without interlacing).
  4. Model, Eq. (16) of arXiv:2505.03626 (real space = its f = 0 limit):
       delta_g = delta_Z - 3/7 f G2par + beta_1 d1 + beta_2 d2^perp
                 + beta_G2 G2^perp + beta_3 d3^perp,
     where delta_Z is the shifted unit field and G2par the shifted
     (k_z^2/k^2) G2, as in lrgs.py and the library's rsd_polynomial_field
     (dz - 3/7 f dG2par). The first two terms have fixed coefficient 1.
     d2, G2, d3 are Gram-Schmidt orthogonalized against the lower operators
     (d1 is not, and delta_Z is not part of the sequence), per k bin in real
     space and per (k, mu) bin in redshift space. The betas are
     <(delta_g - fixed) O_i*> / <|O_i|^2> in each bin, so they are flat and
     mu-independent on large scales, with beta_1 -> b_1 - 1.
     --real-no-dz drops delta_Z from the REAL-space model (beta_1 -> b_1).
     lrgs.py's real-space field files have no dz, so its real-space fit is
     presumably of that form.
  5. P_err = <|delta_g - model|^2>, P_true = <|delta_g|^2>, P_model,
     r = <delta_g model*> / sqrt(P_true P_model), with P = V <|FFT/N^3|^2>.
     k bins of width 2 pi / L starting at pi / L (lrgs.py's), mean |k| per bin;
     Nmu bins in |mu|.

Memory: ~3 GB at 256^3; ~30-40 GB at 512^3 (use a compute node for 512).
"""
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np

try:
    import scipy.fft as _sfft
except ImportError:          # numpy fallback: single-threaded and complex128
    _sfft = None

WORKERS = 8
_T0 = time.time()


def log(msg):
    print("[%7.1f s] %s" % (time.time() - _T0, msg))
    sys.stdout.flush()


def rfftn(x):
    if _sfft is not None:
        return _sfft.rfftn(x, workers=WORKERS)
    return np.fft.rfftn(x)


def irfftn(x, n):
    if _sfft is not None:
        return _sfft.irfftn(x, s=(n, n, n), axes=(0, 1, 2), workers=WORKERS)
    return np.fft.irfftn(x, s=(n, n, n), axes=(0, 1, 2))


# ------------------------------------------------------------------ mesh ---
class Mesh(object):
    """k-space helpers for an n^3 rfft grid: binning, compensation, coefficients."""

    def __init__(self, n, box, nmu, compensation):
        self.n, self.box, self.nmu = n, float(box), nmu
        self.vol = self.box ** 3
        kf = 2 * np.pi / self.box
        self.kf, self.knyq = kf, np.pi * n / self.box
        f1 = (np.fft.fftfreq(n, 1.0 / n) * kf).astype(np.float32)
        fz = (np.fft.rfftfreq(n, 1.0 / n) * kf).astype(np.float32)
        self.kvec = (f1[:, None, None], f1[None, :, None], fz[None, None, :])
        k2 = self.kvec[0] ** 2 + self.kvec[1] ** 2 + self.kvec[2] ** 2
        self.inv_k2 = np.zeros_like(k2)
        np.divide(1.0, k2, out=self.inv_k2, where=k2 > 0)
        self.kmag = np.sqrt(k2)
        del k2
        # rfft stores half the modes: count each (k, -k) pair twice
        wz = np.full(fz.size, 2.0, np.float32)
        wz[0] = 1.0
        if n % 2 == 0:
            wz[-1] = 1.0
        # k bins [ (i + 1/2) kf, (i + 3/2) kf ) up to k_Nyq, as lrgs.py's FFTPower
        self.kedges = np.arange(0.5 * kf, self.knyq + 1e-6 * kf, kf)
        self.nk = self.kedges.size - 1
        ik = np.floor(self.kmag / kf - 0.5).astype(np.int32)        # -1 for k = 0
        mu = np.zeros_like(self.kmag)
        np.divide(np.abs(self.kvec[2]) * np.ones_like(self.kmag), self.kmag,
                  out=mu, where=self.kmag > 0)
        imu = np.minimum((mu * nmu).astype(np.int32), nmu - 1)
        del mu
        valid = ((ik >= 0) & (ik < self.nk)).ravel()
        self.sel = np.flatnonzero(valid)
        del valid
        self.w_sel = np.broadcast_to(wz[None, None, :], self.kmag.shape).ravel()[self.sel]
        self.b1 = ik.ravel()[self.sel]
        self.b2 = (ik * nmu + imu).ravel()[self.sel]
        # per-bin coefficients are applied to EVERY mode: k = 0 uses bin 0 and
        # the corners beyond k_Nyq the last bin (like the library's interpolation)
        ikc = np.clip(ik, 0, self.nk - 1)
        self.app1 = ikc
        self.app2 = ikc * nmu + imu
        del ik, imu, ikc
        self.wsum1 = np.bincount(self.b1, self.w_sel, self.nk)
        self.wsum2 = np.bincount(self.b2, self.w_sel, self.nk * nmu)
        self.kmean = np.bincount(self.b1, self.w_sel * self.kmag.ravel()[self.sel],
                                 self.nk) / self.wsum1
        self.mu_edges = np.linspace(0.0, 1.0, nmu + 1)
        # CIC compensation, applied once to every painted field
        cell = self.box / n
        comp = np.ones(self.kmag.shape, np.float32)
        for kc in self.kvec:
            x = 0.5 * kc * cell
            if compensation == "shotnoise":    # Jing (2005); nbodykit, no interlacing
                c = np.sqrt(1.0 - 2.0 / 3.0 * np.sin(x) ** 2)
            else:                              # plain CIC window
                c = np.ones_like(x)
                np.divide(np.sin(x), x, out=c, where=x != 0)
                c = c ** 2
            comp = comp * c
        self.inv_comp = (1.0 / comp).astype(np.float32)

    def binned(self, x, two_d):
        """Mode-weighted mean of a real half-complex-grid array per bin."""
        b, ws, nb = ((self.b2, self.wsum2, self.nk * self.nmu) if two_d
                     else (self.b1, self.wsum1, self.nk))
        s = np.bincount(b, self.w_sel * x.ravel()[self.sel], nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = s / ws
        return out.reshape(self.nk, self.nmu) if two_d else out

    def cross(self, a, b, two_d):
        return self.binned((a * np.conj(b)).real, two_d)

    def apply(self, coef, two_d):
        """Spread one coefficient per bin onto every mode of the grid."""
        c = np.nan_to_num(np.asarray(coef, np.float64).ravel()).astype(np.float32)
        return c[self.app2 if two_d else self.app1]


def to_fourier(field, mesh):
    """Painted real-space field -> compensated, normalized Fourier field."""
    n = mesh.n
    return rfftn(field.reshape(n, n, n)) * np.float32(1.0 / n ** 3) * mesh.inv_comp


# --------------------------------------------------------- CIC painting ---
def cic_paint(u, n, values):
    """CIC-paint points at cell-unit positions u = (ux, uy, uz) in [0, n).
    values: list of per-point weights (None = 1). Returns flat float32 grids."""
    i0 = [np.floor(ua).astype(np.int32) for ua in u]
    t = [(ua - ia).astype(np.float32) for ua, ia in zip(u, i0)]
    out = [np.zeros(n ** 3, np.float32) for _ in values]
    for cx in (0, 1):
        ix = (i0[0] + cx) % n
        wx = t[0] if cx else 1.0 - t[0]
        for cy in (0, 1):
            ixy = ix * (n * n) + ((i0[1] + cy) % n) * n
            wxy = wx * (t[1] if cy else 1.0 - t[1])
            for cz in (0, 1):
                idx = ixy + (i0[2] + cz) % n
                w = wxy * (t[2] if cz else 1.0 - t[2])
                for o, v in zip(out, values):
                    o += np.bincount(idx, w if v is None else w * v, n ** 3).astype(np.float32)
    return out


def shifted_positions(psi, mesh, f_rsd):
    """Cell-unit positions of the grid particles after the Zel'dovich shift."""
    n = mesh.n
    cell = mesh.box / n
    g = np.arange(n, dtype=np.float32)
    u = []
    for a in range(3):
        shape = [1, 1, 1]
        shape[a] = n
        fac = (1.0 + f_rsd) if (f_rsd is not None and a == 2) else 1.0
        ua = (g.reshape(shape) + psi[a] * np.float32(fac / cell)) % n
        u.append(np.ascontiguousarray(ua, dtype=np.float32).ravel())
    return u


# ----------------------------------------------------------- operators ---
def lagrangian_fields(delta, mesh, d3_kmax, rsd):
    """Bias operators at the grid points, and the Zel'dovich displacement."""
    n = mesh.n
    K = mesh.kvec
    D = rfftn(delta)
    ops = {"d1": delta}
    d2 = delta * delta
    ops["d2"] = d2 - d2.mean()
    g2 = -d2
    del d2
    for i, j, fac in ((0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (0, 1, 2.0), (0, 2, 2.0), (1, 2, 2.0)):
        dij = irfftn(D * (K[i] * K[j] * mesh.inv_k2), n)
        g2 += np.float32(fac) * dij * dij
        del dij
    ops["G2"] = g2
    ds = irfftn(D * (mesh.kmag <= d3_kmax), n)
    d3 = ds * ds * ds
    ops["d3"] = d3 - d3.mean()
    del ds, d3
    if rsd:
        ops["G2par"] = irfftn(rfftn(g2) * (K[2] * K[2] * mesh.inv_k2), n)
    psi = [irfftn(D * (1j * K[a] * mesh.inv_k2), n) for a in range(3)]
    return ops, psi


def orthogonalize(ops, mesh, two_d):
    """Gram-Schmidt per k (or (k, mu)) bin: each operator minus its projection
    on the ones before it."""
    out = [ops[0]]
    for x in ops[1:]:
        y = x.copy()
        for o in out:
            with np.errstate(invalid="ignore", divide="ignore"):
                c = mesh.cross(y, o, two_d) / mesh.cross(o, o, two_d)
            y -= mesh.apply(c, two_d) * o
        out.append(y)
    return out


def fit_transfer(dg, ops, fixed, mesh, two_d):
    """beta_i per bin for mutually orthogonal ops; P_err, P_true, P_model, r."""
    target = dg if fixed is None else dg - fixed
    model = np.zeros_like(dg) if fixed is None else fixed.copy()
    betas = []
    for o in ops:
        with np.errstate(invalid="ignore", divide="ignore"):
            b = mesh.cross(target, o, two_d) / mesh.cross(o, o, two_d)
        betas.append(b)
        model += mesh.apply(b, two_d) * o
    err = dg - model
    V = mesh.vol
    perr = V * mesh.binned(np.abs(err) ** 2, two_d)
    ptrue = V * mesh.binned(np.abs(dg) ** 2, two_d)
    pmodel = V * mesh.binned(np.abs(model) ** 2, two_d)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = V * mesh.cross(dg, model, two_d) / np.sqrt(ptrue * pmodel)
    return perr, ptrue, pmodel, r, betas


# ----------------------------------------------------------------- I/O ---
def truncate_ic(delta, m):
    """Fourier-truncate an N^3 grid to m^3 (m < N, both even); Nyquist planes zeroed."""
    n = delta.shape[0]
    F = rfftn(delta)
    h = m // 2
    ix = np.r_[0:h, n - h:n]
    G = np.zeros((m, m, h + 1), dtype=F.dtype)
    G[:, :, :h] = F[np.ix_(ix, ix, np.arange(h))]
    G[h, :, :] = 0
    G[:, h, :] = 0
    return irfftn(G * np.float32((float(m) / n) ** 3), m).astype(np.float32)


def load_catalog(path, box, centre_shift):
    c = np.load(path)
    if c.dtype.names:
        cols = [c["x"], c["y"], c["z"]]
        zr = c["z_rsd"] if "z_rsd" in c.dtype.names else None
    else:
        a = np.asarray(c)
        if a.ndim != 2 or a.shape[1] < 3:
            sys.exit("catalog must be structured (x, y, z[, z_rsd]) or (Ngal, >=3)")
        cols = [a[:, 0], a[:, 1], a[:, 2]]
        zr = a[:, 3] if a.shape[1] > 3 else None
    if len(cols[0]) == 0:
        sys.exit("catalog %s is empty" % path)
    s = 0.5 * box if centre_shift else 0.0
    xyz = [np.mod(np.asarray(x, np.float64) + s, box) for x in cols]
    zrs = None if zr is None else np.mod(np.asarray(zr, np.float64) + s, box)
    return xyz, zrs


def galaxy_field(x, y, z, mesh):
    n = mesh.n
    u = [(np.asarray(a) * (n / mesh.box)).astype(np.float64) % n for a in (x, y, z)]
    counts = cic_paint(u, n, [None])[0]
    counts *= np.float32(n ** 3 / float(len(x)))
    counts -= 1.0
    return to_fourier(counts, mesh)


def save_result(fn, items):
    res = np.empty(len(items), dtype=object)
    for i, v in enumerate(items):
        res[i] = v
    np.save(fn, res, allow_pickle=True)


def save_fields(path, fields, mesh):
    n = mesh.n
    arr = np.empty((len(fields), n, n, n), np.float32)
    for i, F in enumerate(fields):     # back to the painted (compensated) real grid
        arr[i] = irfftn(F * np.float32(n ** 3), n)
    np.save(path, arr)


# --------------------------------------------------------------- plots ---
KLABEL = r"$k\,[h\,{\rm Mpc}^{-1}]$"
PANELS = ((r"$\beta_1$", "Blues"), (r"$\beta_2$", "Oranges"), (r"$\beta_{\mathcal{G}_2}$", "Greens"),
          (r"$\beta_3$", "Reds"), (r"$P_{\rm err}\,[h^{-3}{\rm Mpc}^3]$", "Purples"))


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        log("matplotlib not available: skipping the plots")
        return None


def load_results(prefix):
    """Arrays for plotting from <prefix>_real.npy (and _rsd.npy if present).
    The RSD betas are used only if they have the (nk, Nmu) shape this script writes."""
    a = np.load(prefix + "_real.npy", allow_pickle=True)
    R = {"k": np.asarray(a[0], float), "perr": np.asarray(a[1], float),
         "betas": [np.asarray(x, float) for x in a[5:9]] if len(a) >= 10 else None,
         "nbar": float(a[-1]), "perr_s": None, "betas_s": None, "mu": None}
    if os.path.exists(prefix + "_rsd.npy"):
        b = np.load(prefix + "_rsd.npy", allow_pickle=True)
        R["mu"], R["perr_s"] = np.asarray(b[1], float), np.asarray(b[2], float)
        shape = R["perr_s"].shape
        if len(b) >= 11 and all(np.shape(x) == shape for x in b[6:10]):
            R["betas_s"] = [np.asarray(x, float) for x in b[6:10]]
    return R


def plot_transfer(fn, R, title):
    """One row like Fig. 4 of arXiv:2505.03626: beta_1, beta_2, beta_G2, beta_3, P_err."""
    plt = _pyplot()
    if plt is None:
        return
    k, nmu = R["k"], (0 if R["mu"] is None else len(R["mu"]))
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.7))
    for p, (ax, (lab, cmap)) in enumerate(zip(axes, PANELS)):
        real = R["perr"] if p == 4 else (R["betas"][p] if R["betas"] else None)
        rsd = R["perr_s"] if p == 4 else (R["betas_s"][p] if R["betas_s"] else None)
        if rsd is not None:
            shades = plt.get_cmap(cmap)(np.linspace(0.35, 0.85, nmu))
            for j in range(nmu):
                ax.plot(k, rsd[:, j], color=shades[j], lw=1.2, label=r"$\mu = %.2f$" % R["mu"][j])
        if real is not None:
            ax.plot(k, real, color="k", lw=1.3, label="real space", zorder=3)
        if p == 4:
            ax.axhline(1.0 / R["nbar"], color="0.45", ls="--", lw=1.2)
        if real is None and rsd is None:
            ax.text(0.5, 0.5, "not in file", transform=ax.transAxes, ha="center", color="0.5")
        ax.set_xscale("log")
        ax.set_xlabel(KLABEL, fontsize=12)
        lo, hi = ax.get_ylim()                     # headroom so the label clears the curves
        ax.set_ylim(lo, hi + 0.25 * (hi - lo))
        ax.text(0.12, 0.92, lab, transform=ax.transAxes, fontsize=16, va="top")
        ax.tick_params(labelsize=11)
    axes[0].legend(loc="lower left", fontsize=11)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    log("wrote %s" % fn)


def plot_perr_panels(fn, R, knyq, title):
    """P_err blown up: one panel for real space and one per mu bin, shared y."""
    plt = _pyplot()
    if plt is None:
        return
    k, nbar = R["k"], R["nbar"]
    curves = [("real space", R["perr"], "k")]
    if R["perr_s"] is not None:
        nmu = len(R["mu"])
        shades = plt.get_cmap("Purples")(np.linspace(0.5, 0.9, nmu))
        e = np.linspace(0.0, 1.0, nmu + 1)
        curves += [(r"$\mu \in [%.2f,\,%.2f]$" % (e[j], e[j + 1]), R["perr_s"][:, j], shades[j])
                   for j in range(nmu)]
    fig, axes = plt.subplots(1, len(curves), figsize=(4.6 * len(curves), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    # y range from k >= 0.02 so the few noisy largest-scale bins don't flatten the rest
    ys = np.concatenate([y[k >= 0.02] for _, y, _ in curves] + [[1.0 / nbar]])
    ys = ys[np.isfinite(ys)]
    lo, hi = np.percentile(ys, 0.5), np.percentile(ys, 99.5)
    pad = 0.08 * (hi - lo)
    for ax, (lab, y, c) in zip(axes, curves):
        ax.plot(k, y, color=c, lw=1.4)
        ax.axhline(1.0 / nbar, color="0.45", ls="--", lw=1.2, label=r"$1/\bar n$")
        for kk, ls, txt in ((knyq, "--", r"$k_{\rm Nyq}$"), (knyq / 2, ":", r"$k_{\rm Nyq}/2$")):
            if kk <= k.max() * 1.05:
                ax.axvline(kk, color="crimson", lw=0.9, ls=ls)
                ax.text(kk * 0.97, 0.97, txt, color="crimson", ha="right", va="top",
                        transform=ax.get_xaxis_transform(), fontsize=9)
        ax.set_xscale("log")
        ax.set_xlim(k.min() * 0.9, k.max() * 1.1)
        ax.set_xlabel(KLABEL, fontsize=12)
        ax.set_title(lab, fontsize=13)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel(r"$P_{\rm err}\,[h^{-3}{\rm Mpc}^3]$", fontsize=13)
    axes[0].legend(loc="upper left", fontsize=10, frameon=False)
    try:                                   # n-bar P_err on the right (matplotlib >= 3.1)
        sec = axes[-1].secondary_yaxis("right", functions=(lambda y: y * nbar, lambda y: y / nbar))
        sec.set_ylabel(r"$\bar n\,P_{\rm err}$", fontsize=13)
    except AttributeError:
        pass
    fig.suptitle(title + r"  —  $P_{\rm err}$", fontsize=15)
    fig.tight_layout()
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    log("wrote %s" % fn)


def make_plots(prefix, knyq, title):
    R = load_results(prefix)
    plot_transfer(prefix + "_transfer.png", R, title)
    plot_perr_panels(prefix + "_perr.png", R, knyq, title)


# ---------------------------------------------------------------- main ---
def main():
    global WORKERS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--ic", help="N^3 .npy linear density grid")
    ap.add_argument("--catalog", help="galaxy catalog .npy")
    ap.add_argument("--box", type=float, default=2000.0, help="box side [Mpc/h] (default 2000)")
    ap.add_argument("--growth", type=float, default=1.0, help="D(z_out)/D(z_IC) (default 1)")
    ap.add_argument("--f", type=float, default=0.759070, help="growth rate at z_out (default c000, z=0.5)")
    ap.add_argument("--nmesh", type=int, default=None, help="Fourier-truncate the IC to this N first")
    ap.add_argument("--nmu", type=int, default=3, help="mu bins (default 3)")
    ap.add_argument("--d3-kmax", type=float, default=0.5, help="cut on delta_1 before cubing [h/Mpc]")
    ap.add_argument("--compensation", choices=["shotnoise", "sinc"], default="shotnoise")
    ap.add_argument("--no-centre-shift", action="store_true", help="positions already in [0, L)")
    ap.add_argument("--no-rsd", action="store_true", help="real space only")
    ap.add_argument("--real-no-dz", action="store_true",
                    help="real space without the fixed delta_Z term (lrgs.py-like; beta_1 -> b_1)")
    ap.add_argument("--title", default=None, help="figure title (default: output name)")
    ap.add_argument("--replot", metavar="PREFIX", default=None,
                    help="only redraw the figures from PREFIX_real.npy / PREFIX_rsd.npy")
    ap.add_argument("--out", default=None, help="output prefix (default: <catalog name>_N<N>)")
    ap.add_argument("--save-fields", default=None, metavar="DIR", help="also save the shifted fields")
    ap.add_argument("--overwrite", action="store_true", help="allow overwriting outputs")
    ap.add_argument("--workers", type=int, default=8, help="FFT threads (default 8)")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    WORKERS = a.workers

    if a.replot:                              # figures only, from saved results
        R = load_results(a.replot)
        nmesh = a.nmesh or 2 * (len(R["k"]) + 1)       # lrgs-style bins: nk = N/2 - 1
        make_plots(a.replot, np.pi * nmesh / a.box, a.title or os.path.basename(a.replot))
        return
    if not (a.ic and a.catalog):
        ap.error("--ic and --catalog are required (or use --replot PREFIX)")

    delta = np.load(a.ic).astype(np.float32)
    if delta.ndim != 3 or len(set(delta.shape)) != 1 or delta.shape[0] % 2:
        sys.exit("--ic must be a cubic N^3 grid with even N, got %s" % (delta.shape,))
    if a.nmesh and a.nmesh < delta.shape[0]:
        log("truncating the IC from %d^3 to %d^3 in Fourier space" % (delta.shape[0], a.nmesh))
        delta = truncate_ic(delta, a.nmesh)
    n = delta.shape[0]
    delta *= np.float32(a.growth)
    delta -= delta.mean()
    out = a.out or "%s_N%d" % (os.path.splitext(os.path.basename(a.catalog))[0], n)
    targets = [out + "_real.npy"] + ([] if a.no_rsd else [out + "_rsd.npy"])
    if not a.overwrite:
        clash = [t for t in targets if os.path.exists(t)]
        if clash:
            sys.exit("refusing to overwrite %s (pass --overwrite)" % ", ".join(clash))
    if a.save_fields:
        os.makedirs(a.save_fields, exist_ok=True)

    log("IC %s: %d^3, box %g Mpc/h, growth %g, std(delta_1) = %.4f"
        % (a.ic, n, a.box, a.growth, float(delta.std())))
    mesh = Mesh(n, a.box, a.nmu, a.compensation)
    xyz, zrsd = load_catalog(a.catalog, a.box, not a.no_centre_shift)
    ngal = len(xyz[0])
    nbar = ngal / a.box ** 3
    rsd = (not a.no_rsd) and zrsd is not None
    if not a.no_rsd and zrsd is None:
        log("catalog has no z_rsd: real space only")
    log("catalog %s: %d galaxies, nbar = %.4e (h/Mpc)^3" % (a.catalog, ngal, nbar))

    ops, psi = lagrangian_fields(delta, mesh, a.d3_kmax, rsd)
    del delta
    log("Lagrangian operators and displacement done")

    # ------------------------------------------------------- real space
    names = ["d1", "d2", "G2", "d3"]
    painted = cic_paint(shifted_positions(psi, mesh, None), n, [None] + [ops[k].ravel() for k in names])
    F = [to_fourier(p, mesh) for p in painted]
    del painted
    F[0][0, 0, 0] = 0                                  # delta_Z: remove the mean
    fixed = None if a.real_no_dz else F[0]
    O = orthogonalize(F[1:], mesh, two_d=False)
    del F
    dg = galaxy_field(xyz[0], xyz[1], xyz[2], mesh)
    perr, ptrue, pmodel, r, betas = fit_transfer(dg, O, fixed, mesh, two_d=False)
    del fixed
    k = mesh.kmean
    save_result(out + "_real.npy", [k, perr, ptrue, pmodel, r] + betas + [nbar])
    log("real space done -> %s_real.npy" % out)
    if a.save_fields:
        save_fields(os.path.join(a.save_fields, "shifted_fields_real_N_%d_c_0.npy" % n), O, mesh)
    del O, dg

    # -------------------------------------------------- redshift space
    perr_s = None
    if rsd:
        names = ["d1", "d2", "G2", "G2par", "d3"]
        painted = cic_paint(shifted_positions(psi, mesh, a.f), n,
                            [None] + [ops[k].ravel() for k in names])
        F = [to_fourier(p, mesh) for p in painted]
        del painted
        F[0][0, 0, 0] = 0                                  # dz: remove the mean
        dz, d1, d2, g2, g2par, d3 = F
        fixed = dz - np.float32(3.0 / 7.0 * a.f) * g2par
        O = orthogonalize([d1, d2, g2, d3], mesh, two_d=True)
        dgs = galaxy_field(xyz[0], xyz[1], zrsd, mesh)
        perr_s, ptrue_s, pmodel_s, r_s, betas_s = fit_transfer(dgs, O, fixed, mesh, two_d=True)
        mu_c = 0.5 * (mesh.mu_edges[1:] + mesh.mu_edges[:-1])
        save_result(out + "_rsd.npy", [k, mu_c, perr_s, ptrue_s, pmodel_s, r_s] + betas_s + [nbar])
        log("redshift space done -> %s_rsd.npy" % out)
        if a.save_fields:
            save_fields(os.path.join(a.save_fields, "shifted_fields_redshift_axisrsd_2_N_%d_c_0.npy" % n),
                        [dz, d1, O[1], O[2], g2par, O[3]], mesh)

    # ------------------------------------------------------- summary
    print("\nmean nbar*P_err per k window:  real" + ("".join(
        " | mu %.2f-%.2f" % (mesh.mu_edges[j], mesh.mu_edges[j + 1]) for j in range(mesh.nmu)) if rsd else ""))
    for lo, hi in ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 0.8)):
        s = (k >= lo) & (k < hi)
        if not s.any():
            continue
        line = "  k %.2f-%.2f   %6.3f" % (lo, hi, nbar * np.nanmean(perr[s]))
        if rsd:
            line += "".join("        %6.3f" % (nbar * np.nanmean(perr_s[s, j])) for j in range(mesh.nmu))
        print(line)
    if not a.no_plot:
        model = "real space without $\\delta_Z$" if a.real_no_dz else "Eq. 16 model"
        make_plots(out, mesh.knyq, a.title or "%s,  $N_{\\rm mesh}=%d$, $L=%g\\,h^{-1}$Mpc  (%s)"
                   % (os.path.basename(out), n, a.box, model))
    log("done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Field-level P_err from an initial-condition grid and a galaxy catalog.

    python perr_from_ic.py --ic ic_delta_256.npy --catalog galaxies.npy --z-out 0.5 --out run1

The defaults describe the simulation this was set up for (from its header):
box 338.3 Mpc/h, ICs at z = 49, flat LCDM with Omega0 = 0.30964144 and
OmegaLambda = 0.6902672. Its HubbleParam (0.6766) and OmegaBaryon
(0.04897468) do not enter. For any other run, pass --box, --z-ic, --omega-m
and --omega-r.

Inputs
  --ic       .npy, an N^3 grid of the linear density contrast at --z-ic
             (default 49). The grid IS the mesh: everything (operators,
             galaxies, P(k)) is done at this N. --nmesh M (< N) first
             truncates it in Fourier space to M^3.
  --box      box side in Mpc/h (default 338.3). The catalog's extent is
             checked against it.
  --z-out    the catalog's redshift, required. The IC grid is multiplied by
             D(z_out)/D(z_ic), and redshift space uses f = dlnD/dlna at z_out.
             Both come from the linear growth equation for a flat universe
             with --omega-m (default 0.30964144), --omega-r (default 9.136e-5)
             and Omega_Lambda = 1 - Omega_m - Omega_r.
             The default Omega_r is the header's 1 - Omega0 - OmegaLambda,
             which is exactly photons plus massless neutrinos. If the N-body
             code leaves radiation out of H(a) (Gadget and N-GenIC do), pass
             --omega-r 0: D(z_out)/D(49) is then 0.94% higher.
             --omega-nu: massive neutrinos within Omega_m, kept smooth (in
             H(a), not in the growth source).
             --growth / --f override the computed values (--growth 1 if the
             grid is already at z_out).
             Cross-check: AbacusSummit c000 (--z-ic 99 --omega-m 0.315192
             --omega-nu 0.0014197 --omega-r 0) gives D(0.5)/D(99) = 59.9116 and
             f = 0.75907; Abacus's own tables give 59.911485 and 0.759070.
  --catalog  .npy galaxy catalog, positions in Mpc/h. Either a structured
             array with x, y, z and z_rsd (or zrsd), or an (Ngal, 3),
             (Ngal, 4: x y z z_rsd) or (Ngal, 6) array.
             Six columns are either x y z xrsd yrsd zrsd (positions with RSD
             along each axis; zrsd is used) or x y z vx vy vz. Which one is
             detected from the data: columns 4-6 sitting within a few Mpc/h
             of columns 1-3 are RSD positions. --cols6 rsd|vel overrides.
             A catalog with vz (peculiar velocity in km/s) but no z_rsd gets
             z_rsd = z + vz (1 + z_out) / (100 E(z_out)).
             The shift z_rsd - z is checked and logged. A real RSD shift has
             no mean and does not track z, so a shift with rms < 0.1 Mpc/h, a
             mean above 0.2 rms or a correlation with z above 0.9 (signs of a
             wrong column or wrong units) triggers a WARNING, repeated at the
             end of the run.
  --los      line(s) of sight for redshift space: z (default), x, y, any
             several, or all. For x or y the problem is rotated so that axis
             becomes the last one (IC grid transposed, catalog columns
             permuted), so everything else is unchanged. Each needs the
             catalog's redshift-space coordinate along that axis: an xrsd /
             yrsd column (6 columns), an x_rsd / xrsd ... field, or vx / vy
             velocities. A 4-column catalog has z only.
             Positions in [0, L) are used as they are. Positions centred on 0
             (any coordinate < 0, as in AbacusHOD catalogs) are shifted by
             +L/2, as lrgs.py does. --no-centre-shift forces [0, L).

Outputs (<out> from --out)
  <out>_real.npy  [k, P_err, P_true, P_model, r, beta1, beta2, betaG2, beta3, nbar]
  <out>_rsd.npy   [k, mu_centers, P_err, P_true, P_model, r, beta1, beta2, betaG2,
                   beta3, nbar], each (k, mu) quantity an (nk, Nmu) array. With
                   several lines of sight this is their mean (r from the mean
                   powers), and <out>_rsd_los<a>.npy holds each one.
  Each figure is written as .png and as a vector .pdf with the same name.
  <out>_transfer.png  Fig. 4 of arXiv:2505.03626 style: one row of panels
                  beta_1, beta_2, beta_G2, beta_3, P_err; real space black,
                  mu bins in darkening shades; P_err in h^-3 Mpc^3 with 1/nbar
  <out>_perr.png  P_err blown up, one panel each for real space and every
                  mu bin, with 1/nbar, k_Nyq and k_Nyq/2 marked
The .npy layout is lrgs.py's, so fit_perr.py and compare_perr.py read these
directly, e.g. to compare with the lrgs.py results for the same HOD.
--replot <out> redraws both figures from saved results without recomputing.
--save-fields DIR also writes the shifted fields in the generator's layout
(shifted_fields_real_N_<N>_c_0.npy, shifted_fields_redshift_axisrsd_<a>_N_<N>_c_0.npy
with a = 0, 1, 2 for line of sight x, y, z, in the original orientation).

Method (Schmittfull et al. 2019, 2021; the public Hi-Fi_mocks library)
  1. delta_1(q) = D(z_out)/D(z_ic) * IC (mean removed) at the N^3 grid points q.
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
import math
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


def fit_transfer(dg, ops, fixed, mesh, two_d, pops=None):
    """beta_i per bin for mutually orthogonal ops; P_err, P_true, P_model, r.
    pops, if a list, receives each operator's power P_O = V <|O|^2> per bin (for
    the transfer-function errors of arXiv:1811.10640, Var beta = P_err / (N P_O))."""
    target = dg if fixed is None else dg - fixed
    model = np.zeros_like(dg) if fixed is None else fixed.copy()
    betas = []
    for o in ops:
        oo = mesh.cross(o, o, two_d)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = mesh.cross(target, o, two_d) / oo
        betas.append(b)
        if pops is not None:
            pops.append(mesh.vol * oo)
        model += mesh.apply(b, two_d) * o
    err = dg - model
    V = mesh.vol
    perr = V * mesh.binned(np.abs(err) ** 2, two_d)
    ptrue = V * mesh.binned(np.abs(dg) ** 2, two_d)
    pmodel = V * mesh.binned(np.abs(model) ** 2, two_d)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = V * mesh.cross(dg, model, two_d) / np.sqrt(ptrue * pmodel)
    return perr, ptrue, pmodel, r, betas


# ----------------------------------------------------------- cosmology ---
# Defaults: the simulation header (BoxSize 338.3, Redshift 49, Omega0 0.30964144,
# OmegaLambda 0.6902672). Omega0 + OmegaLambda = 1 - 9.136e-5; the remainder is
# radiation (photons + massless neutrinos for T_CMB 2.7255 K, N_eff 3.046, h 0.6766).
BOX = 338.3
Z_IC = 49.0
OMEGA_M = 0.30964144
OMEGA_R = 1.0 - 0.30964144 - 0.6902672


def hubble_e(z, omega_m, omega_r):
    """E(z) = H(z) / H0 in a flat universe (Omega_Lambda = 1 - omega_m - omega_r)."""
    a = 1.0 / (1.0 + z)
    return math.sqrt(omega_r / a ** 4 + omega_m / a ** 3 + 1.0 - omega_m - omega_r)


def growth(z_ic, z_out, omega_m, omega_nu=0.0, omega_r=0.0, dx=0.002):
    """(D(z_out) / D(z_ic), f(z_out) = dlnD/dlna) in a flat universe.

    Integrates, in x = ln a, with RK4 steps of at most dx,
        D'' + (2 + dlnE/dx) D' - 3/2 (omega_m - omega_nu) a^-3 / E^2 D = 0.
    It starts on the growing mode deep in matter domination (D = a^p, p = 1
    without neutrinos), or, when omega_r > 0, in radiation domination
    (D = 1 + 3/2 f_cb a / a_eq). omega_nu is the massive-neutrino part of
    omega_m and is kept smooth: it adds to H but not to the source. With
    omega_r = 0 this matches AbacusSummit's growth tables to 3e-6."""
    om, onu, orad = float(omega_m), float(omega_nu), float(omega_r)
    ol = 1.0 - om - orad
    src = 1.5 * (om - onu)
    fcb = (om - onu) / om

    def rhs(x, d, v):
        ia = math.exp(-x)
        m, r = om * ia ** 3, orad * ia ** 4
        e2 = r + m + ol
        return v, -(2.0 - (2.0 * r + 1.5 * m) / e2) * v + src * ia ** 3 / e2 * d

    if orad > 0:
        a0 = 1e-4 * orad / om                       # a / a_eq = 1e-4
        d, v = 1.0 + 1.5 * fcb * 1e-4, 1.5 * fcb * 1e-4
    else:
        a0 = 1e-5
        p = (math.sqrt(1.0 + 24.0 * fcb) - 1.0) / 4.0
        d, v = a0 ** p, p * a0 ** p
    x = math.log(a0)
    ends = []
    for xe in (-math.log1p(z_ic), -math.log1p(z_out)):
        n = max(1, int(math.ceil((xe - x) / dx)))
        h = (xe - x) / n
        for i in range(n):
            k1 = rhs(x, d, v)
            k2 = rhs(x + h / 2, d + h / 2 * k1[0], v + h / 2 * k1[1])
            k3 = rhs(x + h / 2, d + h / 2 * k2[0], v + h / 2 * k2[1])
            k4 = rhs(x + h, d + h * k3[0], v + h * k3[1])
            d += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            v += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            x = xe if i == n - 1 else x + h
        ends.append((d, v))
    (d_ic, _), (d_out, v_out) = ends
    return d_out / d_ic, v_out / d_out


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


def _wrap(d, box):
    """Periodic difference in [-L/2, L/2)."""
    return (d + 0.5 * box) % box - 0.5 * box


def cols6_layout(a, box):
    """('rsd' | 'vel', median offset): columns 4-6 of a 6-column catalog are RSD
    positions (x y z xrsd yrsd zrsd) if they sit close to columns 1-3, else
    velocities (x y z vx vy vz)."""
    med = float(np.median(np.abs(_wrap(np.asarray(a[:, 3:6], np.float64) - a[:, 0:3], box))))
    if med < min(0.05 * box, 20.0):
        return "rsd", med
    if med > box / 8.0:
        return "vel", med
    sys.exit("cannot tell whether catalog columns 4-6 are RSD positions (x y z xrsd yrsd zrsd) or velocities "
             "(x y z vx vy vz): their median distance from columns 1-3 is %.1f Mpc/h. Pass --cols6 rsd or "
             "--cols6 vel." % med)


AXES = "xyz"
LOS_PERM = {0: (1, 2, 0), 1: (2, 0, 1), 2: (0, 1, 2)}    # axis order with the line of sight last (cyclic)


def load_catalog(path, box, force_corner, vz_to_dist, cols6="auto", axes=(2,)):
    """Positions wrapped into [0, L), for the line-of-sight axes asked for:
    ([x, y, z], {axis: redshift-space coordinate along it, or None}, {axis: warning or None}).
    vz_to_dist [Mpc/h per km/s] turns velocities into redshift-space coordinates when the
    catalog has none (None: not available)."""
    c = np.load(path)
    red = dict((ax, None) for ax in axes)
    vel = dict((ax, None) for ax in axes)
    if c.dtype.names:
        missing = [q for q in AXES if q not in c.dtype.names]
        if missing:
            sys.exit("catalog %s has no %s (fields: %s)" % (path, ", ".join(missing), ", ".join(c.dtype.names)))
        cols = [c["x"], c["y"], c["z"]]
        for ax in axes:
            q = AXES[ax]
            for name in (q + "_rsd", q + "rsd"):
                if name in c.dtype.names:
                    red[ax] = c[name]
                    log("redshift-space %s: field '%s'" % (q, name))
                    break
            if red[ax] is None and "v" + q in c.dtype.names:
                vel[ax] = c["v" + q]
    else:
        a = np.asarray(c)
        if a.ndim != 2 or a.shape[1] not in (3, 4, 6):
            sys.exit("catalog must be a structured array (x, y, z and z_rsd or vz) or an (Ngal, 3|4|6) array "
                     "with columns x y z | x y z z_rsd | x y z xrsd yrsd zrsd or x y z vx vy vz; got shape %s"
                     % (a.shape,))
        cols = [a[:, 0], a[:, 1], a[:, 2]]
        if a.shape[1] == 4 and 2 in axes:
            red[2] = a[:, 3]
        elif a.shape[1] == 6:
            kind, med = (cols6, None) if cols6 != "auto" else cols6_layout(a, box)
            how = "--cols6" if med is None else "detected: median |col 4-6 - col 1-3| = %.2f Mpc/h" % med
            for ax in axes:
                (red if kind == "rsd" else vel)[ax] = a[:, 3 + ax]
            log("6 columns read as %s (%s)" % ("x y z xrsd yrsd zrsd" if kind == "rsd" else "x y z vx vy vz", how))
    if len(cols[0]) == 0:
        sys.exit("catalog %s is empty" % path)
    # centred on 0 if any coordinate is clearly negative; x, y, z must then fit
    # [-L/2, L/2], otherwise [0, L] (z_rsd may spill over; it wraps)
    lo = min(float(np.min(x)) for x in cols)
    hi = max(float(np.max(x)) for x in cols)
    centred = lo < -0.01 * box and not force_corner
    lo_ok, hi_ok = (-0.5 * box, 0.5 * box) if centred else (0.0, box)
    tol = 1e-3 * box
    if lo < lo_ok - tol or hi > hi_ok + tol:
        sys.exit("catalog positions span [%.1f, %.1f] Mpc/h, outside %s for --box %g: wrong --box?%s"
                 % (lo, hi, "[-L/2, L/2] (centred on 0, as some are < 0)" if centred else "[0, L]", box,
                    " Or drop --no-centre-shift." if force_corner and lo < 0 else ""))
    log("catalog positions %s" % ("centred on 0: shifted by +L/2" if centred else "in [0, L): used as they are"))
    span = hi - lo
    if len(cols[0]) > 1000 and span < 0.75 * box:
        log("WARNING: catalog spans only %.0f Mpc/h but --box is %g: check --box" % (span, box))
    # A real RSD shift has (statistically) zero mean. Its correlation with position is
    # small but not zero: box-scale modes carry much of the displacement, and one
    # fundamental mode alone can correlate up to +-0.78 with the coordinate (mock,
    # 338 Mpc/h: -0.11 and +0.17). A shift built from the position itself (a wrong
    # column) correlates at ~1, hence the 0.9 threshold. Wrong units shrink the rms.
    warnings = dict((ax, None) for ax in axes)
    for ax in axes:
        q = AXES[ax]
        if red[ax] is None and vel[ax] is not None:
            if vz_to_dist is None:
                log("catalog has v%s but no %s_rsd: pass --z-out to convert v%s" % (q, q, q))
            else:
                red[ax] = np.asarray(cols[ax], np.float64) + vz_to_dist * np.asarray(vel[ax], np.float64)
                log("%s_rsd = %s + v%s * %.6g Mpc/h per km/s = (1 + z_out) / (100 E(z_out))" % (q, q, q, vz_to_dist))
        if red[ax] is None:
            continue
        p64 = np.asarray(cols[ax], np.float64)
        sh = _wrap(np.asarray(red[ax], np.float64) - p64, box)
        rms, mean = float(sh.std()), float(sh.mean())
        corr = float(np.corrcoef(sh, p64)[0, 1]) if rms > 0 else 0.0
        log("RSD shift %s_rsd - %s: rms %.3f Mpc/h, mean %+.3f Mpc/h, correlation with %s %+.3f"
            % (q, q, rms, mean, q, corr))
        if rms < 0.1:
            warnings[ax] = ("%s_rsd - %s has rms %.3f Mpc/h: the redshift-space positions carry (almost) no RSD"
                            % (q, q, rms))
        elif abs(mean) > 0.2 * rms or abs(corr) > 0.9:
            warnings[ax] = ("%s_rsd - %s has mean %+.2f Mpc/h and correlation %+.2f with %s (rms %.2f Mpc/h). A real "
                            "RSD shift has neither: %s_rsd is probably built from the wrong column or units"
                            % (q, q, mean, corr, q, rms, q))
        if warnings[ax]:
            log("WARNING: " + warnings[ax])
    s = 0.5 * box if centred else 0.0
    xyz = [np.mod(np.asarray(x, np.float64) + s, box) for x in cols]
    for ax in axes:
        if red[ax] is not None:
            red[ax] = np.mod(np.asarray(red[ax], np.float64) + s, box)
    return xyz, red, warnings


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


def save_fields(path, fields, mesh, perm=None):
    """perm: the axis order the fields were computed in (a rotated line of sight);
    they are transposed back to the original orientation."""
    n = mesh.n
    arr = np.empty((len(fields), n, n, n), np.float32)
    inv = None if perm is None else tuple(int(i) for i in np.argsort(perm))
    for i, F in enumerate(fields):     # back to the painted (compensated) real grid
        g = irfftn(F * np.float32(n ** 3), n)
        arr[i] = g if inv is None else g.transpose(inv)
    np.save(path, arr)


def mean_rsd(items):
    """Mean of per-line-of-sight redshift-space results (same layout); r is recomputed
    from the mean cross and auto powers rather than averaged."""
    pe, pt, pm = (np.mean([it[i] for it in items], axis=0) for i in (2, 3, 4))
    cross = np.mean([it[5] * np.sqrt(it[3] * it[4]) for it in items], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cross / np.sqrt(pt * pm)
    betas = [np.mean([it[6 + j] for it in items], axis=0) for j in range(4)]
    return [items[0][0], items[0][1], pe, pt, pm, r] + betas + [items[0][-1]]


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


def save_figure(fig, fn, dpi):
    """Save the figure as fn (PNG) and as a vector PDF with the same name."""
    fig.savefig(fn, dpi=dpi)
    pdf = os.path.splitext(fn)[0] + ".pdf"
    fig.savefig(pdf)
    log("wrote %s and %s" % (fn, os.path.basename(pdf)))


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
    save_figure(fig, fn, 150)
    plt.close(fig)


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
        for kk, ls, txt in (() if knyq is None else
                            ((knyq, "--", r"$k_{\rm Nyq}$"), (knyq / 2, ":", r"$k_{\rm Nyq}/2$"))):
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
    save_figure(fig, fn, 150)
    plt.close(fig)


def make_plots(prefix, knyq, title):
    R = load_results(prefix)
    plot_transfer(prefix + "_transfer.png", R, title)
    plot_perr_panels(prefix + "_perr.png", R, knyq, title)


# ---------------------------------------------------------------- main ---
def main():
    global WORKERS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--ic", help="N^3 .npy linear density grid at --z-ic")
    ap.add_argument("--catalog", help="galaxy catalog .npy")
    ap.add_argument("--box", type=float, default=BOX, help="box side [Mpc/h] (default %g)" % BOX)
    ap.add_argument("--z-out", type=float, default=None, help="redshift of the catalog (required)")
    ap.add_argument("--z-ic", type=float, default=Z_IC, help="redshift of the IC grid (default %g)" % Z_IC)
    ap.add_argument("--omega-m", type=float, default=OMEGA_M,
                    help="Omega_m today: all matter in H(a) (default %g)" % OMEGA_M)
    ap.add_argument("--omega-r", type=float, default=OMEGA_R,
                    help="radiation in H(a) (default %.4g; 0 if the N-body code has none, e.g. Gadget)"
                         % OMEGA_R)
    ap.add_argument("--omega-nu", type=float, default=0.0,
                    help="massive neutrinos within --omega-m, kept smooth (default 0)")
    ap.add_argument("--growth", type=float, default=None,
                    help="override D(z_out)/D(z_ic), e.g. 1 if the grid is already at z_out")
    ap.add_argument("--f", type=float, default=None, help="override the growth rate f at z_out")
    ap.add_argument("--nmesh", type=int, default=None, help="Fourier-truncate the IC to this N first")
    ap.add_argument("--nmu", type=int, default=3, help="mu bins (default 3)")
    ap.add_argument("--d3-kmax", type=float, default=0.5, help="cut on delta_1 before cubing [h/Mpc]")
    ap.add_argument("--compensation", choices=["shotnoise", "sinc"], default="shotnoise")
    ap.add_argument("--no-centre-shift", action="store_true",
                    help="positions are in [0, L) (default: detected; centred on 0 if any is < 0)")
    ap.add_argument("--cols6", choices=["auto", "rsd", "vel"], default="auto",
                    help="6-column catalogs: rsd = x y z xrsd yrsd zrsd, vel = x y z vx vy vz "
                         "(default: detected from the data)")
    ap.add_argument("--no-rsd", action="store_true", help="real space only")
    ap.add_argument("--los", nargs="+", choices=["x", "y", "z", "all"], default=["z"],
                    help="line(s) of sight in redshift space (default z; all = x y z). With several, "
                         "<out>_rsd.npy is their mean and <out>_rsd_los<a>.npy each one")
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

    if a.box <= 0:
        ap.error("--box must be positive (Mpc/h)")
    if a.replot:                              # figures only, from saved results
        R = load_results(a.replot)
        nmesh = a.nmesh or 2 * (len(R["k"]) + 1)       # lrgs-style bins: nk = N/2 - 1
        make_plots(a.replot, np.pi * nmesh / a.box, a.title or os.path.basename(a.replot))
        return
    if not (a.ic and a.catalog):
        ap.error("--ic and --catalog are required (or use --replot PREFIX)")

    # D(z_out)/D(z_ic) and f(z_out); vz -> z_rsd needs E(z_out)
    if a.z_out is None and (a.growth is None or (a.f is None and not a.no_rsd)):
        ap.error("--z-out is required: the catalog's redshift (the IC grid at --z-ic %g is grown to it)"
                 % a.z_ic)
    D = f = vz_to_dist = None
    if a.z_out is not None:
        if not -1 < a.z_out <= a.z_ic:
            ap.error("need -1 < --z-out <= --z-ic, got z_out %g, z_ic %g" % (a.z_out, a.z_ic))
        if not (0 < a.omega_m and 0 <= a.omega_nu < a.omega_m and a.omega_r >= 0
                and a.omega_m + a.omega_r <= 1):
            ap.error("need 0 < Omega_m, 0 <= Omega_nu < Omega_m, Omega_r >= 0, Omega_m + Omega_r <= 1")
        D, f = growth(a.z_ic, a.z_out, a.omega_m, a.omega_nu, a.omega_r)
        vz_to_dist = (1.0 + a.z_out) / (100.0 * hubble_e(a.z_out, a.omega_m, a.omega_r))
        log("flat LCDM: Omega_m %g (smooth nu %g), Omega_r %.4g, Omega_Lambda %.8g"
            % (a.omega_m, a.omega_nu, a.omega_r, 1.0 - a.omega_m - a.omega_r))
        log("D(z=%g)/D(z=%g) = %.6f, f(z=%g) = %.6f, E(z=%g) = %.6f"
            % (a.z_out, a.z_ic, D, a.z_out, f, a.z_out, hubble_e(a.z_out, a.omega_m, a.omega_r)))
    if a.growth is not None:
        log("using --growth %g%s" % (a.growth, "" if D is None else " instead of %.6f" % D))
        D = a.growth
    if a.f is not None:
        log("using --f %g%s" % (a.f, "" if f is None else " instead of %.6f" % f))
        f = a.f

    delta = np.load(a.ic).astype(np.float32)
    if delta.ndim != 3 or len(set(delta.shape)) != 1 or delta.shape[0] % 2:
        sys.exit("--ic must be a cubic N^3 grid with even N, got %s" % (delta.shape,))
    if a.nmesh and a.nmesh < delta.shape[0]:
        log("truncating the IC from %d^3 to %d^3 in Fourier space" % (delta.shape[0], a.nmesh))
        delta = truncate_ic(delta, a.nmesh)
    n = delta.shape[0]
    delta *= np.float32(D)
    delta -= delta.mean()
    req = set(AXES) if "all" in a.los else set(a.los)
    los_axes = [ax for ax in (2, 0, 1) if AXES[ax] in req]     # z first: it reuses the real-space operators
    lostag = "" if a.no_rsd or los_axes == [2] else "_los" + "".join(AXES[ax] for ax in sorted(los_axes))
    out = a.out or "%s_N%d%s" % (os.path.splitext(os.path.basename(a.catalog))[0], n, lostag)
    multi = len(los_axes) > 1 and not a.no_rsd
    targets = [out + "_real.npy"]
    if not a.no_rsd:
        targets += [out + "_rsd.npy"] + ([out + "_rsd_los%s.npy" % AXES[ax] for ax in los_axes] if multi else [])
    if not a.overwrite:
        clash = [t for t in targets if os.path.exists(t)]
        if clash:
            sys.exit("refusing to overwrite %s (pass --overwrite)" % ", ".join(clash))
    if a.save_fields:
        os.makedirs(a.save_fields, exist_ok=True)

    log("IC %s: %d^3, box %g Mpc/h, times D = %.6f, std(delta_1) = %.4f"
        % (a.ic, n, a.box, D, float(delta.std())))
    mesh = Mesh(n, a.box, a.nmu, a.compensation)
    xyz, red, rsd_warn = load_catalog(a.catalog, a.box, a.no_centre_shift, vz_to_dist, a.cols6,
                                      [] if a.no_rsd else los_axes)
    ngal = len(xyz[0])
    nbar = ngal / a.box ** 3
    rsd_axes = [] if a.no_rsd else [ax for ax in los_axes if red[ax] is not None]
    lacking = [] if a.no_rsd else [AXES[ax] for ax in los_axes if red[ax] is None]
    if lacking and los_axes == [2]:
        log("catalog has no z_rsd: real space only")
    elif lacking:
        sys.exit("catalog has no redshift-space coordinate for line of sight %s (needs %s)"
                 % (", ".join(lacking), "; ".join("a %srsd column, a %s_rsd or %srsd field, or v%s" % (q, q, q, q)
                                                  for q in lacking)))
    log("catalog %s: %d galaxies, nbar = %.4e (h/Mpc)^3" % (a.catalog, ngal, nbar))

    ops, psi = lagrangian_fields(delta, mesh, a.d3_kmax, 2 in rsd_axes)
    if all(ax == 2 for ax in rsd_axes):
        delta = None                                   # not needed again
    log("Lagrangian operators and displacement done")

    # ------------------------------------------------------- real space
    names = ["d1", "d2", "G2", "d3"]
    painted = cic_paint(shifted_positions(psi, mesh, None), n, [None] + [ops[nm].ravel() for nm in names])
    F = [to_fourier(p, mesh) for p in painted]
    del painted
    F[0][0, 0, 0] = 0                                  # delta_Z: remove the mean
    fixed = None if a.real_no_dz else F[0]
    O = orthogonalize(F[1:], mesh, two_d=False)
    del F
    dg = galaxy_field(xyz[0], xyz[1], xyz[2], mesh)
    pops = []
    perr, ptrue, pmodel, r, betas = fit_transfer(dg, O, fixed, mesh, two_d=False, pops=pops)
    del fixed
    k = mesh.kmean
    save_result(out + "_real.npy", [k, perr, ptrue, pmodel, r] + betas + [nbar])
    save_result(out + "_real_ops.npy", [k, mesh.wsum1.copy()] + pops)
    log("real space done -> %s_real.npy (+ _real_ops.npy: modes and operator powers per bin)" % out)
    if a.save_fields:
        save_fields(os.path.join(a.save_fields, "shifted_fields_real_N_%d_c_0.npy" % n), O, mesh)
    del O, dg

    # -------------------------------------------------- redshift space
    mu_c = 0.5 * (mesh.mu_edges[1:] + mesh.mu_edges[:-1])
    per_los, per_los_ops = [], []
    for ax in rsd_axes:                      # z first: it reuses the operators built above
        perm = LOS_PERM[ax]
        if ax != 2:                          # rotate so this line of sight is the last axis
            ops = psi = None
            ops, psi = lagrangian_fields(np.ascontiguousarray(delta.transpose(perm)), mesh, a.d3_kmax, True)
            log("line of sight %s: operators rebuilt on the rotated grid" % AXES[ax])
        names = ["d1", "d2", "G2", "G2par", "d3"]
        painted = cic_paint(shifted_positions(psi, mesh, f), n,
                            [None] + [ops[nm].ravel() for nm in names])
        ops = psi = None
        F = [to_fourier(p, mesh) for p in painted]
        del painted
        F[0][0, 0, 0] = 0                                  # dz: remove the mean
        dz, d1, d2, g2, g2par, d3 = F
        del F
        fixed = dz - np.float32(3.0 / 7.0 * f) * g2par
        O = orthogonalize([d1, d2, g2, d3], mesh, two_d=True)
        pos = list(xyz)
        pos[ax] = red[ax]                                  # redshift-space coordinate along the LOS
        pos = [pos[p] for p in perm]
        dgs = galaxy_field(pos[0], pos[1], pos[2], mesh)
        pops_s = []
        perr_s, ptrue_s, pmodel_s, r_s, betas_s = fit_transfer(dgs, O, fixed, mesh, two_d=True, pops=pops_s)
        item = [k, mu_c, perr_s, ptrue_s, pmodel_s, r_s] + betas_s + [nbar]
        ops_item = [k, mu_c, mesh.wsum2.reshape(mesh.nk, mesh.nmu).copy()] + pops_s
        per_los.append(item)
        per_los_ops.append(ops_item)
        if multi:
            save_result(out + "_rsd_los%s.npy" % AXES[ax], item)
            save_result(out + "_rsd_ops_los%s.npy" % AXES[ax], ops_item)
            log("line of sight %s done -> %s_rsd_los%s.npy" % (AXES[ax], out, AXES[ax]))
        if a.save_fields:
            save_fields(os.path.join(a.save_fields, "shifted_fields_redshift_axisrsd_%d_N_%d_c_0.npy" % (ax, n)),
                        [dz, d1, O[1], O[2], g2par, O[3]], mesh, None if ax == 2 else perm)
        del dz, d1, d2, g2, g2par, d3, fixed, O, dgs, pos
    perr_s = None
    if per_los:
        res = per_los[0] if len(per_los) == 1 else mean_rsd(per_los)
        save_result(out + "_rsd.npy", res)
        # operator powers for the LOS mean: averaged too (same modes in every LOS)
        ops_res = per_los_ops[0] if len(per_los_ops) == 1 else (
            per_los_ops[0][:3] + [np.mean([it[i] for it in per_los_ops], axis=0) for i in range(3, 7)])
        save_result(out + "_rsd_ops.npy", ops_res)
        perr_s = res[2]
        log("redshift space done -> %s_rsd.npy%s" % (out, "" if len(per_los) == 1 else
            " (mean over lines of sight %s)" % ", ".join(AXES[ax] for ax in sorted(rsd_axes))))
    rsd = perr_s is not None

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
        if rsd_axes and rsd_axes != [2]:
            model += ", LOS %s%s" % ("mean of " if len(rsd_axes) > 1 else "",
                                     ", ".join(AXES[ax] for ax in sorted(rsd_axes)))
        zlab = "" if a.z_out is None else "$z=%g$, " % a.z_out
        make_plots(out, mesh.knyq, a.title or "%s,  %s$N_{\\rm mesh}=%d$, $L=%g\\,h^{-1}$Mpc  (%s)"
                   % (os.path.basename(out), zlab, n, a.box, model))
    for ax in rsd_axes:
        if rsd_warn[ax]:
            log("WARNING (repeated): %s. Do not trust the redshift-space results." % rsd_warn[ax])
    log("done")


if __name__ == "__main__":
    main()

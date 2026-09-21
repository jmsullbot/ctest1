#!/usr/bin/env python3
"""
Plot n̄·P_err(k, μ) from the field-level extraction results and fit

    n̄ P_err(k, μ) = 1 − D · W(kR) · exp(−k² μ² ℓ_F²)

with three parameters (D, ℓ_F, R), jointly to the real-space curve (μ = 0)
and the RSD μ-bins, for k ≤ kmax.

Input files are the pickled object arrays written by the extraction code:
    real : [k, P_err, P_true, P_model, r, ..., nbar]
    rsd  : [k, mu_centers, P_err(k,μ), P_true, P_model, r, ..., nbar]

Modelling choices (all overridable):
  * W(kR): 'gauss'  → exp(−k²R²/2)        (default)
           'tophat' → 3 (sin x − x cos x)/x³, x = kR   (Fourier-space top-hat)
  * μ-bins: the exp(−k²μ²ℓ²) factor is averaged analytically over each μ bin
    (edges reconstructed from the centers), not evaluated at the center.
  * weights: no per-bin errors are stored, so bins are weighted by the mode
    count, σ_i ∝ 1/k_i (N_modes ∝ k² Δk).  χ² is therefore only relative.

Usage
-----
    python fit_perr.py real.npy rsd.npy --kmax 0.4 --out perr_fit.png
"""

import argparse

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def window(k, R, kind):
    x = k * R
    if kind == "gauss":
        return np.exp(-0.5 * x**2)
    if kind == "tophat":
        x = np.where(x < 1e-6, 1e-6, x)
        return 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    raise ValueError(kind)


def mu_average(k, ell, mu1, mu2):
    """<exp(−k²μ²ℓ²)> averaged uniformly over μ ∈ [mu1, mu2]."""
    a = (k * ell) ** 2
    small = a < 1e-12
    a_safe = np.where(small, 1.0, a)
    sa = np.sqrt(a_safe)
    avg = 0.5 * np.sqrt(np.pi) / sa * (erf(sa * mu2) - erf(sa * mu1)) / (mu2 - mu1)
    return np.where(small, 1.0, avg)


def model(k, D, ell, R, mu1, mu2, kind):
    """n̄P_err for a μ bin [mu1, mu2]; mu1 = mu2 = 0 gives real space."""
    if mu1 == mu2:                       # real space, μ = 0
        rsd = np.ones_like(k)
    else:
        rsd = mu_average(k, ell, mu1, mu2)
    return 1.0 - D * window(k, R, kind) * rsd


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load(real_fn, rsd_fn):
    a = np.load(real_fn, allow_pickle=True)
    b = np.load(rsd_fn, allow_pickle=True)
    k, perr_real, nbar = np.asarray(a[0], float), np.asarray(a[1], float), float(a[-1])
    k2, mu_c, perr_rsd = np.asarray(b[0], float), np.asarray(b[1], float), np.asarray(b[2], float)
    assert np.allclose(k, k2), "k grids differ between real and RSD files"
    assert np.isclose(nbar, float(b[-1])), "nbar differs between files"
    # μ-bin edges from centers (uniform bins assumed)
    dmu = np.diff(mu_c).mean() if len(mu_c) > 1 else 1.0
    edges = np.concatenate([[mu_c[0] - dmu / 2], mu_c + dmu / 2])
    edges = np.clip(edges, 0.0, 1.0)
    return k, nbar * perr_real, mu_c, edges, nbar * perr_rsd, nbar


def fit(k, y_real, edges, y_rsd, kmax, kind):
    """Joint weighted least squares over real space + all μ bins."""
    xs, ys, ws, bins = [], [], [], []
    m = k <= kmax
    xs.append(k[m]); ys.append(y_real[m]); ws.append(k[m]); bins.append((0.0, 0.0))
    for j in range(y_rsd.shape[1]):
        good = m & np.isfinite(y_rsd[:, j])
        xs.append(k[good]); ys.append(y_rsd[good, j]); ws.append(k[good]); bins.append((edges[j], edges[j + 1]))
    lens = [len(x) for x in xs]
    X, Y, W = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    sigma = 1.0 / W                      # σ ∝ 1/k  (mode counting)

    def f(_, D, ell, R):
        out, i = [], 0
        for n, (m1, m2) in zip(lens, bins):
            out.append(model(X[i:i + n], D, ell, R, m1, m2, kind)); i += n
        return np.concatenate(out)

    p0 = [0.3, 3.0, 3.0]
    popt, pcov = curve_fit(f, X, Y, p0=p0, sigma=sigma, absolute_sigma=False,
                           bounds=([0, 0, 0], [1, 50, 50]), maxfev=20000)
    resid = Y - f(X, *popt)
    rms = float(np.sqrt(np.mean(resid**2)))                    # unweighted, in n̄P units
    wrms = float(np.sqrt(np.sum((resid / sigma)**2) / np.sum(1 / sigma**2)))   # mode-weighted
    return popt, np.sqrt(np.diag(pcov)), rms, wrms, len(Y)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("real"); ap.add_argument("rsd")
    ap.add_argument("--kmax", type=float, default=0.4)
    ap.add_argument("--window", choices=["gauss", "tophat"], default="gauss")
    ap.add_argument("--out", default="perr_fit.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--kplot", type=float, default=None,
                    help="max k to DISPLAY (default: kmax). Data beyond kmax are shown; the "
                         "kmax fit is extrapolated as a dashed line")
    ap.add_argument("--box", type=float, default=None,
                    help="box side [Mpc/h]; with --nmesh draws k_Nyq and k_Nyq/2")
    ap.add_argument("--nmesh", type=int, default=256)
    args = ap.parse_args()

    k, y_real, mu_c, edges, y_rsd, nbar = load(args.real, args.rsd)
    nmu = y_rsd.shape[1]

    results = {}
    for kind in ("gauss", "tophat"):
        results[kind] = fit(k, y_real, edges, y_rsd, args.kmax, kind)
    popt, perr, rms, wrms, npts = results[args.window]

    print(f"nbar = {nbar:.4e} (Mpc/h)^-3   kmax = {args.kmax}   points = {npts}   (3 parameters)")
    print(f"μ bins: " + ", ".join(f"[{edges[j]:.3f},{edges[j+1]:.3f}]" for j in range(nmu)))
    print("weights: σ_i ∝ 1/k_i (mode counting); errors are from the covariance, scaled by the residual")
    print()
    for kind, (p, e, r, wr, _) in results.items():
        tag = "<- plotted" if kind == args.window else ""
        print(f"W = {kind:6s}  D = {p[0]:.4f} ± {e[0]:.4f}   ℓ_F = {p[1]:.3f} ± {e[1]:.3f} Mpc/h   "
              f"R = {p[2]:.3f} ± {e[2]:.3f} Mpc/h   rms resid = {r:.4f} (mode-weighted {wr:.4f})  {tag}")

    # ---------------- plot ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kplot = args.kplot or args.kmax
    kk  = np.geomspace(k.min(), args.kmax, 400)          # fitted range: solid
    kx  = np.geomspace(args.kmax, max(kplot, args.kmax * 1.001), 400)   # extrapolation: dashed
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    blues = plt.cm.Blues(np.linspace(0.45, 0.95, nmu))
    show = k <= kplot

    def curve(c, m1, m2):
        ax.plot(kk, model(kk, *popt, m1, m2, args.window), "-", color=c, lw=1.6)
        if kplot > args.kmax:
            ax.plot(kx, model(kx, *popt, m1, m2, args.window), "--", color=c, lw=1.2, alpha=0.8)

    ax.plot(k[show], y_real[show], "o", ms=3.5, color="k", label="real space")
    curve("k", 0.0, 0.0)
    for j in range(nmu):
        ax.plot(k[show], y_rsd[show, j], "o", ms=3.5, color=blues[j],
                label=rf"$\mu \in [{edges[j]:.2f},\,{edges[j+1]:.2f}]$")
        curve(blues[j], edges[j], edges[j + 1])
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    if kplot > args.kmax:
        ax.axvline(args.kmax, color="0.4", lw=0.9, ls="-.")
        ax.text(args.kmax * 1.03, 0.02, r"$k_{\max}$ (fit)", color="0.3", fontsize=8.5,
                transform=ax.get_xaxis_transform(), va="bottom")
    if args.box:
        kny = np.pi * args.nmesh / args.box
        if kny <= kplot * 1.2:
            ax.axvline(kny, color="crimson", lw=1, ls="--")
            ax.axvline(kny / 2, color="crimson", lw=0.8, ls=":")
            ax.text(kny * 0.97, 0.98, r"$k_{\rm Nyq}$", color="crimson", ha="right", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=9)
            ax.text(kny / 2 * 0.97, 0.98, r"$k_{\rm Nyq}/2$", color="crimson", ha="right", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$\bar n\, P_{\rm err}(k,\mu)$")
    Wtxt = r"e^{-k^2R^2/2}" if args.window == "gauss" else r"W_{\rm TH}(kR)"
    ax.set_title(args.title or (r"$\bar n P_{\rm err} = 1 - D\,%s\,e^{-k^2\mu^2\ell_F^2}$" % Wtxt), fontsize=11)
    txt = (f"$D = {popt[0]:.3f} \\pm {perr[0]:.3f}$\n"
           f"$\\ell_F = {popt[1]:.2f} \\pm {perr[1]:.2f}\\ h^{{-1}}$Mpc\n"
           f"$R = {popt[2]:.2f} \\pm {perr[2]:.2f}\\ h^{{-1}}$Mpc\n"
           f"$k_{{\\max}} = {args.kmax}$,  $\\bar n = {nbar:.2e}$")
    ax.text(0.03, 0.03, txt, transform=ax.transAxes, va="bottom", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_xlim(k.min() * 0.9, kplot * 1.05)
    ax.set_ylim(min(0.0, np.nanmin(y_rsd[show]) - 0.05), max(1.15, np.nanmax(y_rsd[show]) + 0.05))
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Figures 4 and 5 of arXiv:2505.03626 for several perr_from_ic.py runs, plus a
check that the catalogs' redshift-space positions really carry RSD.

    python plot_fig4.py --run run1 "LAE run1" --run run1b "LAE run1b" \\
        --run run1i "LBG run1i" --run run1ii "LBG run1ii" --z-out 3 --out lae_lbg

--run PREFIX LABEL (repeatable) reads PREFIX_real.npy and PREFIX_rsd.npy (the
perr_from_ic.py / lrgs.py layout). Writes, each as .png and vector .pdf,
  <out>_fig4.png      one row per run: beta_1, beta_2, beta_G2, beta_3, P_err.
                      Real space is black, mu bins are darkening shades, and
                      1/nbar is dashed.
  <out>_fig5.png      cross-correlation coefficient r_cc per run (Fig. 5)
  <out>_rsdcheck.png  does each catalog's z_rsd carry redshift-space distortions?
      left panels: the galaxies' own P(k, mu) / P(k) (redshift over real space,
          shot noise removed), against linear Kaiser for b1 and f(z_out).
          b1 = real-space beta_1 + 1 at low k. A catalog without RSD gives ~1
          in every mu bin at every k; fingers of God push the top mu bin
          below 1 at high k. Below k ~ 0.1 few modes share each (k, mu) bin,
          so single points there scatter by tens of percent; read the trend
          over k >~ 0.1.
      right panel: beta_1(mu) - beta_1(real) at low k against <mu^2>. If the
          catalog's RSD matches the model's f, this is flat at 0. A catalog
          without RSD, analysed with the RSD model, gives slope -f.
f comes from --z-out with perr_from_ic.py's cosmology (pass its --omega-m /
--omega-r if you changed them), or from --f directly. Use the z_out / f the runs
used. --beta1-is-b1: real-space fits without delta_Z (perr_from_ic.py
--real-no-dz, lrgs.py), where beta_1 -> b1 rather than b1 - 1.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perr_from_ic as P  # noqa: E402  (KLABEL, PANELS, growth, cosmology defaults)


def load(prefix):
    a = np.load(prefix + "_real.npy", allow_pickle=True)
    R = {"k": np.asarray(a[0], float), "perr": np.asarray(a[1], float), "ptrue": np.asarray(a[2], float),
         "r": np.asarray(a[4], float), "nbar": float(a[-1]),
         "betas": [np.asarray(x, float) for x in a[5:9]] if len(a) >= 10 else None,
         "mu": None, "perr_s": None, "ptrue_s": None, "r_s": None, "betas_s": None}
    if os.path.exists(prefix + "_rsd.npy"):
        b = np.load(prefix + "_rsd.npy", allow_pickle=True)
        R["mu"] = np.asarray(b[1], float)
        R["perr_s"], R["ptrue_s"], R["r_s"] = (np.asarray(b[i], float) for i in (2, 3, 5))
        if len(b) >= 11 and all(np.shape(x) == R["perr_s"].shape for x in b[6:10]):
            R["betas_s"] = [np.asarray(x, float) for x in b[6:10]]
    return R


def mu_moments(nmu):
    e = np.linspace(0.0, 1.0, nmu + 1)
    d = np.diff(e)
    return (e[1:] ** 3 - e[:-1] ** 3) / (3 * d), (e[1:] ** 5 - e[:-1] ** 5) / (5 * d)


def wmean(x, w):
    """Weighted mean and its standard error from the scatter (NaNs dropped)."""
    ok = np.isfinite(x)
    x, w = x[ok], w[ok]
    if len(x) < 2:
        return (float(x[0]) if len(x) else np.nan), np.nan
    m = np.sum(w * x) / np.sum(w)
    var = np.sum(w * (x - m) ** 2) / np.sum(w)
    neff = np.sum(w) ** 2 / np.sum(w ** 2)
    return m, np.sqrt(var / max(neff - 1, 1))


# ------------------------------------------------------------- figures ---
def fig4(plt, runs, fn):
    n = len(runs)
    fig, axes = plt.subplots(n, 5, figsize=(20, 3.5 * n + 0.5), squeeze=False)
    for i, (label, R) in enumerate(runs):
        k = R["k"]
        nmu = 0 if R["mu"] is None else len(R["mu"])
        for p, (lab, cmap) in enumerate(P.PANELS):
            ax = axes[i, p]
            real = R["perr"] if p == 4 else (R["betas"][p] if R["betas"] else None)
            rsd = R["perr_s"] if p == 4 else (R["betas_s"][p] if R["betas_s"] else None)
            if rsd is not None:
                shades = plt.get_cmap(cmap)(np.linspace(0.35, 0.85, nmu))
                for j in range(nmu):
                    ax.plot(k, rsd[:, j], color=shades[j], lw=1.1, label=r"$\mu = %.2f$" % R["mu"][j])
            if real is not None:
                ax.plot(k, real, color="k", lw=1.2, label="real space", zorder=3)
            if p == 4:
                ax.axhline(1.0 / R["nbar"], color="0.45", ls="--", lw=1.1)
            ax.set_xscale("log")
            ax.set_xlabel(P.KLABEL, fontsize=11)
            lo, hi = ax.get_ylim()                 # headroom for the panel label
            ax.set_ylim(lo, hi + 0.25 * (hi - lo))
            ax.text(0.1, 0.93, lab, transform=ax.transAxes, fontsize=14, va="top")
            ax.tick_params(labelsize=10)
        axes[i, 2].set_title(r"%s   ($\bar n = %.2g\ h^3{\rm Mpc}^{-3}$)" % (label, R["nbar"]), fontsize=13)
    # one legend for the whole figure, in gray: every column uses the same light-to-dark order
    mus = next((R["mu"] for _, R in runs if R["mu"] is not None), [])
    greys = plt.get_cmap("Greys")(np.linspace(0.3, 0.65, len(mus)))    # kept well apart from black
    handles = [plt.Line2D([], [], color=c, lw=2) for c in greys] + [plt.Line2D([], [], color="k", lw=2)]
    labels = [r"$\mu = %.2f$" % m for m in mus] + ["real space"]
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.5 * n + 0.5)))
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=12, frameon=False,
               title="each column: lighter to darker = increasing $\\mu$ (redshift space); black = real space",
               title_fontsize=11, bbox_to_anchor=(0.5, 1.0))
    P.save_figure(fig, fn, 130)
    plt.close(fig)


def fig5(plt, runs, fn):
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.6), squeeze=False)
    for ax, (label, R) in zip(axes[0], runs):
        k = R["k"]
        if R["r_s"] is not None:
            nmu = len(R["mu"])
            shades = plt.get_cmap("Purples")(np.linspace(0.35, 0.85, nmu))
            for j in range(nmu):
                ax.plot(k, R["r_s"][:, j], color=shades[j], lw=1.3, label=r"$\mu = %.2f$" % R["mu"][j])
        ax.plot(k, R["r"], color="k", lw=1.3, label="real space", zorder=3)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(P.KLABEL, fontsize=11)
        ax.set_ylabel(r"$r_{\rm cc}$", fontsize=12)
        ax.set_title(label, fontsize=12)
        ax.legend(loc="lower left", fontsize=9, frameon=False)
        ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    P.save_figure(fig, fn, 130)
    plt.close(fig)


def rsdcheck(plt, runs, fn, f, kcut, beta1_is_b1):
    n = len(runs)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.4 * (n + 1), 4.0), squeeze=False)
    axes = axes[0]
    edges = np.geomspace(0.015, 3.0, 16)
    summary = []
    for ax, (label, R) in zip(axes[:n], runs):
        if R["ptrue_s"] is None:
            ax.text(0.5, 0.5, "no _rsd.npy", transform=ax.transAxes, ha="center")
            continue
        k, nbar, nmu = R["k"], R["nbar"], len(R["mu"])
        m2, m4 = mu_moments(nmu)
        w = k ** 2                                            # ~ number of modes per k bin
        pr = R["ptrue"] - 1.0 / nbar
        ps = R["ptrue_s"] - 1.0 / nbar
        low = k < kcut
        b1 = wmean(R["betas"][0][low], w[low])[0] + (0.0 if beta1_is_b1 else 1.0)
        beta = f / b1
        kais = 1 + 2 * beta * m2 + beta ** 2 * m4
        shades = plt.get_cmap("Purples")(np.linspace(0.45, 0.9, nmu))
        kc = np.sqrt(edges[1:] * edges[:-1])
        for j in range(nmu):
            rat = np.full(len(kc), np.nan)
            for c in range(len(kc)):
                s = (k >= edges[c]) & (k < edges[c + 1]) & np.isfinite(ps[:, j])
                # skip shot-noise-dominated bins: the ratio of two tiny numbers is noise
                if s.sum() and np.sum(w[s] * nbar * pr[s]) / np.sum(w[s]) > 0.1:
                    rat[c] = np.sum(w[s] * ps[s, j]) / np.sum(w[s] * pr[s])
            ax.plot(kc, rat, "o-", color=shades[j], ms=4, lw=1.2, label=r"$\mu = %.2f$" % R["mu"][j])
            ax.axhline(kais[j], color=shades[j], ls="--", lw=1.1)
        ax.axhline(1.0, color="0.5", lw=1.0)
        ax.plot([], [], color="0.3", ls="--", lw=1.1, label="linear Kaiser")
        ax.plot([], [], color="0.5", lw=1.0, label="no RSD")
        ax.set_xscale("log")
        ax.set_xlim(0.015, 3.0)
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_ylim(0.0, max(2.0, kais.max() + 0.25))
        ax.set_xlabel(P.KLABEL, fontsize=11)
        ax.set_ylabel(r"$P_{\rm gal}(k,\mu)\,/\,P_{\rm gal}^{\rm real}(k)$  (shot noise removed)", fontsize=10)
        ax.set_title(r"%s  ($b_1 = %.2f$, $f = %.2f$)" % (label, b1, f), fontsize=11)
        ax.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=2)
        ax.grid(alpha=0.25, which="both")
        # f_eff from the top mu bin's ratio on 0.05 <= k < 0.2: enough modes, little FoG yet
        s = (k >= 0.05) & (k < 0.2) & np.isfinite(ps[:, -1])
        rtop = np.sum(w[s] * ps[s, -1]) / np.sum(w[s] * pr[s])
        disc = m2[-1] ** 2 - m4[-1] * (1 - rtop)
        f_kaiser = b1 * (-m2[-1] + np.sqrt(disc)) / m4[-1] if disc >= 0 else np.nan
        summary.append((label, R, b1, rtop, kais[-1], f_kaiser))

    # beta_1(mu) - beta_1(real) at low k vs <mu^2>
    ax = axes[n]
    marks = ["o", "s", "^", "D", "v", "P", "X", "*"]
    cols = ["#1f77b4", "#ff7f0e"]                        # LAE-like / LBG-like families
    for i, (label, R, b1, rtop, ktop, f_kaiser) in enumerate(summary):
        if R["betas_s"] is None:
            continue
        k = R["k"]
        low = k < kcut
        w = k ** 2
        nmu = len(R["mu"])
        m2, _ = mu_moments(nmu)
        b_real = wmean(R["betas"][0][low], w[low])[0]
        y, e = zip(*[wmean(R["betas_s"][0][low, j], w[low]) for j in range(nmu)])
        y = np.array(y) - b_real
        slope = np.polyfit(m2, y, 1)[0]
        c = cols[0] if "LAE" in label.upper() else cols[1] if "LBG" in label.upper() else cols[i % 2]
        ax.errorbar(m2, y, yerr=e, fmt=marks[i % len(marks)] + "-", color=c, ms=6, lw=1.2, capsize=3,
                    label="%s (slope %+.2f)" % (label, slope))
        print("%-12s b1 %.2f | P(k,mu_top)/P_real at 0.05<=k<0.2: %.3f (linear Kaiser with f=%.2f: %.3f)"
              " -> f_eff %.2f | beta_1 slope vs <mu^2> at k<%.2f: %+.2f -> f_eff = f + slope = %.2f"
              % (label, b1, rtop, f, ktop, f_kaiser, kcut, slope, f + slope))
    xx = np.linspace(0, 1, 50)
    ax.plot(xx, 0 * xx, color="0.5", lw=1.2, label="catalog RSD = model $f$")
    ax.plot(xx, -f * xx, color="k", ls="--", lw=1.2, label="catalog without RSD (slope $-f$)")
    ax.set_xlim(0, 0.8)
    ax.set_xlabel(r"$\langle\mu^2\rangle$ of the $\mu$ bin", fontsize=11)
    ax.set_ylabel(r"$\beta_1(\mu) - \beta_1^{\rm real}$ at $k < %.2f$" % kcut, fontsize=11)
    ax.set_title(r"large-scale $\beta_1$: $\mu$ dependence", fontsize=12)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    P.save_figure(fig, fn, 130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--run", nargs=2, action="append", metavar=("PREFIX", "LABEL"), required=True)
    ap.add_argument("--out", default="fig", help="output prefix (default fig)")
    ap.add_argument("--z-out", type=float, default=None, help="redshift the runs used (for f)")
    ap.add_argument("--f", type=float, default=None, help="growth rate the runs used (instead of --z-out)")
    ap.add_argument("--omega-m", type=float, default=P.OMEGA_M)
    ap.add_argument("--omega-r", type=float, default=P.OMEGA_R)
    ap.add_argument("--kcut", type=float, default=0.12, help="large-scale window k < kcut (default 0.12)")
    ap.add_argument("--beta1-is-b1", action="store_true", help="real-space beta_1 -> b1 (fit without delta_Z)")
    a = ap.parse_args()

    f = a.f
    if f is None and a.z_out is not None:
        f = P.growth(max(a.z_out, 0.0) + 1.0, a.z_out, a.omega_m, 0.0, a.omega_r)[1]
    runs = [(label, load(prefix)) for prefix, label in a.run]
    plt = P._pyplot()
    if plt is None:
        sys.exit("needs matplotlib")
    fig4(plt, runs, a.out + "_fig4.png")
    fig5(plt, runs, a.out + "_fig5.png")
    if f is None:
        print("no --z-out / --f: skipping %s_rsdcheck.png" % a.out)
    else:
        print("f = %.4f" % f)
        rsdcheck(plt, runs, a.out + "_rsdcheck.png", f, a.kcut, a.beta1_is_b1)


if __name__ == "__main__":
    main()

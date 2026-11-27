#!/usr/bin/env python3
"""
n̄ P_err(k, μ) for one HOD row over the AbacusSummit base phase boxes, from the
lrgs.py results (real space, and redshift space along z). Same quantities and
fit as compare_perr.py / fit_perr.py, now with the phase-to-phase scatter:

    n̄ P_err(k, μ) = 1 − D · W(kR) · exp(−k² μ² ℓ_F²)

    python phase_perr.py $PSCRATCH/lrg_hods_phases \\
        $PSCRATCH/qso_sbp/Heavy_files_Andrej/AbacusBase/results \\
        --row 1003 --nmesh 512 --out perr_hod1003_N512

Arguments are directories (searched recursively) or files. A phase counts when
both results_AbacusBase_c=0_ph=NNN_hod=ROW_z=Z_Nmesh_N.npy and
results_RSD_AbacusBase_c=0_ph=NNN_hod=ROW_z=Z_Nmesh_N_axisrsd_2.npy are found.
Names without '=' and with a prefix (as uploads rename them) work too.

Writes, each as PNG and PDF:
  <out>_mean     ph000 alone (its own fit, as compare_perr.py), then the mean of
                 all phases: ±1σ phase scatter as a band, the mean's own error
                 (scatter/√N) as error bars, and the fit to the mean weighted by
                 that error, so its χ² is absolute
  <out>_bins     one panel each for real space and every μ bin: every phase thin,
                 ph000 dashed, the mean ± scatter, and the fit to the mean
  <out>_phases   one compare_perr.py panel per phase, each with its own fit
  <out>_fits     per-phase D, ℓ_F, R and the mean n̄P_err in one k window, with
                 the ensemble mean ± scatter
  <out>_summary.txt  the tables printed at the end
Every plot shows the full k range: the fit (k ≤ --kmax) solid, its
extrapolation dashed, k_Nyq and k_Nyq/2 marked.
"""

import argparse
import os
import re
import sys

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_perr import fit, load, model  # noqa: E402

NAME = re.compile(r"results_(RSD_)?AbacusBase_c=?(\d+)_ph=?(\d+)_hod=?(\d+)_z=?([0-9.]+?)_Nmesh_(\d+)"
                  r"(?:_axisrsd_(\d+))?\.npy$")


def find(paths, row, nmesh, axis):
    """{phase: (real file, rsd file)} for this row and Nmesh."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            for d, _, names in os.walk(p, followlinks=True):
                files += [os.path.join(d, n) for n in names if n.endswith(".npy")]
        else:
            files.append(p)
    real, rsd = {}, {}
    for f in sorted(files):
        m = NAME.search(os.path.basename(f))
        if not m or int(m.group(4)) != row or int(m.group(6)) != nmesh:
            continue
        ph = int(m.group(3))
        if m.group(1):
            if m.group(7) is None or int(m.group(7)) != axis:
                continue
            store = rsd
        else:
            store = real
        if ph in store and os.path.realpath(store[ph]) != os.path.realpath(f):
            print("note: ph%03d found twice; using %s, not %s" % (ph, store[ph], f))
            continue
        store[ph] = f
    for ph in sorted(set(real) ^ set(rsd)):
        print("note: ph%03d has only its %s file: left out" % (ph, "real" if ph in real else "RSD"))
    return dict((ph, (real[ph], rsd[ph])) for ph in sorted(set(real) & set(rsd)))


def fit_sigma(k, y_real, edges, y_rsd, s_real, s_rsd, kmax, kind):
    """Joint fit with given per-bin errors (absolute): popt, perr, chi2, dof."""
    m = k <= kmax
    xs, ys, ss, bins = [], [], [], []
    good = m & np.isfinite(y_real) & np.isfinite(s_real) & (s_real > 0)
    xs.append(k[good]); ys.append(y_real[good]); ss.append(s_real[good]); bins.append((0.0, 0.0))
    for j in range(y_rsd.shape[1]):
        good = m & np.isfinite(y_rsd[:, j]) & np.isfinite(s_rsd[:, j]) & (s_rsd[:, j] > 0)
        xs.append(k[good]); ys.append(y_rsd[good, j]); ss.append(s_rsd[good, j]); bins.append((edges[j], edges[j + 1]))
    lens = [len(x) for x in xs]
    X, Y, S = np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)

    def f(_, D, ell, R):
        out, i = [], 0
        for n, (m1, m2) in zip(lens, bins):
            out.append(model(X[i:i + n], D, ell, R, m1, m2, kind)); i += n
        return np.concatenate(out)

    popt, pcov = curve_fit(f, X, Y, p0=[0.3, 3.0, 3.0], sigma=S, absolute_sigma=True,
                           bounds=([0, 0, 0], [1, 50, 50]), maxfev=20000)
    chi2 = float(np.sum(((Y - f(X, *popt)) / S) ** 2))
    with np.errstate(invalid="ignore"):
        perr = np.sqrt(np.diag(pcov))
    return popt, perr, chi2, len(Y) - 3


def window_means(k, y, edges):
    """Mean of y over each k window [lo, hi): (n_windows,) or (n_windows, nmu)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (k >= lo) & (k < hi)
        with np.errstate(invalid="ignore"), _quiet():
            out.append(np.nanmean(y[sel], axis=0) if sel.any() else np.full(y.shape[1:], np.nan))
    return np.array(out)


class _quiet(object):
    def __enter__(self):
        import warnings
        self.w = warnings.catch_warnings(); self.w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self.w.__exit__(*a)


def save(fig, out, dpi=160):
    fig.savefig(out + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print("wrote %s.png and .pdf" % out)


def marks(ax, kny, kmax, xmax, with_kmax=True):
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    if with_kmax and kmax < xmax:
        ax.axvline(kmax, color="0.4", lw=0.9, ls="-.")
        ax.text(kmax * 1.03, 0.02, r"$k_{\max}$ (fit)", color="0.3", fontsize=8,
                transform=ax.get_xaxis_transform(), va="bottom")
    for kk, lab, ls, fs in ((kny, r"$k_{\rm Nyq}$", "--", 9), (kny / 2, r"$k_{\rm Nyq}/2$", ":", 8)):
        if kk <= xmax:
            ax.axvline(kk, color="crimson", lw=0.9, ls=ls)
            ax.text(kk * 0.97, 0.98, lab, color="crimson", ha="right", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=fs)
    ax.set_xscale("log")
    ax.grid(alpha=0.25, which="both")


def curves(ax, popt, edges, nmu, kmin, kfit, kend, kind, colors, lw=1.5):
    kk = np.geomspace(kmin, kfit, 300)
    kx = np.geomspace(kfit, kend, 300) if kend > kfit * 1.001 else None
    for j, (m1, m2) in enumerate([(0.0, 0.0)] + [(edges[i], edges[i + 1]) for i in range(nmu)]):
        ax.plot(kk, model(kk, *popt, m1, m2, kind), "-", color=colors[j], lw=lw)
        if kx is not None:
            ax.plot(kx, model(kx, *popt, m1, m2, kind), "--", color=colors[j], lw=lw * 0.75, alpha=0.8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="directories (searched recursively) or result files")
    ap.add_argument("--row", type=int, default=1003)
    ap.add_argument("--nmesh", type=int, default=512)
    ap.add_argument("--box", type=float, default=2000.0, help="box side [Mpc/h], for k_Nyq")
    ap.add_argument("--axis", type=int, default=2, help="RSD axis in the file names (lrgs.py: 2)")
    ap.add_argument("--kmax", type=float, default=0.4, help="fit range (the plots show all k)")
    ap.add_argument("--kplot", type=float, default=None, help="cut the plots at this k")
    ap.add_argument("--window", choices=["gauss", "tophat"], default="gauss")
    ap.add_argument("--kedges", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
                    help="k windows for the mean n̄P_err table")
    ap.add_argument("--kwin", nargs=2, type=float, default=[0.3, 0.4], help="k window for the per-phase plot")
    ap.add_argument("--kscan", nargs="*", type=float, default=[0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--ref", type=int, default=0, help="phase shown on its own in <out>_mean (default ph000)")
    ap.add_argument("--out", default=None, help="output prefix (default perr_hod<ROW>_N<NMESH>)")
    a = ap.parse_args()
    out = a.out or "perr_hod%d_N%d" % (a.row, a.nmesh)
    kny = np.pi * a.nmesh / a.box

    runs = find(a.paths, a.row, a.nmesh, a.axis)
    if not runs:
        sys.exit("no lrgs.py results for row %d at Nmesh %d under %s" % (a.row, a.nmesh, " ".join(a.paths)))
    phases = sorted(runs)
    print("%d phases: %s" % (len(phases), " ".join("ph%03d" % p for p in phases)))

    data = dict((ph, load(*runs[ph])) for ph in phases)        # k, y_real, mu_c, edges, y_rsd, nbar
    k = data[phases[0]][0]
    edges = data[phases[0]][3]
    for ph in phases:
        if len(data[ph][0]) != len(k) or not np.allclose(data[ph][0], k):
            sys.exit("ph%03d has different k bins from ph%03d (different Nmesh or box?)" % (ph, phases[0]))
    nmu = data[phases[0]][4].shape[1]
    YR = np.array([data[ph][1] for ph in phases])               # (nph, nk)
    YS = np.array([data[ph][4] for ph in phases])               # (nph, nk, nmu)
    nbar = np.array([data[ph][5] for ph in phases])
    n = len(phases)
    with _quiet():
        mR, mS = np.nanmean(YR, 0), np.nanmean(YS, 0)
        sR = np.nanstd(YR, 0, ddof=1) if n > 1 else np.zeros_like(mR)
        sS = np.nanstd(YS, 0, ddof=1) if n > 1 else np.zeros_like(mS)
        nR, nS = np.sum(np.isfinite(YR), 0), np.sum(np.isfinite(YS), 0)
        eR, eS = sR / np.sqrt(np.maximum(nR, 1)), sS / np.sqrt(np.maximum(nS, 1))

    # fits: each phase as in fit_perr.py (σ ∝ 1/k); the mean with its own error
    pf = dict((ph, fit(data[ph][0], data[ph][1], edges, data[ph][4], a.kmax, a.window)) for ph in phases)
    ens = fit_sigma(k, mR, edges, mS, eR, eS, a.kmax, a.window) if n > 2 else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blues = plt.cm.Blues(np.linspace(0.45, 0.95, nmu))
    colors = ["k"] + list(blues)
    labels = ["real space"] + [r"$\mu \in [%.2f,\,%.2f]$" % (edges[j], edges[j + 1]) for j in range(nmu)]
    xmax = a.kplot or k.max() * 1.05
    kfit = min(a.kmax, k.max())
    W = r"e^{-k^2R^2/2}" if a.window == "gauss" else r"W_{\rm TH}(kR)"
    model_txt = r"$\bar n P_{\rm err} = 1 - D\,%s\,e^{-k^2\mu^2\ell_F^2}$" % W
    info = r"row %d, $N_{\rm mesh}=%d$, $k_{\rm Nyq}=%.2f$" % (a.row, a.nmesh, kny)

    # ---------------------------------------------------- <out>_mean
    ref = a.ref if a.ref in data else phases[0]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    ax = axes[0]
    _, yr, _, _, ys, nb = data[ref]
    p, e = pf[ref][0], pf[ref][1]
    ax.plot(k, yr, "o", ms=3, color="k", label=labels[0])
    for j in range(nmu):
        ax.plot(k, ys[:, j], "o", ms=3, color=blues[j], label=labels[j + 1])
    curves(ax, p, edges, nmu, k.min(), kfit, k.max(), a.window, colors)
    marks(ax, kny, a.kmax, xmax)
    ax.set_title("ph%03d alone   (%s)" % (ref, info), fontsize=10)
    ax.text(0.03, 0.03, "$D = %.3f \\pm %.3f$\n$\\ell_F = %.2f \\pm %.2f$\n$R = %.2f \\pm %.2f$\n$\\bar n = %.3e$"
            % (p[0], e[0], p[1], e[1], p[2], e[2], nb), transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.set_ylabel(r"$\bar n\, P_{\rm err}(k,\mu)$")
    ax = axes[1]
    for j, (m_, s_, e_) in enumerate([(mR, sR, eR)] + [(mS[:, i], sS[:, i], eS[:, i]) for i in range(nmu)]):
        ax.fill_between(k, m_ - s_, m_ + s_, color=colors[j], alpha=0.15, lw=0)
        ax.errorbar(k, m_, yerr=e_, fmt="o", ms=3, color=colors[j], elinewidth=0.8, capsize=0)
    if ens is not None:
        curves(ax, ens[0], edges, nmu, k.min(), kfit, k.max(), a.window, colors)
        p, e, chi2, dof = ens
        ax.text(0.03, 0.03, "$D = %.4f \\pm %.4f$\n$\\ell_F = %.3f \\pm %.3f$\n$R = %.2f \\pm %.2f$\n"
                "$\\chi^2/{\\rm dof} = %.0f/%d$\n$\\bar n = %.3e \\pm %.1e$"
                % (p[0], e[0], p[1], e[1], p[2], e[2], chi2, dof, nbar.mean(), nbar.std(ddof=1) if n > 1 else 0),
                transform=ax.transAxes, fontsize=8.5, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    marks(ax, kny, a.kmax, xmax)
    ax.set_title(r"mean of %d phases: band $\pm1\sigma$ phase scatter, bars $\sigma/\sqrt{N}$" % n, fontsize=10)
    for ax in axes:
        ax.set_xlim(0.0028, xmax)
        ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    fig.suptitle(model_txt + r",  $k_{\max}=%g$" % a.kmax, fontsize=11)
    fig.tight_layout()
    save(fig, out + "_mean")
    plt.close(fig)

    # ---------------------------------------------------- <out>_bins
    fig, axes = plt.subplots(1, nmu + 1, figsize=(4.6 * (nmu + 1), 4.4), sharey=True)
    series = [(YR, mR, sR)] + [(YS[:, :, i], mS[:, i], sS[:, i]) for i in range(nmu)]
    for j, (ax, (Y, m_, s_)) in enumerate(zip(axes, series)):
        c = colors[j] if j else "0.25"
        for i, ph in enumerate(phases):
            ax.plot(k, Y[i], "-", color=c, lw=0.5, alpha=0.35)
        if ref in data:
            ax.plot(k, Y[phases.index(ref)], "--", color="darkorange", lw=1.1, label="ph%03d" % ref)
        ax.fill_between(k, m_ - s_, m_ + s_, color=c, alpha=0.18, lw=0, label=r"mean $\pm1\sigma$")
        ax.plot(k, m_, "-", color="k" if j == 0 else blues[j - 1], lw=2.0)
        if ens is not None:
            m1, m2 = (0.0, 0.0) if j == 0 else (edges[j - 1], edges[j])
            kk = np.geomspace(k.min(), kfit, 300)
            ax.plot(kk, model(kk, *ens[0], m1, m2, a.window), "-", color="crimson", lw=1.0, label="fit to mean")
            if k.max() > kfit * 1.001:
                kx = np.geomspace(kfit, k.max(), 300)
                ax.plot(kx, model(kx, *ens[0], m1, m2, a.window), "--", color="crimson", lw=0.8)
        marks(ax, kny, a.kmax, xmax)
        ax.set_xlim(0.0028, xmax)
        ax.set_title(labels[j] + ("  (%d phases)" % n if j == 0 else ""), fontsize=10)
        ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    axes[0].set_ylabel(r"$\bar n\, P_{\rm err}$")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(info + ": every phase (thin), the mean and its fit", fontsize=11)
    fig.tight_layout()
    save(fig, out + "_bins")
    plt.close(fig)

    # ---------------------------------------------------- <out>_phases
    ncol = min(5, n)
    nrow = int(np.ceil(n / float(ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.2 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax, ph in zip(axes.flat, phases):
        _, yr, _, _, ys, nb = data[ph]
        p, e = pf[ph][0], pf[ph][1]
        ax.plot(k, yr, ".", ms=2.5, color="k")
        for j in range(nmu):
            ax.plot(k, ys[:, j], ".", ms=2.5, color=blues[j])
        curves(ax, p, edges, nmu, k.min(), kfit, k.max(), a.window, colors, lw=1.1)
        marks(ax, kny, a.kmax, xmax, with_kmax=False)
        ax.set_xlim(0.0028, xmax)
        ax.text(0.03, 0.03, "ph%03d  $D=%.3f$  $\\ell_F=%.2f$  $R=%.2f$" % (ph, p[0], p[1], p[2]),
                transform=ax.transAxes, fontsize=7.5, bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85))
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\bar n\, P_{\rm err}$")
    fig.suptitle(info + ": each phase with its own fit (real black, μ bins blue)", fontsize=11)
    fig.tight_layout()
    save(fig, out + "_phases", dpi=110)
    plt.close(fig)

    # ---------------------------------------------------- <out>_fits
    lo, hi = a.kwin
    wsel = (k >= lo) & (k < hi)
    with _quiet():
        win = np.column_stack([np.nanmean(YR[:, wsel], 1)] + [np.nanmean(YS[:, wsel, j], 1) for j in range(nmu)])
    P = np.array([pf[ph][0] for ph in phases])
    E = np.array([pf[ph][1] for ph in phases])
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.0))
    names = [r"$D$", r"$\ell_F\ [h^{-1}{\rm Mpc}]$", r"$R\ [h^{-1}{\rm Mpc}]$"]
    for i, ax in enumerate(axes[:3]):
        ax.errorbar(phases, P[:, i], yerr=E[:, i], fmt="o", ms=4, color="k", capsize=0, elinewidth=0.8)
        mu_, sd_ = P[:, i].mean(), (P[:, i].std(ddof=1) if n > 1 else 0)
        ax.axhspan(mu_ - sd_, mu_ + sd_, color="0.6", alpha=0.2, lw=0)
        ax.axhline(mu_, color="0.3", lw=1, label="mean of phases $\\pm1\\sigma$")
        if ens is not None:
            ax.axhline(ens[0][i], color="crimson", lw=1, ls="--", label="fit to the mean")
        ax.set_ylabel(names[i]); ax.set_xlabel("phase")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    ax = axes[3]
    for j in range(nmu + 1):
        ax.plot(phases, win[:, j], "o-", ms=4, lw=0.8, color=colors[j], label=labels[j])
        ax.axhline(np.nanmean(win[:, j]), color=colors[j], lw=0.8, ls=":")
    ax.set_ylabel(r"$\langle\bar n P_{\rm err}\rangle$, $k\in[%g, %g)$" % (lo, hi)); ax.set_xlabel("phase")
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(alpha=0.25)
    fig.suptitle(info + r": per-phase fit ($k \leq %g$, weights $\propto k$ as in fit_perr.py) and window means" % a.kmax,
                 fontsize=11)
    fig.tight_layout()
    save(fig, out + "_fits")
    plt.close(fig)

    # ---------------------------------------------------- tables
    L = []
    L.append("row %d, Nmesh %d (k_Nyq %.3f), %d phases: %s" % (a.row, a.nmesh, kny, n, " ".join("%03d" % p for p in phases)))
    L.append("nbar: mean %.4e, phase scatter %.2e (%.2f %%)" % (nbar.mean(), nbar.std(ddof=1) if n > 1 else 0,
                                                              100 * (nbar.std(ddof=1) if n > 1 else 0) / nbar.mean()))
    L.append("")
    L.append("per-phase fit, k <= %g, weights ∝ k (fit_perr.py); window means for k in [%g, %g):" % (a.kmax, lo, hi))
    L.append("%-6s %10s  %15s %15s %15s   %s" % ("phase", "nbar", "D", "ell_F", "R",
                                              " ".join("%7s" % s for s in ["real"] + ["mu%d" % (j + 1) for j in range(nmu)])))
    for i, ph in enumerate(phases):
        L.append("ph%03d  %10.4e  %7.4f±%-7.4f %7.3f±%-7.3f %7.3f±%-7.3f   %s" % (
            ph, nbar[i], P[i, 0], E[i, 0], P[i, 1], E[i, 1], P[i, 2], E[i, 2], " ".join("%7.4f" % v for v in win[i])))
    if n > 1:
        L.append("mean   %10.4e  %15.4f %15.3f %15.3f   %s" % (nbar.mean(), P[:, 0].mean(), P[:, 1].mean(), P[:, 2].mean(),
                                                            " ".join("%7.4f" % v for v in np.nanmean(win, 0))))
        L.append("scatter%10.2e  %15.4f %15.3f %15.3f   %s" % (nbar.std(ddof=1), P[:, 0].std(ddof=1), P[:, 1].std(ddof=1),
                                                            P[:, 2].std(ddof=1), " ".join("%7.4f" % v for v in np.nanstd(win, 0, ddof=1))))
        L.append("err/√N %10.2e  %15.4f %15.3f %15.3f   %s" % (nbar.std(ddof=1) / np.sqrt(n), P[:, 0].std(ddof=1) / np.sqrt(n),
                                                            P[:, 1].std(ddof=1) / np.sqrt(n), P[:, 2].std(ddof=1) / np.sqrt(n),
                                                            " ".join("%7.4f" % v for v in np.nanstd(win, 0, ddof=1) / np.sqrt(n))))
        if ref in data:
            i = phases.index(ref)
            z = (win[i] - np.nanmean(win, 0)) / np.nanstd(win, 0, ddof=1)
            L.append("ph%03d vs the mean in [%g, %g), in units of the phase scatter: %s" % (
                ref, lo, hi, " ".join("%+.2f" % v for v in z)))
    if ens is not None:
        p, e, chi2, dof = ens
        L.append("")
        L.append("fit to the mean (errors σ/√N per bin, absolute; real and RSD bins treated as independent):")
        L.append("  D = %.4f ± %.4f   ell_F = %.3f ± %.3f   R = %.3f ± %.3f   chi2/dof = %.1f/%d" % (
            p[0], e[0], p[1], e[1], p[2], e[2], chi2, dof))
        if a.kscan:
            L.append("  refit vs k_max:  " + "   ".join(
                "k_max %g: D %.4f, ell_F %.3f, R %.3f, chi2/dof %.1f/%d" % ((km,) + tuple(r[0]) + (r[2], r[3]))
                for km, r in ((km, fit_sigma(k, mR, edges, mS, eR, eS, km, a.window)) for km in a.kscan)))
    L.append("")
    L.append("mean n̄P_err per k window, mean ± phase scatter:   real | " + " | ".join("mu%d" % (j + 1) for j in range(nmu)))
    wm = np.array([window_means(k, np.column_stack([YR[i]] + [YS[i][:, j] for j in range(nmu)]), a.kedges)
                   for i in range(n)])                           # (nph, nwin, nmu + 1)
    with _quiet():
        for w, (lo_, hi_) in enumerate(zip(a.kedges[:-1], a.kedges[1:])):
            m_, s_ = np.nanmean(wm[:, w], 0), (np.nanstd(wm[:, w], 0, ddof=1) if n > 1 else np.zeros(nmu + 1))
            L.append("  k in [%.2f, %.2f)  " % (lo_, hi_) + " | ".join("%.3f ± %.3f" % (x, y) for x, y in zip(m_, s_)))
    text = "\n".join(L)
    print("\n" + text)
    with open(out + "_summary.txt", "w") as fh:
        fh.write(text + "\n")
    print("wrote %s_summary.txt" % out)


if __name__ == "__main__":
    main()

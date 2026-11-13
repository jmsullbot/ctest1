#!/usr/bin/env python3
"""
Side-by-side n̄·P_err(k, μ) for two or more runs (e.g. base vs small box,
same HOD row), each with its own 3-parameter fit 1 − D·W(kR)·exp(−k²μ²ℓ_F²).

    python compare_perr.py --left  real_A.npy rsd_A.npy --left-label  "base 2000 Mpc/h" \
                           --right real_B.npy rsd_B.npy --right-label "small 500 Mpc/h" \
                           --kmax 0.4 --out compare.png

Any number of panels: one --run REAL RSD LABEL BOX NMESH per run, e.g.

    python compare_perr.py --run b256_real.npy b256_rsd.npy "base 256" 2000 256 \
                           --run b512_real.npy b512_rsd.npy "base 512" 2000 512 \
                           --run s256_real.npy s256_rsd.npy "small 256" 500 256

Besides the figure it prints, per run: the fit at --kmax, the mean n̄P_err
(real and each μ bin) in the k windows between --kedges, and D / ℓ_F refit
at each --kscan k_max. Reuses the model/fit machinery from fit_perr.py.
"""

import argparse

import numpy as np

from fit_perr import fit, load, model


def panel(ax, real, rsd, label, kmax, kind, box, nmesh=256):
    k, y_real, mu_c, edges, y_rsd, nbar = load(real, rsd)
    popt, perr, rms, wrms, npts = fit(k, y_real, edges, y_rsd, kmax, kind)
    nmu = y_rsd.shape[1]
    kny = np.pi * nmesh / box if box else None

    import matplotlib.pyplot as plt
    blues = plt.cm.Blues(np.linspace(0.45, 0.95, nmu))
    kk = np.geomspace(k.min(), kmax, 400)

    ax.plot(k, y_real, "o", ms=3.2, color="k", label="real space")
    ax.plot(kk, model(kk, *popt, 0.0, 0.0, kind), "-", color="k", lw=1.5)
    for j in range(nmu):
        ax.plot(k, y_rsd[:, j], "o", ms=3.2, color=blues[j],
                label=rf"$\mu \in [{edges[j]:.2f},\,{edges[j+1]:.2f}]$")
        ax.plot(kk, model(kk, *popt, edges[j], edges[j + 1], kind), "-", color=blues[j], lw=1.5)
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    xmax = kmax * 1.05                       # matches the x-limit set in main()
    if kny and kny <= xmax:                  # draw only what falls inside the plot
        ax.axvline(kny, color="crimson", lw=1, ls="--")
        ax.text(kny * 0.97, 0.98, r"$k_{\rm Nyq}$", color="crimson", ha="right", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9)
    if kny and kny / 2 <= xmax:
        ax.axvline(kny / 2, color="crimson", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    ax.set_title(label + (f"   ($N_{{\\rm mesh}}={nmesh}$, $k_{{\\rm Nyq}}={kny:.2f}$)" if kny else ""), fontsize=10)
    txt = (f"$D = {popt[0]:.3f} \\pm {perr[0]:.3f}$\n"
           f"$\\ell_F = {popt[1]:.2f} \\pm {perr[1]:.2f}$\n"
           f"$R = {popt[2]:.2f} \\pm {perr[2]:.2f}$\n"
           f"$\\bar n = {nbar:.2e}$")
    ax.text(0.03, 0.03, txt, transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.grid(alpha=0.25, which="both")
    return popt, perr, nbar, rms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", nargs=2, metavar=("REAL", "RSD"))
    ap.add_argument("--right", nargs=2, metavar=("REAL", "RSD"))
    ap.add_argument("--left-label", default="left"); ap.add_argument("--right-label", default="right")
    ap.add_argument("--left-box", type=float, default=None, help="box side [Mpc/h], for k_Nyq line")
    ap.add_argument("--right-box", type=float, default=None)
    ap.add_argument("--left-nmesh", type=int, default=256); ap.add_argument("--right-nmesh", type=int, default=256)
    ap.add_argument("--run", nargs=5, action="append", metavar=("REAL", "RSD", "LABEL", "BOX", "NMESH"),
                    help="one panel per --run (repeatable); replaces --left/--right")
    ap.add_argument("--kmax", type=float, default=0.4)
    ap.add_argument("--window", choices=["gauss", "tophat"], default="gauss")
    ap.add_argument("--kedges", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.4],
                    help="k windows for the mean n̄P_err table")
    ap.add_argument("--kscan", nargs="*", type=float, default=[0.2, 0.3, 0.4],
                    help="refit at each of these k_max (table only)")
    ap.add_argument("--out", default="compare_perr.png")
    a = ap.parse_args()

    if a.run:
        runs = [(r[0], r[1], r[2], float(r[3]), int(r[4])) for r in a.run]
    elif a.left and a.right:
        runs = [(a.left[0], a.left[1], a.left_label, a.left_box, a.left_nmesh),
                (a.right[0], a.right[1], a.right_label, a.right_box, a.right_nmesh)]
    else:
        ap.error("give --left and --right, or one --run per panel")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(runs), figsize=(6.25 * len(runs), 5.0), sharey=True)
    axes = np.atleast_1d(axes)
    res = [panel(ax, real, rsd, lab, a.kmax, a.window, box, nm)
           for ax, (real, rsd, lab, box, nm) in zip(axes, runs)]
    axes[0].set_ylabel(r"$\bar n\, P_{\rm err}(k,\mu)$")
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False)
    Wtxt = r"e^{-k^2R^2/2}" if a.window == "gauss" else r"W_{\rm TH}(kR)"
    fig.suptitle(r"$\bar n P_{\rm err} = 1 - D\,%s\,e^{-k^2\mu^2\ell_F^2}$,  $k_{\max}=%g$" % (Wtxt, a.kmax), fontsize=11)
    for ax in axes:
        ax.set_xlim(0.0028, a.kmax * 1.05)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)

    w = max(12, max(len(r[2]) for r in runs))
    print(f"fit at k_max = {a.kmax}")
    print(f"{'':{w}s} {'D':>16s} {'ell_F [Mpc/h]':>18s} {'R [Mpc/h]':>16s} {'nbar':>10s} {'rms':>7s}")
    for (_, _, lab, _, _), (p, e, nb, rms) in zip(runs, res):
        print(f"{lab:{w}s} {p[0]:7.4f} ± {e[0]:6.4f} {p[1]:8.3f} ± {e[1]:6.3f}   {p[2]:7.3f} ± {e[2]:6.3f} {nb:10.3e} {rms:7.4f}")

    data = [load(real, rsd) for real, rsd, _, _, _ in runs]
    nmu = data[0][4].shape[1]
    print("\nmean n̄P_err per k window:   real | " + " | ".join(f"mu bin {j + 1}" for j in range(nmu)))
    for lo, hi in zip(a.kedges[:-1], a.kedges[1:]):
        print(f"  k in [{lo:.2f}, {hi:.2f})")
        for (_, _, lab, _, _), (k, yr, _, _, ys, _) in zip(runs, data):
            sel = (k >= lo) & (k < hi)
            if not sel.any():
                print(f"    {lab:{w}s}  (no bins)"); continue
            print(f"    {lab:{w}s}  {yr[sel].mean():6.3f} | " + " | ".join(f"{ys[sel, j].mean():6.3f}" for j in range(ys.shape[1])))

    if a.kscan:
        print("\nrefit vs k_max:  D / ell_F [Mpc/h] / R [Mpc/h]")
        print(f"  {'':{w}s} " + "  ".join(f"{'k_max=' + format(km, 'g'):>22s}" for km in a.kscan))
        for (_, _, lab, _, _), (k, yr, _, edges, ys, _) in zip(runs, data):
            cells = []
            for km in a.kscan:
                p = fit(k, yr, edges, ys, km, a.window)[0]
                cells.append(f"{p[0]:6.3f}/{p[1]:5.2f}/{p[2]:5.2f}")
            print(f"  {lab:{w}s} " + "  ".join(f"{c:>22s}" for c in cells))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

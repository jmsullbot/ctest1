#!/usr/bin/env python3
"""
Side-by-side n̄·P_err(k, μ) for two runs (e.g. base vs small box, same HOD
row), each with its own 3-parameter fit  1 − D·W(kR)·exp(−k²μ²ℓ_F²).

    python compare_perr.py --left  real_A.npy rsd_A.npy --left-label  "base 2000 Mpc/h" \
                           --right real_B.npy rsd_B.npy --right-label "small 500 Mpc/h" \
                           --kmax 0.4 --out compare.png

Reuses the model/fit machinery from fit_perr.py.
"""

import argparse

import numpy as np

from fit_perr import fit, load, model


def panel(ax, real, rsd, label, kmax, kind, box):
    k, y_real, mu_c, edges, y_rsd, nbar = load(real, rsd)
    popt, perr, rms, wrms, npts = fit(k, y_real, edges, y_rsd, kmax, kind)
    nmu = y_rsd.shape[1]
    kny = np.pi * 256 / box if box else None

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
    if kny and kny < 3 * kmax:
        ax.axvline(kny, color="crimson", lw=1, ls="--")
        ax.text(kny * 0.97, 0.98, r"$k_{\rm Nyq}$", color="crimson", ha="right", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9)
        ax.axvline(kny / 2, color="crimson", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    ax.set_title(label + (f"   ($N_{{\\rm mesh}}=256$, $k_{{\\rm Nyq}}={kny:.2f}$)" if kny else ""), fontsize=10)
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
    ap.add_argument("--left", nargs=2, required=True, metavar=("REAL", "RSD"))
    ap.add_argument("--right", nargs=2, required=True, metavar=("REAL", "RSD"))
    ap.add_argument("--left-label", default="left"); ap.add_argument("--right-label", default="right")
    ap.add_argument("--left-box", type=float, default=None, help="box side [Mpc/h], for k_Nyq line")
    ap.add_argument("--right-box", type=float, default=None)
    ap.add_argument("--kmax", type=float, default=0.4)
    ap.add_argument("--window", choices=["gauss", "tophat"], default="gauss")
    ap.add_argument("--out", default="compare_perr.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    res = {}
    for ax, files, lab, box, key in ((axes[0], a.left, a.left_label, a.left_box, "left"),
                                     (axes[1], a.right, a.right_label, a.right_box, "right")):
        res[key] = panel(ax, files[0], files[1], lab, a.kmax, a.window, box)
    axes[0].set_ylabel(r"$\bar n\, P_{\rm err}(k,\mu)$")
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False)
    Wtxt = r"e^{-k^2R^2/2}" if a.window == "gauss" else r"W_{\rm TH}(kR)"
    fig.suptitle(r"$\bar n P_{\rm err} = 1 - D\,%s\,e^{-k^2\mu^2\ell_F^2}$,  $k_{\max}=%g$" % (Wtxt, a.kmax), fontsize=11)
    for ax in axes:
        ax.set_xlim(min(res["left"][0].size and 0.0028, 0.0028), a.kmax * 1.05)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)

    print(f"{'':12s} {'D':>16s} {'ell_F [Mpc/h]':>18s} {'R [Mpc/h]':>16s} {'nbar':>10s} {'rms':>7s}")
    for key, lab in (("left", a.left_label), ("right", a.right_label)):
        p, e, nb, rms = res[key]
        print(f"{lab[:12]:12s} {p[0]:7.4f} ± {e[0]:6.4f} {p[1]:8.3f} ± {e[1]:6.3f}   {p[2]:7.3f} ± {e[2]:6.3f} {nb:10.3e} {rms:7.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

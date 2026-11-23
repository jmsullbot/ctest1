#!/usr/bin/env python
"""Ensemble over phase boxes of the perr_from_ic.py results for one HOD row.

    python phase_ensemble.py --root $PSCRATCH/lrg_hods_phases --row 1003 --nmesh 512 \\
        --z-out 0.5 --omega-m 0.315192 --omega-nu 0.0014197 --omega-r 0 --out ens

Reads ROOT/phNNN/perr/hod<ROW>_N<NMESH>_{real,rsd}.npy (+ the _ops sidecars) for
every phase found (or --phases). Writes
  <out>_mean_real.npy, _mean_rsd.npy  ensemble mean in the lrgs.py layout, so
                                      plot_fig4.py / compare_perr.py / fit_perr.py read it
  <out>_bias.txt, _bias.csv           per-phase bias and stochastic parameters from
                                      fit_bias.py (even polynomials, arXiv:1811.10640
                                      covariance), the ensemble mean and scatter, and
                                      scatter / mean quoted error per parameter
  <out>_transfer.png/.pdf             mean +- 1 sigma phase scatter of beta_1..beta_3 and
                                      nbar P_err, real space and every mu bin
  <out>_bias.png/.pdf                 per-phase parameters with their quoted errors
--lrgs REAL RSD compares one phase (--lrgs-phase, default 0) with lrgs.py results for
the same row: window means printed side by side and an overlay figure
<out>_lrgs_check.png/.pdf. lrgs.py fits real space without delta_Z; unless these runs
used --real-no-dz too (pass it here), our real-space beta_1 / beta_G2 are shifted by
+1 / +1/2 for that comparison (their low-k offset, arXiv:2409.10609 eqs. 20 vs 45).
"""
from __future__ import print_function

import argparse
import glob
import os
import re
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_bias as FB  # noqa: E402
import perr_from_ic as P  # noqa: E402

PARAMS = ("b1", "b2", "bG2", "b3", "alpha0", "alpha1", "alpha2")
WINDOWS = ((0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 0.8))


def L(fn):
    return [np.asarray(x, float) for x in np.load(fn, allow_pickle=True)]


def find_phases(root, row, nmesh, only):
    out = []
    for fn in sorted(glob.glob(os.path.join(root, "ph*", "perr", "hod%d_N%d_real.npy" % (row, nmesh)))):
        m = re.search(r"ph(\d+)", fn)
        ph = int(m.group(1))
        if only is not None and ph not in only:
            continue
        prefix = fn[:-len("_real.npy")]
        if os.path.exists(prefix + "_rsd.npy"):
            out.append((ph, prefix))
    return out


def save_obj(fn, items):
    arr = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        arr[i] = x
    np.save(fn, arr, allow_pickle=True)


def ensemble_mean(runs):
    """Mean over phases in the results layout (r from the mean cross and auto powers)."""
    real = [r["real"] for r in runs]
    rsd = [r["rsd"] for r in runs]

    def mean(items, i):
        return np.nanmean([it[i] for it in items], axis=0)

    m_real = [real[0][0]] + [mean(real, i) for i in range(1, 10)]
    m_real[4] = mean(real, 4)
    m_rsd = [rsd[0][0], rsd[0][1]] + [mean(rsd, i) for i in range(2, 11)]
    return m_real, m_rsd


def window_means(k, y):
    out = []
    for lo, hi in WINDOWS:
        s = (k >= lo) & (k < hi)
        out.append(np.nanmean(y[s]) if s.any() else np.nan)
    return np.array(out)


def plot_transfer(plt, runs, fn, title, overlay=None):
    """1 x 5 panels: beta_1, beta_2, beta_G2, beta_3, nbar P_err; mean +- 1 sigma."""
    k = runs[0]["real"][0]
    nmu = runs[0]["rsd"][2].shape[1]
    mus = runs[0]["rsd"][1]
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.0))
    for p, (ax, (lab, cmap)) in enumerate(zip(axes, P.PANELS)):
        if p < 4:
            real = np.array([r["real"][5 + p] for r in runs])
            rsd = np.array([r["rsd"][6 + p] for r in runs])
        else:
            real = np.array([r["real"][-1] * r["real"][1] for r in runs])
            rsd = np.array([r["rsd"][-1] * r["rsd"][2] for r in runs])
        shades = plt.get_cmap(cmap)(np.linspace(0.4, 0.85, nmu))
        for j in range(nmu):
            m, s = np.nanmean(rsd[:, :, j], 0), np.nanstd(rsd[:, :, j], 0, ddof=1) if len(runs) > 1 else 0 * k
            ax.plot(k, m, color=shades[j], lw=1.2, label=r"$\mu = %.2f$" % mus[j])
            ax.fill_between(k, m - s, m + s, color=shades[j], alpha=0.18, lw=0)
        m, s = np.nanmean(real, 0), np.nanstd(real, 0, ddof=1) if len(runs) > 1 else 0 * k
        ax.plot(k, m, color="k", lw=1.3, label="real space", zorder=3)
        ax.fill_between(k, m - s, m + s, color="k", alpha=0.15, lw=0, zorder=2)
        if overlay is not None:
            for q, (lbl, ls, ov_real, ov_rsd) in enumerate(overlay):
                yr = ov_real[p]
                ax.plot(k[:len(yr)], yr[:len(k)], color="k", ls=ls, lw=1.0, label=lbl if p == 0 else None)
                for j in range(nmu):
                    ax.plot(k[:len(yr)], ov_rsd[p][:len(k), j], color=shades[j], ls=ls, lw=0.9)
        if p == 4:
            ax.axhline(1.0, color="0.45", ls="--", lw=1.0)
            lab = r"$\bar n\,P_{\rm err}$"
        ax.set_xscale("log")
        ax.set_xlabel(P.KLABEL, fontsize=11)
        lo, hi = np.nanpercentile(np.concatenate([real[:, k > 0.03].ravel(), rsd[:, k > 0.03].ravel()]), [1, 99])
        pad = 0.3 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
        ax.text(0.06, 0.93, lab, transform=ax.transAxes, fontsize=14, va="top")
        ax.grid(alpha=0.2, which="both")
    axes[0].legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    P.save_figure(fig, fn, 130)
    plt.close(fig)


def plot_bias(plt, phases, fits, fn, title):
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    for ax, p in zip(axes, ("b1", "b2", "bG2", "b3")):
        v = np.array([f[p][0] for f in fits])
        e = np.array([f[p][1] for f in fits])
        ax.errorbar(phases, v, yerr=e, fmt="o", ms=4, color="#1f77b4", capsize=2, label="fit error (1811.10640)")
        m, s = np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0
        ax.axhline(m, color="k", lw=1.0)
        ax.axhspan(m - s, m + s, color="0.6", alpha=0.25, lw=0, label="mean $\\pm$ phase scatter")
        ax.set_xlabel("phase", fontsize=11)
        ax.set_title("%s = %.3f $\\pm$ %.3f (scatter/err %.2f)" % (p, m, s, s / np.mean(e) if len(v) > 1 else np.nan),
                     fontsize=11)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    P.save_figure(fig, fn, 130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--row", type=int, default=1003)
    ap.add_argument("--nmesh", type=int, default=512)
    ap.add_argument("--phases", type=int, nargs="+", default=None, help="only these phase numbers")
    ap.add_argument("--z-out", type=float, default=0.5)
    ap.add_argument("--omega-m", type=float, default=0.315192)
    ap.add_argument("--omega-nu", type=float, default=0.0014197)
    ap.add_argument("--omega-r", type=float, default=0.0)
    ap.add_argument("--kmax", type=float, default=0.4)
    ap.add_argument("--kmax-rsd", type=float, default=0.2)
    ap.add_argument("--order", type=int, choices=(2, 4), default=4)
    ap.add_argument("--no-theory-error", action="store_true")
    ap.add_argument("--real-no-dz", action="store_true", help="the runs fitted real space without delta_Z")
    ap.add_argument("--lrgs", nargs=2, metavar=("REAL", "RSD"), default=None,
                    help="lrgs.py results (real, RSD) for the same row, to compare with --lrgs-phase")
    ap.add_argument("--lrgs-phase", type=int, default=0)
    ap.add_argument("--out", default="ensemble")
    a = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice")     # empty (k, mu) bins at low k
    warnings.filterwarnings("ignore", message="Degrees of freedom")

    found = find_phases(a.root, a.row, a.nmesh, a.phases)
    if not found:
        sys.exit("no results under %s/ph*/perr/hod%d_N%d_{real,rsd}.npy" % (a.root, a.row, a.nmesh))
    print("%d phases: %s" % (len(found), " ".join("ph%03d" % p for p, _ in found)))
    D_ratio = 1.0 / P.growth(a.z_out, 0.0, a.omega_m, a.omega_nu, a.omega_r)[0]
    fa = types.SimpleNamespace(kmax=a.kmax, kmax_rsd=a.kmax_rsd, order=a.order,
                               no_theory_error=a.no_theory_error, real_no_dz=a.real_no_dz)
    runs, fits, phases = [], [], []
    for ph, prefix in found:
        R = {"real": L(prefix + "_real.npy"), "rsd": L(prefix + "_rsd.npy"), "prefix": prefix}
        if os.path.exists(prefix + "_real_ops.npy"):
            R.update(FB.load(prefix, None))
            res, _ = FB.fit_run(R, fa, D_ratio)
            fits.append(res)
        else:
            print("ph%03d: no _ops sidecars (old perr_from_ic.py?): left out of the bias fits" % ph)
            fits.append(None)
        runs.append(R)
        phases.append(ph)
    k0 = runs[0]["real"][0]
    if any(len(r["real"][0]) != len(k0) or not np.allclose(r["real"][0], k0) for r in runs):
        sys.exit("phases have different k bins (different --nmesh or box?)")

    # ensemble mean in the results layout
    m_real, m_rsd = ensemble_mean(runs)
    save_obj(a.out + "_mean_real.npy", m_real)
    save_obj(a.out + "_mean_rsd.npy", m_rsd)

    # bias table
    ok = [(ph, f) for ph, f in zip(phases, fits) if f is not None]
    lines = ["Row %d, Nmesh %d, %d phases. Bias from fit_bias.py: a0 + a2 k^2%s, k <= %.2f, arXiv:1811.10640 "
             "covariance%s; conversion arXiv:2409.10609 eq. %s." % (
                 a.row, a.nmesh, len(ok), " + a4 k^4" if a.order == 4 else "", a.kmax,
                 "" if a.no_theory_error else " (+ theory error)", "20" if a.real_no_dz else "45"), ""]
    hdr = "%-6s " % "phase" + " ".join("%16s" % p for p in PARAMS)
    lines.append(hdr)
    for ph, f in ok:
        lines.append("ph%03d  " % ph + " ".join("%8.3f+-%-6.3f" % f[p] if p in f else "%16s" % "-" for p in PARAMS))
    if len(ok) > 1:
        lines.append("")
        for name, fn_ in (("mean", np.mean), ("scatter", lambda x: np.std(x, ddof=1))):
            lines.append("%-6s " % name + " ".join("%16.3f" % fn_([f[p][0] for _, f in ok if p in f]) for p in PARAMS))
        lines.append("%-6s " % "err/sqN" + " ".join(
            "%16.3f" % (np.std([f[p][0] for _, f in ok if p in f], ddof=1) / np.sqrt(len(ok))) for p in PARAMS))
        lines.append("%-6s " % "quoted" + " ".join("%16.3f" % np.mean([f[p][1] for _, f in ok if p in f]) for p in PARAMS))
        lines.append("%-6s " % "ratio" + " ".join(
            "%16.2f" % (np.std([f[p][0] for _, f in ok if p in f], ddof=1) / np.mean([f[p][1] for _, f in ok if p in f]))
            for p in PARAMS))
        lines.append("ratio = phase scatter / mean quoted fit error: ~1 if the covariance is right "
                     "(alpha2 and RSD-side numbers also carry the RSD model).")
    text = "\n".join(lines)
    print(text)
    with open(a.out + "_bias.txt", "w") as fh:
        fh.write(text + "\n")
    with open(a.out + "_bias.csv", "w") as fh:
        fh.write("phase," + ",".join("%s,%s_err" % (p, p) for p in PARAMS) + "\n")
        for ph, f in ok:
            fh.write("%d," % ph + ",".join("%.6g,%.6g" % f[p] if p in f else "," for p in PARAMS) + "\n")

    # P_err windows: ensemble mean and scatter
    nmu = runs[0]["rsd"][2].shape[1]
    print("\nnbar*P_err per k window: mean +- phase scatter   real | mu bins")
    for w, (lo, hi) in enumerate(WINDOWS):
        vals = [np.array([window_means(k0, r["real"][-1] * r["real"][1])[w] for r in runs])]
        vals += [np.array([window_means(k0, r["rsd"][-1] * r["rsd"][2][:, j])[w] for r in runs]) for j in range(nmu)]
        print("  k %.2f-%.2f  " % (lo, hi) + " | ".join(
            "%.3f+-%.3f" % (np.nanmean(v), np.nanstd(v, ddof=1) if len(v) > 1 else 0) for v in vals))

    plt = P._pyplot()
    if plt is not None:
        title = "row %d, $N_{\\rm mesh}=%d$: mean $\\pm$ phase scatter over %d boxes" % (a.row, a.nmesh, len(runs))
        plot_transfer(plt, runs, a.out + "_transfer.png", title)
        if len(ok) > 1:
            plot_bias(plt, [ph for ph, _ in ok], [f for _, f in ok], a.out + "_bias.png",
                      "row %d: per-phase bias, fit errors vs phase scatter" % a.row)

    # optional: one phase against lrgs.py
    if a.lrgs:
        if a.lrgs_phase not in phases:
            sys.exit("--lrgs-phase %d has no results here" % a.lrgs_phase)
        R = dict(runs[phases.index(a.lrgs_phase)])
        if os.path.exists(R["prefix"] + "_rsd_losz.npy"):          # lrgs.py's RSD is LOS z only
            R["rsd"] = L(R["prefix"] + "_rsd_losz.npy")
            print("\n(redshift space compared on our LOS z file, %s_rsd_losz.npy)" % os.path.basename(R["prefix"]))
        lr, ls_ = L(a.lrgs[0]), L(a.lrgs[1])
        kl = lr[0]
        n = min(len(kl), len(k0))
        if not np.allclose(kl[:n], k0[:n], rtol=1e-3):
            print("WARNING: lrgs.py k bins differ from ours; comparing bin by bin anyway")
        shift = (0.0, 0.0) if a.real_no_dz else (1.0, 0.5)
        ours_real = [R["real"][5] + shift[0], R["real"][6], R["real"][7] + shift[1], R["real"][8],
                     R["real"][-1] * R["real"][1]]
        ours_rsd = [R["rsd"][6 + p] for p in range(4)] + [R["rsd"][-1] * R["rsd"][2]]
        their_real = [lr[5 + p] for p in range(4)] + [lr[-1] * lr[1]]
        their_rsd = [ls_[6 + p] for p in range(4)] + [ls_[-1] * ls_[2]]
        print("\nph%03d vs lrgs.py (%s): window means, ours / lrgs%s" % (
            a.lrgs_phase, os.path.basename(a.lrgs[0]),
            "" if a.real_no_dz else "; our real beta_1, beta_G2 shifted by +1, +1/2"))
        names = ("beta1", "beta2", "betaG2", "beta3", "nbarPerr")
        for p, nm in enumerate(names):
            wo, wt = window_means(k0[:n], ours_real[p][:n]), window_means(kl[:n], their_real[p][:n])
            print("  real %-8s " % nm + "  ".join("%.3f/%.3f" % (x, y) for x, y in zip(wo, wt)))
            for j in range(nmu):
                wo = window_means(k0[:n], ours_rsd[p][:n, j])
                wt = window_means(kl[:n], their_rsd[p][:n, j])
                print("  mu%d  %-8s " % (j, nm) + "  ".join("%.3f/%.3f" % (x, y) for x, y in zip(wo, wt)))
        print("  windows: " + "  ".join("%.2f-%.2f" % w for w in WINDOWS))
        if plt is not None:
            real_shifted = list(R["real"])
            real_shifted[5] = R["real"][5] + shift[0]
            real_shifted[7] = R["real"][7] + shift[1]
            one = [{"real": real_shifted, "rsd": R["rsd"]}]
            plot_transfer(plt, one, a.out + "_lrgs_check.png",
                          "ph%03d row %d: perr_from_ic.py (solid) vs lrgs.py (dashed)" % (a.lrgs_phase, a.row),
                          overlay=[("lrgs.py", "--", their_real, their_rsd)])


if __name__ == "__main__":
    main()

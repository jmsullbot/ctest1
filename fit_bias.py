#!/usr/bin/env python
"""Bias parameters from even-in-k polynomial fits to the low-k limits of the
transfer functions (perr_from_ic.py / lrgs.py results), weighted with the
transfer-function covariance of arXiv:1811.10640, compared with Table III of
arXiv:2505.03626.

    python fit_bias.py --run LAE_ODIN "ODIN LAE" --run LAE_S5 "S5 LAE" \\
        --run LBG_CARS "CARS LBG" --run LBG_S5 "S5 LBG" --z-out 3 --out bias

Needs, per run, PREFIX_real.npy and PREFIX_real_ops.npy (mode counts and
operator powers per k bin, written by perr_from_ic.py), and optionally
PREFIX_rsd.npy / PREFIX_rsd_ops.npy. The operator powers depend only on the IC
grid, not on the galaxies, so --ops PREFIX lets every run share one set.

Fit (per transfer function, real space by default)
  beta_X(k) = a0 + a2 k^2 + a4 k^4   for k <= --kmax (default 0.4 h/Mpc,
  --order 2 drops a4), as the real-space templates of arXiv:2409.10609 (eq. 58)
  used for arXiv:2412.01888 / 2505.03626, with generalized least squares and
    Cov(k_i, k_j) = C_noise(k_i) delta_ij
                    + dbeta(k_i) dbeta(k_j) exp[-(k_i - k_j)^2 / (2 Dk^2)]
  (arXiv:1811.10640, eqs. 86-88):
    C_noise = P_err / (N_k P_O)         N_k = all modes in the bin (both k and
                                        -k), P_O the operator's power; eq. 87
                                        has P_11 for beta_1. Checked on 16 mock
                                        realizations: scatter / prediction
                                        = 0.9-1.1 for every beta, real and RSD.
    dbeta = b1 (D/D0)^4 (k / 0.45)^3.3  beta_1 (two-loop theory error)
    dbeta = b1 (D/D0)^2 (k / 0.30)^1.8  beta_2, beta_G2, beta_3 (one-loop)
    Dk = 0.2 h/Mpc; b1 from a noise-only pass. --no-theory-error drops it.

Low-k limits -> Eulerian bias (arXiv:2409.10609 eq. 45; the model keeps
delta_Z with coefficient 1, as perr_from_ic.py does by default):
  b1 = 1 + a0[beta_1]   b2 = 2 a0[beta_2]   b3 = 6 a0[beta_3]
  b_G2 = a0[beta_G2] - 2/7 b1 + 1/2
  (--real-no-dz, i.e. fits without delta_Z as in lrgs.py: b1 = a0[beta_1],
  b_G2 = a0[beta_G2] - 2/7 b1; eq. 20 there.)
Stochastic terms (as arXiv:2409.10609 / 2505.03626 eq. 12, k_S = 0.45 h/Mpc):
  real:  nbar P_err = 1 + alpha0 + alpha1 (k/k_S)^2,         k <= --kmax
  RSD:   ... + alpha2 <mu^2>_bin (k/k_S)^2, alpha0/1 fixed,    k <= --kmax-rsd (0.2)
  weights Var(P_err) = 2 P_err^2 / N_k (Gaussian).
Not obtainable from polynomials: b_Gamma3 and the Table III b_nabla2delta come
from the one-loop shape fitted to beta_1 in those papers; the k^2 coefficient
of beta_1 printed here also absorbs that shape. c_mu2 and b4 need the one-loop
redshift-space shape too.

Outputs: <out>_table.txt (and .csv), <out>_fits.png/.pdf (real-space transfer
functions with errors and the fits).
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perr_from_ic as P  # noqa: E402

KS = 0.45
DK = 0.2
NAMES = ("beta1", "beta2", "betaG2", "beta3")
LABELS = (r"$\beta_1$", r"$\beta_2$", r"$\beta_{\mathcal{G}_2}$", r"$\beta_3$")

# Table III of arXiv:2505.03626 (z = 3)
TABLE3_COLS = ("ODIN LAE MTNG", "S5 LAE MTNG", "S5 LBG MTNG", "CARS LBG MTNG",
               "ODIN LAE Astrid", "S5 LAE Astrid", "S5 LBG Astrid", "CARS LBG Astrid")
TABLE3 = {
    "b1": (2.33, 2.68, 3.92, 4.24, 2.15, 1.89, 4.05, 3.96),
    "b2": (0.17, 0.61, 4.85, 6.21, 1.05, -0.052, 7.76, 6.02),
    "b3": (-1.23, -2.68, 8.75, 13.14, 6.73, -1.05, 12.46, 9.17),
    "bG2": (-0.803, -0.84, -2.28, -3.03, -0.98, -0.52, -1.19, -1.44),
    "bGamma3": (1.76, 1.71, 4.83, 7.04, 4.37, -9.44, -2.62, 5.38),
    "bnabla2delta": (-0.84, -1.05, -1.35, -1.69, 0.66, -5.49, -2.83, 1.25),
    "alpha0": (0.046, 0.041, -0.17, -0.16, 0.074, 0.023, -0.307, -0.32),
    "alpha1": (-0.023, -0.018, -0.0085, -0.037, -0.068, 0.063, 0.010, -0.047),
    "alpha2": (0.18, 0.32, 0.73, 0.48, -0.12, -0.093, 0.0081, 0.044),
}


def table3_key(label):
    """Which Table III sample a run label refers to (e.g. 'S5 LBG'), or None."""
    u = label.upper().replace("_", " ")
    kind = "LAE" if "LAE" in u else ("LBG" if ("LBG" in u or "CARS" in u) else None)
    sel = "ODIN" if "ODIN" in u else ("CARS" if "CARS" in u else ("S5" if "S5" in u else None))
    return None if kind is None or sel is None else "%s %s" % (sel, kind)


def load(prefix, ops_prefix):
    L = lambda f: [np.asarray(x, float) for x in np.load(f, allow_pickle=True)]
    R = {"real": L(prefix + "_real.npy")}
    src = ops_prefix or prefix
    if not os.path.exists(src + "_real_ops.npy"):
        sys.exit("%s_real_ops.npy not found: rerun perr_from_ic.py (it writes it), or pass --ops PREFIX "
                 "of any run on the same IC grid" % src)
    R["real_ops"] = L(src + "_real_ops.npy")
    if os.path.exists(prefix + "_rsd.npy") and os.path.exists(src + "_rsd_ops.npy"):
        R["rsd"], R["rsd_ops"] = L(prefix + "_rsd.npy"), L(src + "_rsd_ops.npy")
    if len(R["real_ops"][0]) != len(R["real"][0]) or not np.allclose(R["real_ops"][0], R["real"][0]):
        sys.exit("k bins of %s_real_ops.npy do not match %s_real.npy (different grid or box?)" % (src, prefix))
    return R


def gls(k, y, var, dbeta, kmax, order):
    """Even polynomial fit with Cov = diag(var) + theory-error kernel."""
    m = (k <= kmax) & np.isfinite(y) & np.isfinite(var) & (var > 0)
    kk, yy = k[m], y[m]
    A = np.stack([kk ** p for p in range(0, order + 1, 2)], axis=1)
    C = np.diag(var[m])
    if dbeta is not None:
        db = dbeta(kk)
        C = C + np.outer(db, db) * np.exp(-0.5 * (kk[:, None] - kk[None, :]) ** 2 / DK ** 2)
    Ci = np.linalg.inv(C)
    F = A.T @ Ci @ A
    cov = np.linalg.inv(F)
    a = cov @ A.T @ Ci @ yy
    r = yy - A @ a
    return a, cov, float(r @ Ci @ r), int(m.sum() - len(a)), m


def mu_moments(nmu):
    e = np.linspace(0.0, 1.0, nmu + 1)
    return (e[1:] ** 3 - e[:-1] ** 3) / (3 * np.diff(e))


def fit_run(R, a, D_ratio):
    """Real-space polynomial fits -> bias, stochastic terms; RSD consistency."""
    re, ops = R["real"], R["real_ops"]
    k, nbar, perr = re[0], re[-1], re[1]
    N, PO = ops[1], ops[2:6]
    out, fits = {}, {}
    b1 = None
    for it in range(2 if not a.no_theory_error else 1):          # b1 sets the theory-error amplitude
        for i, nm in enumerate(NAMES):
            var = perr / (N * PO[i])
            db = None
            if not a.no_theory_error and b1 is not None:
                if i == 0:
                    db = lambda q, b=b1: b * D_ratio ** 4 * (q / 0.45) ** 3.3
                else:
                    db = lambda q, b=b1: b * D_ratio ** 2 * (q / 0.30) ** 1.8
            fits[nm] = gls(k, re[5 + i], var, db, a.kmax, a.order) + (var,)
        b1 = (0.0 if a.real_no_dz else 1.0) + fits["beta1"][0][0]
    a0 = {nm: (fits[nm][0][0], np.sqrt(fits[nm][1][0, 0])) for nm in NAMES}
    s_b1 = a0["beta1"][1]
    out["b1"] = (b1, s_b1)
    out["b2"] = (2 * a0["beta2"][0], 2 * a0["beta2"][1])
    out["b3"] = (6 * a0["beta3"][0], 6 * a0["beta3"][1])
    shift = 0.0 if a.real_no_dz else 0.5
    out["bG2"] = (a0["betaG2"][0] - 2.0 / 7.0 * b1 + shift, np.hypot(a0["betaG2"][1], 2.0 / 7.0 * s_b1))
    c2 = fits["beta1"][0][1]
    out["beta1_k2"] = (c2, np.sqrt(fits["beta1"][1][1, 1]))
    out["chi2"] = dict((nm, (fits[nm][2], fits[nm][3])) for nm in NAMES)
    # stochastic: nbar P_err = 1 + alpha0 + alpha1 (k/kS)^2, Var = 2 P^2 / N
    y = nbar * perr - 1.0
    var = 2.0 * (nbar * perr) ** 2 / N
    m = (k <= a.kmax) & np.isfinite(y)
    A = np.stack([np.ones(m.sum()), (k[m] / KS) ** 2], axis=1)
    W = 1.0 / var[m]
    cov = np.linalg.inv(A.T @ (A * W[:, None]))
    al = cov @ A.T @ (W * y[m])
    out["alpha0"], out["alpha1"] = (al[0], np.sqrt(cov[0, 0])), (al[1], np.sqrt(cov[1, 1]))
    # RSD: alpha2 with alpha0/1 fixed; and the a0 of each beta from a joint fit over mu bins
    if "rsd" in R:
        rs, rops = R["rsd"], R["rsd_ops"]
        ks, Nm, pe_s = rs[0], rops[2], rs[2]
        nmu = pe_s.shape[1]
        m2 = mu_moments(nmu)
        rows, ys, ws = [], [], []
        for j in range(nmu):
            mm = (ks <= a.kmax_rsd) & np.isfinite(pe_s[:, j]) & (Nm[:, j] > 0)
            base = 1.0 + al[0] + al[1] * (ks[mm] / KS) ** 2
            ys.append(nbar * pe_s[mm, j] - base)
            rows.append(m2[j] * (ks[mm] / KS) ** 2)
            ws.append(Nm[mm, j] / (2.0 * (nbar * pe_s[mm, j]) ** 2))
        X, Y, Wt = np.concatenate(rows), np.concatenate(ys), np.concatenate(ws)
        out["alpha2"] = (np.sum(Wt * X * Y) / np.sum(Wt * X * X), 1.0 / np.sqrt(np.sum(Wt * X * X)))
        rsd_a0 = {}
        for i, nm in enumerate(NAMES):
            Ys, As, Vs = [], [], []
            npar = a.order // 2
            for j in range(nmu):
                var_j = pe_s[:, j] / (Nm[:, j] * rops[3 + i][:, j])
                mm = (ks <= a.kmax_rsd) & np.isfinite(rs[6 + i][:, j]) & np.isfinite(var_j) & (var_j > 0)
                block = np.zeros((mm.sum(), 1 + nmu * npar))
                block[:, 0] = 1.0
                for p in range(npar):
                    block[:, 1 + j * npar + p] = ks[mm] ** (2 * (p + 1))
                Ys.append(rs[6 + i][mm, j]); As.append(block); Vs.append(var_j[mm])
            Aall, Yall, Wall = np.vstack(As), np.concatenate(Ys), 1.0 / np.concatenate(Vs)
            cv = np.linalg.inv(Aall.T @ (Aall * Wall[:, None]))
            sol = cv @ Aall.T @ (Wall * Yall)
            rsd_a0[nm] = (sol[0], np.sqrt(cv[0, 0]))
        out["rsd_a0"] = rsd_a0
    return out, fits


def fmt(v, s):
    return "%8.3f +- %-6.3f" % (v, s)


def report(runs, results, a, fn):
    lines = []
    w = lines.append
    w("Bias from even-in-k polynomial fits (a0 + a2 k^2%s) to the real-space transfer functions, k <= %.2f h/Mpc,"
      % (" + a4 k^4" if a.order >= 4 else "", a.kmax))
    w("weighted with the arXiv:1811.10640 covariance (noise P_err/(N P_O)%s). z = %g, f = %.4f."
      % ("" if a.no_theory_error else " + 1-/2-loop theory error, Dk = 0.2", a.z_out, a.f))
    w("Conversion: arXiv:2409.10609 eq. %s. Table III of arXiv:2505.03626 is the reference (MTNG | Astrid)."
      % ("20 (no delta_Z)" if a.real_no_dz else "45 (delta_Z kept)"))
    w("")
    params = ("b1", "b2", "bG2", "b3", "alpha0", "alpha1", "alpha2")
    for (label, _), res in zip(runs, results):
        key = table3_key(label)
        w("=== %s%s" % (label, "" if key is None else "   (compared with %s)" % key))
        w("  %-8s %-22s %-10s %-10s" % ("param", "this run", "MTNG", "Astrid"))
        for p in params:
            if p not in res:
                continue
            ref = ("", "")
            if key is not None and p in TABLE3:
                ref = tuple("%.3g" % TABLE3[p][TABLE3_COLS.index("%s %s" % (key, sim))] for sim in ("MTNG", "Astrid"))
            w("  %-8s %-22s %-10s %-10s" % (p, fmt(*res[p]), ref[0], ref[1]))
        w("  %-8s %-22s (k^2 coefficient of beta_1 [(Mpc/h)^2]; NOT Table III's b_nabla2delta, which is fitted"
          % ("beta1_k2", fmt(*res["beta1_k2"])))
        w("  %-8s %-22s  on top of the one-loop shape)" % ("", ""))
        w("  chi2/dof: " + ", ".join("%s %.1f/%d" % (nm, c[0], c[1]) for nm, c in res["chi2"].items()))
        if "rsd_a0" in res:
            w("  redshift space, k <= %.2f, common a0 over mu bins (should equal real space for beta2, betaG2, "
              "beta3):" % a.kmax_rsd)
            w("    " + ", ".join("%s %.3f +- %.3f (real %.3f)" % (nm, v[0], v[1], res["real_a0"][nm])
                                 for nm, v in res["rsd_a0"].items()))
        w("")
    w("Not from polynomial fits: b_Gamma3, b_nabla2delta (one-loop shape of beta_1), c_mu2, b4 (one-loop RSD shape).")
    text = "\n".join(lines)
    print(text)
    with open(fn + "_table.txt", "w") as fh:
        fh.write(text + "\n")
    with open(fn + "_table.csv", "w") as fh:
        fh.write("run,param,value,error,MTNG,Astrid\n")
        for (label, _), res in zip(runs, results):
            key = table3_key(label)
            for p in params + ("beta1_k2",):
                if p not in res:
                    continue
                ref = ["", ""]
                if key is not None and p in TABLE3:
                    ref = [str(TABLE3[p][TABLE3_COLS.index("%s %s" % (key, s))]) for s in ("MTNG", "Astrid")]
                fh.write("%s,%s,%.6g,%.6g,%s,%s\n" % (label, p, res[p][0], res[p][1], ref[0], ref[1]))
    print("wrote %s_table.txt and %s_table.csv" % (fn, fn))


def plot(runs, allfits, a, fn):
    plt = P._pyplot()
    if plt is None:
        return
    n = len(runs)
    fig, axes = plt.subplots(n, 4, figsize=(17, 3.3 * n + 0.3), squeeze=False)
    cols = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    for r, ((label, R), fits) in enumerate(zip(runs, allfits)):
        k = R["real"][0]
        for i, nm in enumerate(NAMES):
            ax = axes[r, i]
            coef, cov, chi2, dof, m, var = fits[nm]
            y = R["real"][5 + i]
            show = k <= max(1.0, a.kmax * 2)
            ax.errorbar(k[show], y[show], yerr=np.sqrt(var[show]), fmt="o", ms=2.5, lw=0.8, color=cols[i],
                        alpha=0.85, label="data, $\\sqrt{P_{\\rm err}/(N_k P_O)}$")
            kk = np.linspace(0.0, max(k[show]), 300)
            poly = sum(c * kk ** (2 * p) for p, c in enumerate(coef))
            fit = kk <= a.kmax
            ax.plot(kk[fit], poly[fit], "-", color="k", lw=1.5, label="fit, $k\\leq%.2f$" % a.kmax)
            ax.plot(kk[~fit], poly[~fit], "--", color="k", lw=1.0)
            ax.axhline(coef[0], color="0.5", lw=0.8, ls=":")
            ax.errorbar([k[show][0] * 0.8], [coef[0]], yerr=[np.sqrt(cov[0, 0])], fmt="s", color="k", ms=5,
                        capsize=3)
            ax.axvline(a.kmax, color="0.6", lw=0.8, ls="-.")
            ax.set_xscale("log")
            ylo, yhi = np.nanpercentile(y[show & (k > 0.03)], [2, 98])
            pad = 0.35 * (yhi - ylo + 1e-9)
            ax.set_ylim(min(ylo, coef[0]) - pad, max(yhi, coef[0]) + pad)
            ax.set_xlabel(P.KLABEL, fontsize=10)
            ax.text(0.05, 0.92, LABELS[i], transform=ax.transAxes, fontsize=13, va="top")
            ax.text(0.95, 0.06, "$a_0 = %.3f \\pm %.3f$\n$\\chi^2$/dof = %.1f/%d" % (coef[0], np.sqrt(cov[0, 0]), chi2, dof),
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
            if i == 0:
                ax.set_ylabel(label, fontsize=12)
            if r == 0 and i == 0:
                ax.legend(loc="upper center", fontsize=8, frameon=False)
    fig.suptitle("real-space transfer functions: even polynomial fits with the arXiv:1811.10640 covariance",
                 fontsize=13)
    fig.tight_layout()
    P.save_figure(fig, fn + "_fits.png", 130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--run", nargs=2, action="append", metavar=("PREFIX", "LABEL"), required=True)
    ap.add_argument("--ops", default=None, help="PREFIX of the _real_ops/_rsd_ops files to use for every run")
    ap.add_argument("--z-out", type=float, required=True, help="redshift of the runs (for f and D/D0)")
    ap.add_argument("--omega-m", type=float, default=P.OMEGA_M)
    ap.add_argument("--omega-r", type=float, default=P.OMEGA_R)
    ap.add_argument("--kmax", type=float, default=0.4, help="real-space fit range (default 0.4)")
    ap.add_argument("--kmax-rsd", type=float, default=0.2, help="redshift-space fit range (default 0.2)")
    ap.add_argument("--order", type=int, choices=(2, 4), default=4, help="highest even power (default 4)")
    ap.add_argument("--no-theory-error", action="store_true", help="noise term of the covariance only")
    ap.add_argument("--real-no-dz", action="store_true", help="real-space results fitted without delta_Z")
    ap.add_argument("--out", default="bias")
    a = ap.parse_args()
    a.f = P.growth(a.z_out + 1.0, a.z_out, a.omega_m, 0.0, a.omega_r)[1]
    D_ratio = 1.0 / P.growth(a.z_out, 0.0, a.omega_m, 0.0, a.omega_r)[0]            # D(z)/D(0)
    runs = [(label, load(prefix, a.ops)) for prefix, label in a.run]
    results, allfits = [], []
    for label, R in runs:
        res, fits = fit_run(R, a, D_ratio)
        res["real_a0"] = dict((nm, fits[nm][0][0]) for nm in NAMES)
        results.append(res)
        allfits.append(fits)
    report(runs, results, a, a.out)
    plot(runs, allfits, a, a.out)


if __name__ == "__main__":
    main()

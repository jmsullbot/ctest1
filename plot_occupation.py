#!/usr/bin/env python3
"""
Plot QSO HOD occupation functions <N_cen>(M), <N_sat>(M), <N_tot>(M) vs halo mass.

QSO HOD form (Yuan+2023 / AbacusHOD):

    <N_cen>(M) = p_max/2 · erfc[ ln(M_cut / M) / (√2 σ) ]
                 where  p_max = 10^{log10_f_ic}

    <N_sat>(M) = [(M − κ M_cut) / M1]^α   for M > κ M_cut, else 0

Note: <N_sat> is NOT modulated by <N_cen>.

Can be used as a standalone script or imported as a module.

Usage
-----
    python plot_occupation.py \\
        --logM_cut 12.7 --logM1 14.5 --sigma 0.5 \\
        --alpha 1.0 --kappa 0.5 --log10_f_ic -1.35 \\
        --output occupation.png
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import erfc


# ---------------------------------------------------------------------------
# Analytic HOD functions
# ---------------------------------------------------------------------------

def n_cen(
    M: np.ndarray,
    logM_cut: float,
    sigma: float,
    log10_f_ic: float,
) -> np.ndarray:
    """Mean central QSO occupation as a function of halo mass.

    Parameters
    ----------
    M : array_like
        Halo mass [M☉/h].
    logM_cut : float
        log10 of the minimum halo mass for central occupation [log10 M☉/h].
    sigma : float
        Width of the complementary error function step.
    log10_f_ic : float
        log10 of the incompleteness / max central occupation probability.

    Returns
    -------
    ndarray, same shape as M.
    """
    M_cut = 10.0 ** logM_cut
    p_max = 10.0 ** log10_f_ic
    return (p_max / 2.0) * erfc(
        np.log(M_cut / np.asarray(M, dtype=float)) / (np.sqrt(2.0) * sigma)
    )


def n_sat(
    M: np.ndarray,
    logM_cut: float,
    logM1: float,
    alpha: float,
    kappa: float,
) -> np.ndarray:
    """Mean satellite QSO occupation as a function of halo mass.

    Parameters
    ----------
    M : array_like
        Halo mass [M☉/h].
    logM_cut : float
        log10 of the minimum halo mass [log10 M☉/h].
    logM1 : float
        log10 of the satellite characteristic mass [log10 M☉/h].
    alpha : float
        Power-law slope.
    kappa : float
        Satellite truncation parameter; occupation is zero for M < κ M_cut.

    Returns
    -------
    ndarray, same shape as M.

    Note
    ----
    NOT modulated by <N_cen>.  QSO satellite occupation is treated
    independently of the central (unlike the LRG Zheng+2007 form).
    """
    M_cut = 10.0 ** logM_cut
    M1    = 10.0 ** logM1
    x = (np.asarray(M, dtype=float) - kappa * M_cut) / M1
    # Clamp to zero before raising to avoid NaN for negative x with fractional alpha
    return np.where(x > 0.0, np.maximum(x, 0.0) ** alpha, 0.0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_occupation(
    params_list: list[dict],
    labels: list[str] | None = None,
    output_file: str | None = None,
    M_range: tuple[float, float] = (1e11, 1e16),
    n_points: int = 300,
) -> tuple:
    """
    Plot <N_cen>, <N_sat>, <N_tot> vs halo mass for one or more parameter sets.

    Parameters
    ----------
    params_list : list of dict
        Each dict must contain: logM_cut, logM1, sigma, alpha, kappa, log10_f_ic.
    labels : list of str, optional
        Legend label for each parameter set.
    output_file : str, optional
        Save figure here.  If None the figure is returned but not displayed.
    M_range : (float, float)
        Halo mass range [M☉/h].
    n_points : int
        Number of mass grid points.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import matplotlib.pyplot as plt  # deferred so callers can set the backend first

    M      = np.logspace(np.log10(M_range[0]), np.log10(M_range[1]), n_points)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, p in enumerate(params_list):
        c   = colors[i % len(colors)]
        lbl = labels[i] if labels else f"set {i}"
        nc  = n_cen(M, p["logM_cut"], p["sigma"], p["log10_f_ic"])
        ns  = n_sat(M, p["logM_cut"], p["logM1"], p["alpha"], p["kappa"])
        ax.loglog(M, nc + ns, color=c, lw=2,              label=f"{lbl}  tot")
        ax.loglog(M, nc,      color=c, lw=1.5, ls="--",   label=f"{lbl}  cen")
        ax.loglog(M, ns,      color=c, lw=1.5, ls=":",    label=f"{lbl}  sat")

    ax.set_xlabel(r"Halo mass $M$  [$h^{-1}\,M_\odot$]", fontsize=12)
    ax.set_ylabel(r"$\langle N \rangle (M)$", fontsize=12)
    ax.set_title("QSO HOD occupation functions", fontsize=13)
    ax.legend(fontsize=8, ncol=min(2, len(params_list)))
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=150)
        print(f"Saved to {output_file}")

    return fig, ax


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--logM_cut",   type=float, default=12.7)
    parser.add_argument("--logM1",      type=float, default=14.5)
    parser.add_argument("--sigma",      type=float, default=0.5)
    parser.add_argument("--alpha",      type=float, default=1.0)
    parser.add_argument("--kappa",      type=float, default=0.5)
    parser.add_argument("--log10_f_ic", type=float, default=-1.35)
    parser.add_argument(
        "--output", default="occupation.png",
        help="Output figure path  (default: occupation.png)",
    )
    return parser.parse_args()


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless — writing to file, no display needed

    args = _parse_args()
    p    = {k: getattr(args, k)
            for k in ("logM_cut", "logM1", "sigma", "alpha", "kappa", "log10_f_ic")}
    lbl  = (
        rf"$\log M_{{cut}}$={p['logM_cut']}, $\log M_1$={p['logM1']}, "
        rf"$\sigma$={p['sigma']}, $\alpha$={p['alpha']}, "
        rf"$\kappa$={p['kappa']}, $\log_{{10}} f_{{ic}}$={p['log10_f_ic']}"
    )
    plot_occupation([p], labels=[lbl], output_file=args.output)


if __name__ == "__main__":
    main()

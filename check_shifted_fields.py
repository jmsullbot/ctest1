#!/usr/bin/env python
"""Check a shifted-fields .npy for integrity and against a reference grid.

    python check_shifted_fields.py FILE [REF] [--box 2000]

FILE and REF hold arrays of shape (nfields, N, N, N), e.g.

    FILE = shifted_fields_real_N_512_c_0.npy
    REF  = shifted_fields_real_N_256_c_0.npy   (what every 256 result used)

Integrity (both files): the file loads (a truncated .npy fails here), every
value is finite, and no slab along any axis is empty (min/median slab std).

Consistency (needs REF): FFT every field of both files, keep the Fourier
modes the two grids share (identical wavevectors, |n_i| < N_ref/2), and per
|k| bin print

    r     = <Re a b*> / sqrt(<|a|^2> <|b|^2>)   cross-correlation coefficient
    ratio = <|a|^2> / <|b|^2>                   power ratio FILE / REF

Field names follow lrgs.py's unpacking: real space d1, d2, dG2, d3;
redshift space dz, d1, d2, dG2, dG2par, d3.

If FILE is the same field on a finer grid (same IC phases, same code):
  - r ~ 1 at low k, falling toward REF's Nyquist (aliasing on the coarse
    grid);
  - for fields linear in delta_1 at low k (d1; dz and d1 in redshift space)
    the power ratio is ~1 in the lowest k-bin (windows and smoothing move it
    by ~2% there), rising toward REF's Nyquist;
  - the nonlinear operators' amplitudes depend on the grid's k-cutoff, so
    their ratio is NOT expected to be 1. Example: delta^3 ~ 3 sigma^2 delta
    on large scales, and sigma^2 grows with the cutoff. In a synthetic test
    with cutoffs 0.8 vs 0.4 h/Mpc (no smoothing), delta^2 / G2 / delta^3 gave
    r = 0.93 / 0.98 / 0.98 at k < 0.1 but power ratios 1.1 / 0.95 / ~5.
    Amplitudes are therefore tested only for the linear fields.
It is wrong if r ~ 0 (other phases/seed, transposed axes), if a linear
field has r ~ 1 but ratio != 1 (e.g. D(z) applied differently: a 0.769
growth error gives 0.59), or if the low-k r matrix peaks off the diagonal
(fields stored in a different order).

Exit code 0 = no problems found, 1 = something was flagged.
"""
import argparse
import os
import sys

import numpy as np

try:
    import scipy.fft as _sfft

    def rfftn(x):
        return _sfft.rfftn(x, workers=-1)
except ImportError:          # numpy fallback: single-threaded, still fine
    def rfftn(x):
        return np.fft.rfftn(x)

EDGES = np.array([0, .05, .125, .25, .375, .5, .625, .75, .875, 1.0])  # x k_Nyq(REF)
LOWK_MATRIX = 0.25   # k < 0.25 k_Nyq(REF) for the r matrix and the r verdict
LOWK_RATIO = 0.05    # k < 0.05 k_Nyq(REF) (first bin) for the amplitude test

# lrgs.py's order; fields that are linear in delta_1 at low k get the
# amplitude test.
NAMES = {4: (["d1", "d2", "dG2", "d3"], {"d1"}),
         6: (["dz", "d1", "d2", "dG2", "dG2par", "d3"], {"dz", "d1"})}


def load(path):
    try:
        a = np.load(path, mmap_mode="r")
    except Exception as e:                       # truncated, or not an .npy
        print(f"FAIL  {path}: cannot load ({e})")
        sys.exit(1)
    if a.ndim != 4 or not a.shape[1] == a.shape[2] == a.shape[3]:
        print(f"FAIL  {path}: expected shape (nfields, N, N, N), got {a.shape}")
        sys.exit(1)
    return a


def integrity(path, a):
    """Per-field stats. Returns True if nothing looks broken."""
    print(f"{path}\n  shape {a.shape} {a.dtype}, {os.path.getsize(path):,} bytes")
    names = NAMES.get(a.shape[0], ([f"f{i}" for i in range(a.shape[0])], set()))[0]
    ok = True
    for i in range(a.shape[0]):
        f = np.asarray(a[i])
        finite = bool(np.isfinite(f).all())
        slab = np.concatenate([f.std(axis=ax, dtype=np.float64)
                               for ax in ((1, 2), (0, 2), (0, 1))])
        med = float(np.median(slab))
        worst = float(slab.min()) / med if med > 0 else 0.0
        bad = (not finite) or worst < 0.5
        ok &= not bad
        print(f"  {names[i]:>6}: mean {f.mean(dtype=np.float64):+.3e}  "
              f"std {f.std(dtype=np.float64):.4f}  "
              f"min/median slab std {worst:.2f}  "
              f"{'finite' if finite else 'NON-FINITE VALUES'}"
              f"{'   <-- PROBLEM' if bad else ''}")
    return ok


def shared_modes(a, nref, box):
    """Fourier modes of every field of `a` at the wavevectors an nref^3 grid
    resolves (Nyquist planes excluded). Returns (list of 1-D complex arrays,
    |k| of those modes, k_Nyq of the reference grid)."""
    n = a.shape[1]
    h = nref // 2
    m = np.r_[0:h, -(h - 1):0]                  # shared integer wavenumbers
    kf = 2 * np.pi / box
    kxy, kz = kf * m, kf * np.arange(h)
    kk = np.sqrt(kxy[:, None, None] ** 2 + kxy[None, :, None] ** 2
                 + kz[None, None, :] ** 2)
    knyq = kf * h
    sel = (kk > 0) & (kk < knyq)
    out = []
    for i in range(a.shape[0]):
        F = rfftn(np.asarray(a[i], dtype=np.float32))
        sub = F[np.ix_(m % n, m % n, np.arange(h))][sel] / float(n) ** 3
        out.append(sub.astype(np.complex64))
        del F
    return out, kk[sel], knyq


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file")
    ap.add_argument("ref", nargs="?")
    ap.add_argument("--box", type=float, default=2000.0, help="box size [Mpc/h]")
    args = ap.parse_args()

    a = load(args.file)
    ok = integrity(args.file, a)
    if not args.ref:
        print("OK (integrity only)" if ok else "PROBLEMS FOUND")
        return 0 if ok else 1

    b = load(args.ref)
    ok &= integrity(args.ref, b)
    nref = min(a.shape[1], b.shape[1])
    if a.shape[0] != b.shape[0]:
        print(f"NOTE: {a.shape[0]} fields vs {b.shape[0]} in the reference")

    A, kk, knyq = shared_modes(a, nref, args.box)
    B, _, _ = shared_modes(b, nref, args.box)
    edges = EDGES * knyq
    nb = len(edges) - 1
    which = np.digitize(kk, edges[1:-1])
    cnt = np.bincount(which, minlength=nb)

    def binned(x):
        return np.bincount(which, weights=x, minlength=nb)

    nf = min(len(A), len(B))
    names, linear = NAMES.get(len(A), ([f"f{i}" for i in range(len(A))], set()))
    if len(A) != len(B):
        linear = set()          # can't tell which field is which
    r = np.empty((nb, nf))
    ratio = np.empty((nb, nf))
    for i in range(nf):
        pa, pb = binned(np.abs(A[i]) ** 2), binned(np.abs(B[i]) ** 2)
        pab = binned((A[i] * np.conj(B[i])).real)
        r[:, i], ratio[:, i] = pab / np.sqrt(pa * pb), pa / pb

    print(f"\nFILE vs REF on the {nref}^3 modes both resolve "
          f"(k_Nyq = {knyq:.3f} h/Mpc), per field: r  ratio")
    print("  k [h/Mpc]      modes  "
          + "".join(f"| {names[i]:>6}: r ratio " for i in range(nf)))
    for j in range(nb):
        print(f"  {edges[j]:.3f}-{edges[j + 1]:.3f} {cnt[j]:9d}  "
              + "".join(f"| {r[j, i]:6.3f} {ratio[j, i]:6.3f} " for i in range(nf)))

    low = kk < LOWK_MATRIX * knyq
    R = np.empty((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            num = np.sum((A[i][low] * np.conj(B[j][low])).real, dtype=np.float64)
            den = np.sqrt(np.sum(np.abs(A[i][low]) ** 2, dtype=np.float64)
                          * np.sum(np.abs(B[j][low]) ** 2, dtype=np.float64))
            R[i, j] = num / den
    print(f"\nr matrix at k < {LOWK_MATRIX * knyq:.3f} h/Mpc "
          "(rows FILE fields, columns REF fields; should peak on the diagonal):")
    print("          " + " ".join(f"{n:>7}" for n in names[:len(B)]))
    for i in range(len(A)):
        print(f"  {names[i]:>6}: " + " ".join(f"{R[i, j]:+7.3f}" for j in range(len(B))))

    print(f"\nverdict per field (r at k < {LOWK_MATRIX * knyq:.3f}, "
          f"power ratio at k < {LOWK_RATIO * knyq:.3f} h/Mpc):")
    lowr = kk < LOWK_RATIO * knyq
    for i in range(nf):
        amp = (np.sum(np.abs(A[i][lowr]) ** 2, dtype=np.float64)
               / np.sum(np.abs(B[i][lowr]) ** 2, dtype=np.float64))
        best = int(np.argmax(np.abs(R[i])))
        rr = R[i, i]
        lin = names[i] in linear
        if best != i and abs(R[i, best]) > abs(rr) + 0.1:
            v, bad = f"FIELD ORDER? correlates best with REF {names[best]}", True
        elif rr < 0.5:
            v, bad = "MISMATCH (different phases / axes / definition)", True
        elif lin and not 0.9 <= amp <= 1.1:
            v, bad = "AMPLITUDE DIFFERS (growth / normalisation?)", True
        elif rr >= 0.9:
            v, bad = "consistent", False
        else:
            v, bad = "partially correlated (possible for nonlinear operators)", False
        ok &= not bad
        note = "" if lin else "  (nonlinear: ratio not tested)"
        print(f"  {names[i]:>6}: r = {rr:.3f}, ratio {amp:.3f}  ->  {v}{note}")
    print("\nOK: no problems found" if ok else "\nPROBLEMS FOUND (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

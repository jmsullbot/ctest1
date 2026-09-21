#!/usr/bin/env python3
"""
Verify downloaded AbacusSummit IC files and report what is in them.

For each directory given:
  1. checks every file against checksums.crc32 (POSIX `cksum` format:
     "<crc> <size> <name>"), if present;
  2. opens each ic_*.asdf, prints the box, cosmology, GrowthTable at z=0.5,
     and the shape/dtype of every array (so you can see the grid N and whether
     the file is density or displacement);
  3. prints the physical cell size and Nyquist k for that grid.

Requires abacusutils (registers the Blosc ASDF codec) -- run in the
`abacus` env.

    python check_ics.py $PSCRATCH/abacus_ics_000 $PSCRATCH/small_abacus_ics_000
"""

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def verify_checksums(d: Path) -> None:
    cs = d / "checksums.crc32"
    if not cs.exists():
        print("  (no checksums.crc32 -- skipping verification)")
        return
    expected = {}
    for line in cs.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            expected[Path(parts[-1]).name] = (parts[0], parts[1])
    for name, (crc, size) in sorted(expected.items()):
        f = d / name
        if not f.exists():
            print(f"  MISSING   {name}")
            continue
        out = subprocess.run(["cksum", str(f)], capture_output=True, text=True).stdout.split()
        got = (out[0], out[1]) if len(out) >= 2 else ("?", "?")
        status = "ok" if got == (crc, size) else "MISMATCH"
        print(f"  {status:9s} {name:24s} crc {got[0]:>12s} size {got[1]:>14s}"
              + ("" if status == "ok" else f"   expected crc {crc} size {size}"))


def walk_arrays(node, prefix=""):
    """Yield (path, array) for every ndarray-like leaf in an ASDF tree."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk_arrays(v, f"{prefix}/{k}")
    elif hasattr(node, "shape") and hasattr(node, "dtype"):
        yield prefix, node


def inspect(f: Path) -> None:
    import asdf
    try:
        import abacusnbody  # noqa: F401  (registers the 'blsc' codec)
    except ImportError:
        print("  WARNING: abacusutils not importable; Blosc-compressed arrays will fail to read")
    with asdf.open(f, lazy_load=True) as af:
        tree = dict(af.tree)
        hdr = tree.get("header", {})
        L = float(hdr.get("BoxSize", np.nan))
        print(f"  {f.name}")
        print(f"    SimName        : {hdr.get('SimName', '?')}")
        print(f"    BoxSize        : {L:g} Mpc/h    NP: {hdr.get('NP', '?')}    z_init: {hdr.get('InitialRedshift', '?')}")
        print(f"    Omega_M, h     : {hdr.get('Omega_M', '?')}, {hdr.get('H0', '?')}")
        gt = hdr.get("GrowthTable")
        if gt:
            key = next((k for k in gt if abs(float(k) - 0.5) < 1e-6), None)
            print(f"    D(z=0.5)/D(zi) : {gt[key] if key is not None else 'z=0.5 not in table'}")
        for path, arr in walk_arrays({k: v for k, v in tree.items() if k not in ("asdf_library", "history")}):
            shape = tuple(arr.shape)
            n = shape[0] if shape else 0
            line = f"    {path:32s} shape {shape}  dtype {arr.dtype}"
            if len(shape) >= 3 and n and np.isfinite(L):
                cell = L / n
                line += f"   cell {cell:.3f} Mpc/h   k_Nyq {np.pi / cell:.2f} h/Mpc"
            print(line)


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or [Path(".")]
    for d in dirs:
        print("=" * 72); print(d); print("=" * 72)
        print("--- checksums ---"); verify_checksums(d)
        print("--- contents ---")
        for f in sorted(d.glob("ic_*.asdf")):
            try:
                inspect(f)
            except Exception as exc:  # keep going; report the file
                print(f"  {f.name}: ERROR {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    main()

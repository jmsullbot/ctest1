#!/usr/bin/env python
"""Can the active Python environment run SCRIPT?  Checks without running it.

    python check_generator_env.py my_generate_shifted_fields.py [--ic FILE]

Run it on a login node, from the directory you will submit the job from,
with the env the job will use already activated.

1. Executes ONLY the script's top-level `sys.path` edits and import
   statements, in order (so `from lib.tng_lib import *` really imports
   nbodykit), and reports every import as ok or failed. It keeps going, so
   you see all the missing modules at once.
2. If the imports pulled in asdf, reads a few values from an Abacus IC file
   (default $PSCRATCH/abacus_ics_000/ic_dens_N576.asdf, the smallest; a
   few seconds). Abacus files can use Blosc ('blsc') compression, which
   plain asdf cannot decode without abacusutils installed, so an env can
   pass every import and still fail on the first read.

Exit code 0 = everything passed.
"""
import argparse
import ast
import os
import sys


def run_imports(script):
    src = open(script).read()
    tree = ast.parse(src, script)
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))  # as `python script.py`
    ns = {"__name__": "__check_generator_env__", "__file__": os.path.abspath(script)}
    ok = True
    for stmt in tree.body:
        is_import = isinstance(stmt, (ast.Import, ast.ImportFrom))
        is_path_edit = (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                        and "sys.path" in (ast.get_source_segment(src, stmt.value.func) or ""))
        if not (is_import or is_path_edit):
            continue
        code = ast.get_source_segment(src, stmt) or "<statement>"
        try:
            exec(compile(ast.Module(body=[stmt], type_ignores=[]), script, "exec"), ns)
            if is_import:
                print(f"  ok       line {stmt.lineno:3d}: {code}")
        except BaseException as e:  # ImportError, or anything raised at import time
            ok = False
            print(f"  FAILED   line {stmt.lineno:3d}: {code}\n"
                  f"           -> {type(e).__name__}: {e}")
    return ok


def find_3d(node):
    if isinstance(node, dict):
        for v in node.values():
            hit = find_3d(v)
            if hit is not None:
                return hit
    elif hasattr(node, "shape") and len(getattr(node, "shape", ())) == 3:
        return node
    return None


def read_ic(path):
    import numpy as np
    import asdf
    print(f"\nreading a few values from {path}")
    try:
        with asdf.open(path, lazy_load=True) as af:
            arr = find_3d({k: v for k, v in af.tree.items()
                           if k not in ("asdf_library", "history")})
            if arr is None:
                print("  no 3-D array found in the file; nothing to test")
                return True
            vals = np.asarray(arr[0, 0, :4])
        print(f"  ok       shape {tuple(arr.shape)}, first values {vals}")
        return True
    except Exception as e:
        print(f"  FAILED   {type(e).__name__}: {e}")
        if "blsc" in str(e):
            print("           -> Blosc-compressed: this env also needs abacusutils "
                  "(see the install line printed at the end)")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("script")
    ap.add_argument("--ic", default=os.path.join(os.environ.get("PSCRATCH", "."),
                                                 "abacus_ics_000", "ic_dens_N576.asdf"))
    args = ap.parse_args()

    print(f"python : {sys.executable}  (Python {sys.version.split()[0]})")
    print(f"env    : {os.environ.get('CONDA_DEFAULT_ENV', '?')}\n\nimports in {args.script}:")
    ok = run_imports(args.script)

    if "asdf" in sys.modules:
        if os.path.exists(args.ic):
            ok &= read_ic(args.ic)
        else:
            print(f"\n(skipping the IC read test: {args.ic} not found; pass --ic FILE)")

    try:
        import numpy
        npv = numpy.__version__
    except ImportError:
        npv = None
    if ok:
        print("\nALL OK: this env can run the script.")
    else:
        print("\nNOT READY. Install what failed WITHOUT letting pip change numpy "
              "(nbodykit's compiled parts depend on it):")
        pin = f' "numpy=={npv}"' if npv else ""
        print(f"    python -m pip install <package>{pin}")
        print("  e.g. asdf for 'No module named asdf'; abacusutils for the 'blsc' read error.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

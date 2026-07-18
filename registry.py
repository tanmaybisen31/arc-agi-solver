"""Assemble the full detector ensemble: base detectors + everything in gen/*.py.

Each gen/<name>.py must expose a module-level list `DETECTORS` of detector
functions with signature det(train)->transform_fn|None (see detectors.py).
Broken plugins are skipped, never crash the ensemble.
"""
import os, importlib
import detectors as base

def load_all(verbose=False):
    all_dets = list(base.DETECTORS)
    gendir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen")
    if os.path.isdir(gendir):
        for fn in sorted(os.listdir(gendir)):
            if not fn.endswith(".py") or fn.startswith("_"):
                continue
            mod = "gen." + fn[:-3]
            try:
                m = importlib.import_module(mod)
                dets = list(getattr(m, "DETECTORS", []))
                all_dets += dets
                if verbose:
                    print(f"  + {mod}: {len(dets)} detectors")
            except Exception as e:
                if verbose:
                    print(f"  ! skip {mod}: {e}")
    return all_dets

ALL = load_all()

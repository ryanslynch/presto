"""Test the fftfit method selector (classic vs aarchiba) and the aarchiba adapter.

Verifies: the two methods agree on well-determined fits; the aarchiba path returns a
fully-populated, finite 7-tuple (unlike a nan-filled adapter); aarchiba fits the
symmetric/low-harmonic (pure-sine) profiles the classic method bails on; and an unknown
method raises. Run with pytest, or:

    python python/fftfit_src/test_selector.py
"""

import os
import warnings
import numpy as np

import fftfit_py

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "fftfit_reference.npz")


def _cases():
    return np.load(REF, allow_pickle=True)


def test_methods_agree_on_well_determined_fits():
    d = _cases()
    worst = 0.0
    n = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in d["meta"]:
            if r["eshift"] >= 999 or r["eshift"] > 0.2 or r["snr"] <= 0:
                continue
            i, nb = r["idx"], r["nbins"]
            c = fftfit_py.fftfit(d[f"prof_{i}"], d[f"amp_{i}"], d[f"pha_{i}"], code="classic")
            a = fftfit_py.fftfit(d[f"prof_{i}"], d[f"amp_{i}"], d[f"pha_{i}"], code="aarchiba")
            dshift = abs((a[0] - c[0] + nb / 2) % nb - nb / 2)   # wrapped bin difference
            worst = max(worst, dshift)
            n += 1
    print(f"classic vs aarchiba: {n} well-determined cases, worst |dshift| = {worst:.3g} bins")
    assert worst < 0.3, f"methods disagree by {worst} bins"


def test_aarchiba_tuple_is_fully_populated():
    d = _cases()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in list(d["meta"])[:50]:
            i = r["idx"]
            out = fftfit_py.fftfit(d[f"prof_{i}"], d[f"amp_{i}"], d[f"pha_{i}"], code="aarchiba")
            assert all(np.isfinite(x) for x in out), f"non-finite output: {out}"


def test_aarchiba_fits_what_classic_bails_on():
    d = _cases()
    checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in d["meta"]:
            if r.get("shape") != "sine" or r.get("amp_name") != "strong":
                continue
            i = r["idx"]
            args = (d[f"prof_{i}"], d[f"amp_{i}"], d[f"pha_{i}"])
            c = fftfit_py.fftfit(*args, code="classic")
            a = fftfit_py.fftfit(*args, code="aarchiba")
            assert c[1] >= 999, "expected classic to bail on a pure sinusoid"
            assert a[1] < 1.0 and np.isfinite(a[0]), f"aarchiba failed to fit sine: {a}"
            checked += 1
    assert checked > 0
    print(f"aarchiba fit {checked} strong pure-sine cases that classic bails on")


def test_unknown_method_raises():
    d = _cases()
    r = d["meta"][0]
    i = r["idx"]
    try:
        fftfit_py.fftfit(d[f"prof_{i}"], d[f"amp_{i}"], d[f"pha_{i}"], code="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown method")


if __name__ == "__main__":
    test_methods_agree_on_well_determined_fits()
    test_aarchiba_tuple_is_fully_populated()
    test_aarchiba_fits_what_classic_bails_on()
    test_unknown_method_raises()
    print("selector + aarchiba adapter OK.")

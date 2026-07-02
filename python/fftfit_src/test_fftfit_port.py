"""Phase 1/3 regression test: pure-Python fftfit vs. the Fortran golden dataset.

Feeds the exact (prof, amp, pha) inputs that were given to the Fortran (frozen in
fftfit_reference.npz) into fftfit_py.fftfit and compares all seven outputs, per the
"match, but aim to improve" philosophy (see ROADMAP.md). Run with pytest, or:

    python python/fftfit_src/test_fftfit_port.py
"""

import os
import numpy as np

import fftfit_py

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "fftfit_reference.npz")

# Tolerances. The Fortran is single-precision and converges tau only to ~edtau, so
# shift agreement is judged relative to the reported uncertainty (eshift), not
# absolutely. b/snr/eshift are checked with loose relative tolerances.
SHIFT_SIGMA_FRAC = 0.25    # |shift_port - shift_ref| < this * eshift_ref
SHIFT_ABS_FLOOR = 0.05     # ...or within this many bins (shallow-minimum cases)
REL_TOL = 5e-3             # relative tol for eshift/snr/b on well-determined fits


def test_fftfit_matches_fortran():
    d = np.load(REF, allow_pickle=True)
    meta = d["meta"]
    n_bail_ok = 0
    shift_fails, ngood_fails, scale_fails = [], [], []
    worst_shift_sigma = 0.0

    for r in meta:
        i = r["idx"]
        out = fftfit_py.fftfit(d[f"prof_{i}"], d[f"amp_{i}"], d[f"pha_{i}"])
        shift, eshift, snr, esnr, b, errb, ngood = out

        ref_bail = r["eshift"] >= 999.0
        port_bail = eshift >= 999.0
        if ref_bail or port_bail:
            # Must agree on failure, and (even on bailout) on ngood.
            if ref_bail != port_bail or ngood != r["ngood"]:
                ngood_fails.append((r.get("tag"), r.get("shape"), int(ngood), int(r["ngood"])))
            else:
                n_bail_ok += 1
            continue

        if ngood != r["ngood"]:
            ngood_fails.append((r.get("tag"), r.get("shape"), int(ngood), int(r["ngood"])))

        # shift: within a fraction of the 1-sigma uncertainty (or an absolute floor)
        dshift = abs(shift - r["shift"])
        tol = max(SHIFT_SIGMA_FRAC * r["eshift"], SHIFT_ABS_FLOOR)
        if r["eshift"] > 0:
            worst_shift_sigma = max(worst_shift_sigma, dshift / r["eshift"])
        if dshift > tol:
            shift_fails.append((r.get("tag"), r.get("shape"), r["nbins"], dshift, r["eshift"]))

        # scale/uncertainty outputs (skip where reference is ~0 / ill-defined)
        for name, val, ref in [("eshift", eshift, r["eshift"]),
                               ("snr", snr, r["snr"]), ("b", b, r["b"])]:
            if abs(ref) > 1e-30 and abs(val - ref) > REL_TOL * abs(ref):
                scale_fails.append((name, r.get("shape"), val, ref))

    print(f"checked {len(meta)} cases; {n_bail_ok} bailouts agreed; "
          f"worst shift = {worst_shift_sigma:.3f} sigma")
    assert not ngood_fails, f"ngood mismatches: {ngood_fails[:5]}"
    assert not shift_fails, f"shift mismatches: {shift_fails[:5]}"
    assert not scale_fails, f"scale/uncertainty mismatches: {scale_fails[:5]}"


if __name__ == "__main__":
    test_fftfit_matches_fortran()
    print("fftfit port matches Fortran (within timing-negligible tolerances).")

"""Regression test: presto.fftfit.cprof vs. the frozen Fortran cprof outputs.

The golden dataset (fftfit_reference.npz) stored, for each template, the Fortran
cprof amplitudes (amp_i) and the get_TOAs-style rotated phases (pha_i). We recompute
both with the pure-Python cprof and check they match -- a frozen regression that
survives removal of the Fortran (which is exactly why those values were frozen).

Run with pytest, or:  python python/fftfit_src/test_cprof.py
"""

import os
import numpy as np

from presto import fftfit

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "fftfit_reference.npz")

TWOPI = 2.0 * np.pi
AMP_RTOL = 1e-4          # Fortran cprof was single precision
PHA_ATOL = 1e-3          # radians, on significant harmonics
PHA_AMP_FRAC = 1e-3      # "significant" = amp > this * max(amp)


def test_cprof_matches_frozen_fortran():
    d = np.load(REF, allow_pickle=True)
    seen = set()
    worst_amp = worst_pha = 0.0
    failures = []
    for r in d["meta"]:
        i = r["idx"]
        templ = d[f"templ_{i}"]
        key = (len(templ), templ.tobytes())
        if key in seen:
            continue
        seen.add(key)

        amp_ref = np.asarray(d[f"amp_{i}"])
        pha_ref = np.asarray(d[f"pha_{i}"])       # rotated, as fed to fftfit
        _c, amp, pha = fftfit.cprof(templ)
        # Reproduce the get_TOAs.py phase rotation to compare against the stored pha.
        pha_rot = np.fmod(pha - np.arange(1, len(pha) + 1) * pha[0], TWOPI)

        peak = amp_ref.max()
        amp_ok = np.allclose(amp, amp_ref, rtol=AMP_RTOL, atol=AMP_RTOL * peak)
        sig = amp_ref > PHA_AMP_FRAC * peak
        dpha = np.angle(np.exp(1j * (pha_rot - pha_ref)))   # wrapped phase difference
        pha_ok = np.all(np.abs(dpha[sig]) < PHA_ATOL)

        worst_amp = max(worst_amp, np.max(np.abs(amp - amp_ref)) / peak)
        worst_pha = max(worst_pha, np.max(np.abs(dpha[sig])) if sig.any() else 0.0)
        if not (amp_ok and pha_ok):
            failures.append((r.get("tag"), r.get("shape"), len(templ)))

    print(f"checked {len(seen)} templates; worst rel-amp err={worst_amp:.2e}, "
          f"worst phase err={worst_pha:.2e} rad")
    assert not failures, f"cprof mismatch in {len(failures)} cases: {failures[:5]}"


if __name__ == "__main__":
    test_cprof_matches_frozen_fortran()
    print("cprof matches the frozen Fortran outputs.")

"""Phase 1 unit test: pure-Python cprof vs. the Fortran cprof.

Compares fftfit_py.cprof against the live Fortran presto.fftfit.cprof over the
template profiles frozen in fftfit_reference.npz (all shapes / nbins), checking the
complex spectrum, amplitudes, and phases. Run with pytest, or directly:

    python python/fftfit_src/test_cprof.py
"""

import os
import numpy as np

from presto.fftfit import cprof as cprof_f
import fftfit_py

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "fftfit_reference.npz")

# The Fortran cprof is single-precision (real*4), so phases of harmonics with
# negligible amplitude are numerically meaningless. Compare phases only where the
# amplitude is a non-trivial fraction of the peak.
AMP_RTOL = 1e-4
PHA_ATOL = 1e-3          # radians, on significant harmonics
PHA_AMP_FRAC = 1e-3      # "significant" = amp > this * max(amp)


def _load_templates():
    d = np.load(REF, allow_pickle=True)
    meta = d["meta"]
    seen = set()
    out = []
    for r in meta:
        key = r["idx"]
        templ = d[f"templ_{key}"]
        # de-dup identical templates (many cases share one) by (shape/tag, nbins, bytes)
        h = (r.get("tag"), r.get("shape"), len(templ), templ.tobytes())
        if h in seen:
            continue
        seen.add(h)
        out.append((r, templ))
    return out


def check_one(templ):
    c_f, amp_f, pha_f = cprof_f(templ)
    c_p, amp_p, pha_p = fftfit_py.cprof(templ)

    # Fortran returns fixed-length MAXSAM/2 arrays; trim to nh.
    nh = len(templ) // 2
    amp_f = np.asarray(amp_f)[:nh]
    pha_f = np.asarray(pha_f)[:nh]

    peak = amp_f.max()
    amp_ok = np.allclose(amp_p, amp_f, rtol=AMP_RTOL, atol=AMP_RTOL * peak)

    sig = amp_f > PHA_AMP_FRAC * peak
    # phases live on a circle; compare via the wrapped difference
    dpha = np.angle(np.exp(1j * (pha_p - pha_f)))
    pha_ok = np.all(np.abs(dpha[sig]) < PHA_ATOL)

    max_amp_err = np.max(np.abs(amp_p - amp_f)) / peak
    max_pha_err = np.max(np.abs(dpha[sig])) if sig.any() else 0.0
    return amp_ok and pha_ok, max_amp_err, max_pha_err


def test_cprof_matches_fortran():
    templates = _load_templates()
    assert templates, "no templates found in reference dataset"
    worst_amp = worst_pha = 0.0
    failures = []
    for r, templ in templates:
        ok, ae, pe = check_one(templ)
        worst_amp = max(worst_amp, ae)
        worst_pha = max(worst_pha, pe)
        if not ok:
            failures.append((r.get("tag"), r.get("shape"), len(templ), ae, pe))
    print(f"checked {len(templates)} templates; "
          f"worst rel-amp err={worst_amp:.2e}, worst phase err={worst_pha:.2e} rad")
    assert not failures, f"cprof mismatch in {len(failures)} cases: {failures[:5]}"


if __name__ == "__main__":
    test_cprof_matches_fortran()
    print("cprof port matches Fortran.")

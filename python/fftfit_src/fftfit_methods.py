"""High-level, method-selecting FFTFIT API, mirroring the interface of Anne
Archibald's PINT `pint.profile` so her test suite can drive both implementations.

Codes:
  * "classic"  -- the faithful NumPy port of Taylor's Fortran (fftfit_py).
  * "aarchiba" -- Anne Archibald's independent algorithm (fftfit_aarchiba).

`fftfit_full`/`fftfit_basic` take (template, profile) and return shifts in *turns*
(Anne's convention). `fftfit_cprof`/`fftfit_classic` provide the PRESTO-style
(amp, pha) / bins interface. This module is dev-only during the rewrite; in Phase 4
it becomes the public `presto` API surface.
"""

import numpy as np

import fftfit_py as _classic
import fftfit_aarchiba as _aarchiba
from fftfit_aarchiba import wrap, shift, irfft_value, FFTFITResult  # noqa: F401 (re-exported)


def _classic_full(template, profile):
    """Run the classic method via its PRESTO-style API and repackage as an FFTFITResult."""
    template = np.asarray(template, dtype=np.float64)
    n = len(template)
    _c, amp, pha = _classic.cprof(template)
    sh, esh, snr, esnr, b, errb, ngood = _classic.fftfit(profile, amp, pha, code="classic")
    r = FFTFITResult()
    r.shift = wrap(sh / n)
    r.uncertainty = esh / n           # note: esh=999 (bailout) -> a huge, obviously-bad sigma
    r.scale = b
    r.snr = snr
    r.ngood = ngood
    return r


def fftfit_full(template, profile, code="classic"):
    if code == "aarchiba":
        return _aarchiba.fftfit_full(template, profile)
    elif code == "classic":
        return _classic_full(template, profile)
    raise ValueError("Unrecognized FFTFIT method %r" % code)


def fftfit_basic(template, profile, code="classic"):
    if code == "aarchiba":
        return _aarchiba.fftfit_basic(template, profile)
    return fftfit_full(template, profile, code=code).shift


def fftfit_cprof(template):
    """PRESTO-style template transform: return (c, amp, pha) for fftfit_classic."""
    return _classic.cprof(template)


def fftfit_classic(profile, template_amplitudes, template_angles, code="classic"):
    """PRESTO-style call: (profile, amp, pha) -> (shift, eshift, snr, esnr, b, errb, ngood)."""
    return _classic.fftfit(profile, template_amplitudes, template_angles, code=code)

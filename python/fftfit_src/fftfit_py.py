"""Pure-Python (NumPy/SciPy) reimplementation of PRESTO's FFTFIT.

Work in progress -- see ROADMAP.md, "FFTFIT pure-Python rewrite". This module is
being built and validated bottom-up against the original Fortran (`presto.fftfit`)
using the frozen oracle in fftfit_reference.npz. It currently lives under
python/fftfit_src/ during development; it will move into python/presto/ and replace
the f2py extension in Phase 4.

Public API mirrors the Fortran exactly, for drop-in use by get_TOAs.py / sum_profiles.py:
    c, amp, pha = cprof(template)
    shift, eshift, snr, esnr, b, errb, ngood = fftfit(profile, amp, pha)
"""

import numpy as np
from scipy.optimize import brentq

import fftfit_aarchiba as _aarchiba

TWOPI = 2.0 * np.pi


def cprof(y):
    """FFT a real profile; return spectrum and its amplitude/phase.

    Mirrors cprof.f: for a profile of length ``nmax`` with ``nh = nmax // 2``,
    returns the complex spectrum ``c[0..nh]`` (``c[0]`` = DC, ``c[nh]`` = Nyquist)
    plus ``amp[1..nh]`` and ``pha[1..nh]`` for the harmonics (length ``nh`` each).

    Two conventions of Taylor's ``ffft`` (isign=1, ireal=1) must be matched:
      * phase sign: opposite of NumPy's forward ``rfft``, so we conjugate;
      * amplitude: Taylor's half-length real-FFT packing yields 2x NumPy's
        unnormalized forward transform (uniformly, incl. DC and Nyquist).
    """
    y = np.asarray(y, dtype=np.float64)
    nmax = len(y)
    nh = nmax // 2
    # Conjugate for Taylor's isign=1 phase convention; x2 for his amplitude convention.
    c = 2.0 * np.conjugate(np.fft.rfft(y))  # length nh+1: c[0]=DC, c[1..nh]=harmonics
    harm = c[1:nh + 1]
    amp = np.abs(harm)
    pha = np.where(amp > 0.0, np.angle(harm), 0.0)
    return c, amp, pha


def _dchisqr(tau, tmp, r, nsum):
    """Derivative of chi^2 w.r.t. tau over the first ``nsum`` harmonics (dchisqr in brent.f).

    ``sum_{k=1..nsum} k * tmp[k] * sin(-r[k] + k*tau)`` with k the physical harmonic number.
    """
    if nsum <= 0:
        return 0.0
    k = np.arange(1, nsum + 1)
    return float(np.sum(k * tmp[:nsum] * np.sin(-r[:nsum] + k * tau)))


def _fccf(tmp, r):
    """Coarse cross-correlation initial shift (radians), mirroring fccf.f.

    Builds a 64-lag CCF from the first 16 harmonics of ``tmp`` (=p*s) and ``r``
    (=theta-phi), inverse-transforms, finds the peak lag, parabolically interpolates,
    and returns the shift in radians. The Fortran uses its ``ffft`` with isign=-1 on a
    length-64 complex array; that convention matches NumPy's forward FFT (see cprof).
    """
    nprof = 64
    nh = nprof // 2       # 32
    nhalf = nh // 2       # 16 harmonics used
    ccf = np.zeros(nprof, dtype=np.complex128)
    i = np.arange(1, nhalf + 1)
    ccf[1:nhalf + 1] = tmp[:nhalf] * np.exp(1j * r[:nhalf])
    ccf[nprof - i] = np.conjugate(ccf[i])
    # isign=-1, ireal=0 complex transform == NumPy forward FFT (normalization is
    # irrelevant here: we only use the argmax and a ratio of real parts).
    cc = np.fft.fft(ccf).real
    imax = int(np.argmax(cc))
    fb = cc[imax]
    fa = cc[(imax - 1) % nprof]
    fc = cc[(imax + 1) % nprof]
    denom = 2 * fb - fc - fa
    shift = imax + 0.5 * (fa - fc) / denom if denom != 0 else float(imax)
    if shift > nh:
        shift -= nprof
    return shift * TWOPI / nprof


def _classic_tau(tmp, r, ngood, nh):
    """Find the shift tau (radians) by Taylor's coarse CCF + harmonic-continuation loop.

    Returns tau, or None if the root cannot be bracketed (the ntries>100 bailout).
    """
    tau = _fccf(tmp, r)
    nsum0 = min(16, ngood // 4)
    for nsum in range(nsum0, ngood + 1):
        dtau = 0.2 / nsum if nsum > 0 else 1e30
        edtau = 0.01 / nsum if nsum > 0 else 1e30
        if nsum > (nh / 2.0 + 0.5):
            edtau = 1e-4

        a = b = None
        low = high = False
        ntries = 0
        while True:
            ftau = _dchisqr(tau, tmp, r, nsum)
            ntries += 1
            if ftau < 0.0:
                a = tau
                tau += dtau
                low = True
            else:
                b = tau
                tau -= dtau
                high = True
            if ntries > 100:
                return None            # cannot bracket -> caller emits the bailout tuple
            if low != high:
                continue
            break
        lo, hi = (a, b) if a < b else (b, a)
        tau = brentq(_dchisqr, lo, hi, args=(tmp, r, nsum), xtol=edtau)
    return tau


def _finalize(tau, p, s, r, nh, ngood, fac):
    """Given a shift tau, compute Taylor's scale/S-N/uncertainty outputs.

    Shared by both methods so their diagnostic outputs (snr, b, ...) are consistent;
    only the way ``tau`` was found differs. Returns (shift, eshift, snr, esnr, b, errb).
    """
    tmp = p * s
    k = np.arange(1, ngood + 1)
    cosfac = np.cos(-r[:ngood] + k * tau)
    s1 = np.sum(tmp[:ngood] * cosfac)
    s2 = np.sum(s[:ngood] ** 2)
    s3 = np.sum(k ** 2 * tmp[:ngood] * cosfac)
    b = s1 / s2

    sq = (p[:ngood] ** 2 - 2.0 * b * p[:ngood] * s[:ngood] * np.cos(r[:ngood] - k * tau)
          + (b * s[:ngood]) ** 2)
    rms = np.sqrt(max(np.sum(sq), 0.0) / ngood)   # clip: a perfect noiseless fit can round <0
    errb = rms / np.sqrt(2.0 * s2)
    errtau = rms / np.sqrt(2.0 * b * s3) if s3 > 0.0 else 0.0
    snr = 2.0 * np.sqrt(2.0 * nh) * b / rms if rms > 0.0 else 0.0
    esnr = snr * errb / b if b != 0.0 else 0.0
    return fac * tau, fac * errtau, snr, esnr, b, errb


def fftfit(prof, s, phi, code="classic"):
    """Determine the shift between ``prof`` and a template, in the Fourier domain.

    ``s`` and ``phi`` are the template amplitude and phase (as returned by :func:`cprof`,
    with ``phi`` optionally rotated as get_TOAs.py does). Returns
    ``(shift, eshift, snr, esnr, b, errb, ngood)`` with ``shift``/``eshift`` in bins.

    ``code`` selects the algorithm:
      * ``"classic"`` -- faithful port of Taylor's fftfit.f (default; backward-compatible).
      * ``"aarchiba"`` -- Anne Archibald's independent algorithm (handles symmetric /
        low-harmonic profiles the classic one bails on). Its own shift and uncertainty
        are reported; snr/b/errb are computed with the same post-fit formulas as classic
        so the diagnostic outputs stay consistent between methods.
    """
    prof = np.asarray(prof, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    nmax = len(prof)
    nh = nmax // 2
    fac = nmax / TWOPI

    # Number of "good" (significant) template harmonics: leading harmonics whose
    # amplitude exceeds twice the mean of the upper half of the spectrum.
    ave = 2.0 * np.sum(s[nh // 2:nh]) / nh
    below = np.where(s[:nh] < ave)[0]
    ngood = nh if below.size == 0 else int(below[0])

    # Profile spectrum; form the cross terms.
    _, p, theta = cprof(prof)
    r = theta - phi

    eshift_override = None
    if code == "classic":
        tau = _classic_tau(p * s, r, ngood, nh)
        if tau is None:
            return (0.0, 999.0, 0.0, 0.0, 0.0, 0.0, ngood)
    elif code == "aarchiba":
        # Reconstruct the template from (s, phi) and run the independent algorithm.
        tc = np.zeros(nh + 1, dtype=complex)
        tc[1:] = (s / 2.0) * np.exp(-1.0j * phi)
        template = np.fft.irfft(tc, nmax)
        try:
            rr = _aarchiba.fftfit_full(template, prof)
        except Exception:
            return (0.0, 999.0, 0.0, 0.0, 0.0, 0.0, ngood)
        tau = rr.shift * TWOPI                 # turns -> radians (shift = fac*tau = nmax*rr.shift)
        eshift_override = rr.uncertainty * nmax
    else:
        raise ValueError("Unrecognized FFTFIT method %r (use 'classic' or 'aarchiba')" % code)

    shift, eshift, snr, esnr, b, errb = _finalize(tau, p, s, r, nh, ngood, fac)
    if eshift_override is not None:
        eshift = eshift_override
    return shift, eshift, snr, esnr, b, errb, ngood

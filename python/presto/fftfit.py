"""FFTFIT: align a pulse profile with a template in the Fourier domain to measure
a phase shift (pulsar time-of-arrival), after Taylor (1992).

Pure-Python (NumPy/SciPy) reimplementation that replaces the former f2py-wrapped
Fortran. Two algorithms are available, selectable via ``code=``:

  * ``"classic"`` -- a faithful port of Joe Taylor's original Fortran FFTFIT
    (default; reproduces the long-standing PRESTO behavior).
  * ``"aarchiba"`` -- Anne Archibald's independent algorithm (see
    :mod:`presto.fftfit_aarchiba`), which also handles near-symmetric / very-low-
    harmonic profiles that the classic method cannot.

Low-level PRESTO-compatible API (used by get_TOAs.py, sum_profiles.py)::

    c, amp, pha = cprof(template)
    shift, eshift, snr, esnr, b, errb, ngood = fftfit(profile, amp, pha, code="classic")

High-level API (template + profile, shift in *turns*), mirroring PINT's interface::

    r = fftfit_full(template, profile, code=...)      # r.shift, r.uncertainty, ...
    shift = fftfit_basic(template, profile, code=...)

Note: the reported uncertainties are only approximately calibrated (both methods tend
to under-cover, more so for broad profiles); see ROADMAP.md. The shift is reliable.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from presto import fftfit_aarchiba as _aarchiba
# Re-export the shared helpers so callers/tests can use presto.fftfit.wrap etc.
from presto.fftfit_aarchiba import wrap, shift, irfft_value, FFTFITResult  # noqa: F401

TWOPI = 2.0 * np.pi

# The PRESTO-style ``fftfit`` return tuple: shift, eshift, snr, esnr, b, errb, ngood.
FFTFITTuple = tuple[float, float, float, float, float, float, int]


def cprof(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FFT a real profile and return its spectrum, amplitudes, and phases.

    Mirrors the Fortran ``cprof``. Two conventions of Taylor's real FFT (``ffft`` with
    ``isign=1, ireal=1``) are matched so the results are drop-in compatible:

    * phase sign: opposite of NumPy's forward ``rfft``, so the transform is conjugated;
    * amplitude: Taylor's half-length real-FFT packing yields twice NumPy's
      unnormalized forward transform (uniformly, including the DC and Nyquist terms).

    Parameters
    ----------
    y : numpy.ndarray
        Real pulse profile of length ``nmax`` (typically a power of two).

    Returns
    -------
    c : numpy.ndarray
        Complex spectrum ``c[0..nh]`` where ``nh = nmax // 2``; ``c[0]`` is the DC term
        and ``c[nh]`` the Nyquist term.
    amp : numpy.ndarray
        Amplitudes ``|c[k]|`` of the harmonics ``k = 1..nh`` (length ``nh``).
    pha : numpy.ndarray
        Phases ``angle(c[k])`` of the harmonics ``k = 1..nh`` (length ``nh``); set to
        zero where the amplitude is zero.
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


def _dchisqr(tau: float, tmp: np.ndarray, r: np.ndarray, nsum: int) -> float:
    """Derivative of chi-squared with respect to the shift ``tau``.

    Mirrors ``dchisqr`` in the Fortran ``brent.f``: the sum over the first ``nsum``
    harmonics of ``k * tmp[k] * sin(-r[k] + k*tau)``, with ``k`` the physical harmonic
    number. The best-fit ``tau`` is the root of this function.

    Parameters
    ----------
    tau : float
        Trial shift, in radians of the fundamental.
    tmp : numpy.ndarray
        Product of the profile and template harmonic amplitudes (``p * s``).
    r : numpy.ndarray
        Harmonic phase differences ``theta - phi``.
    nsum : int
        Number of leading harmonics to include.

    Returns
    -------
    float
        The value of the derivative at ``tau`` (zero when ``nsum <= 0``).
    """
    if nsum <= 0:
        return 0.0
    k = np.arange(1, nsum + 1)
    return float(np.sum(k * tmp[:nsum] * np.sin(-r[:nsum] + k * tau)))


def _fccf(tmp: np.ndarray, r: np.ndarray) -> float:
    """Coarse cross-correlation estimate of the initial shift.

    Mirrors the Fortran ``fccf``: builds a 64-lag cross-correlation from the first 16
    harmonics of ``tmp`` and ``r``, inverse-transforms it, finds the peak lag, and
    parabolically interpolates. The Fortran uses ``ffft`` with ``isign=-1`` on a
    length-64 complex array, a convention that matches NumPy's forward FFT.

    Parameters
    ----------
    tmp : numpy.ndarray
        Product of the profile and template harmonic amplitudes (``p * s``).
    r : numpy.ndarray
        Harmonic phase differences ``theta - phi``.

    Returns
    -------
    float
        The coarse shift, in radians of the fundamental.
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


def _classic_tau(tmp: np.ndarray, r: np.ndarray, ngood: int, nh: int) -> float | None:
    """Locate the best-fit shift by Taylor's coarse CCF plus continuation loop.

    Starts from the :func:`_fccf` estimate and progressively includes more harmonics
    (``nsum`` from ``min(16, ngood // 4)`` up to ``ngood``), re-solving
    ``_dchisqr(tau) = 0`` at each step to home in on the global minimum while avoiding
    local ones.

    Parameters
    ----------
    tmp : numpy.ndarray
        Product of the profile and template harmonic amplitudes (``p * s``).
    r : numpy.ndarray
        Harmonic phase differences ``theta - phi``.
    ngood : int
        Number of significant template harmonics.
    nh : int
        Half the profile length (``nmax // 2``).

    Returns
    -------
    float or None
        The best-fit shift ``tau`` in radians, or ``None`` if the root cannot be
        bracketed (the Fortran's ``ntries > 100`` bailout).
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


def _finalize(
    tau: float,
    p: np.ndarray,
    s: np.ndarray,
    r: np.ndarray,
    nh: int,
    ngood: int,
    fac: float,
) -> tuple[float, float, float, float, float, float]:
    """Compute Taylor's scale, S/N, and uncertainty outputs for a given shift.

    Shared by both methods so that their diagnostic outputs are computed identically;
    only the way ``tau`` was found differs between ``"classic"`` and ``"aarchiba"``.

    Parameters
    ----------
    tau : float
        The best-fit shift, in radians of the fundamental.
    p : numpy.ndarray
        Profile harmonic amplitudes.
    s : numpy.ndarray
        Template harmonic amplitudes.
    r : numpy.ndarray
        Harmonic phase differences ``theta - phi``.
    nh : int
        Half the profile length (``nmax // 2``).
    ngood : int
        Number of significant template harmonics used in the sums.
    fac : float
        Conversion factor from radians to bins (``nmax / 2pi``).

    Returns
    -------
    tuple of float
        ``(shift, eshift, snr, esnr, b, errb)`` with ``shift`` and ``eshift`` in bins.
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


def fftfit(prof: np.ndarray, s: np.ndarray, phi: np.ndarray,
           code: str = "classic") -> FFTFITTuple:
    """Determine the shift between a profile and a template in the Fourier domain.

    This is the low-level, PRESTO-compatible entry point used by ``get_TOAs.py`` and
    ``sum_profiles.py``. The template is supplied as its harmonic amplitudes and phases
    (from :func:`cprof`, with ``phi`` optionally rotated as ``get_TOAs.py`` does).

    Parameters
    ----------
    prof : numpy.ndarray
        The observed profile.
    s : numpy.ndarray
        Template harmonic amplitudes (from :func:`cprof`).
    phi : numpy.ndarray
        Template harmonic phases (from :func:`cprof`), possibly rotated.
    code : {"classic", "aarchiba"}, optional
        Which algorithm to use. ``"classic"`` (default) is the faithful port of
        Taylor's ``fftfit.f``. ``"aarchiba"`` is Anne Archibald's independent algorithm,
        which handles the near-symmetric / low-harmonic profiles the classic method
        bails on; it reports its own shift and uncertainty, while ``snr``/``b``/``errb``
        are computed with the same post-fit formulas as classic for consistency.

    Returns
    -------
    tuple
        ``(shift, eshift, snr, esnr, b, errb, ngood)``, with ``shift`` and ``eshift``
        in bins. A failed fit is flagged by ``shift = 0.0`` and ``eshift = 999.0``.

    Raises
    ------
    ValueError
        If ``code`` is not ``"classic"`` or ``"aarchiba"``.
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

    shift_bins, eshift, snr, esnr, b, errb = _finalize(tau, p, s, r, nh, ngood, fac)
    if eshift_override is not None:
        eshift = eshift_override
    return shift_bins, eshift, snr, esnr, b, errb, ngood


# --------------------------------------------------------------------------- #
# High-level, template+profile API (shift in turns), mirroring PINT's interface.
# --------------------------------------------------------------------------- #
def _classic_full(template: np.ndarray, profile: np.ndarray) -> FFTFITResult:
    """Run the classic method via the PRESTO-style API and repackage the result.

    Parameters
    ----------
    template : numpy.ndarray
        The template profile.
    profile : numpy.ndarray
        The observed profile.

    Returns
    -------
    FFTFITResult
        Result with ``shift`` and ``uncertainty`` in turns, plus ``scale``, ``snr``,
        and ``ngood``.
    """
    template = np.asarray(template, dtype=np.float64)
    n = len(template)
    _c, amp, pha = cprof(template)
    sh, esh, snr, esnr, b, errb, ngood = fftfit(profile, amp, pha, code="classic")
    r = FFTFITResult()
    r.shift = wrap(sh / n)
    r.uncertainty = esh / n           # note: esh=999 (bailout) -> a huge, obviously-bad sigma
    r.scale = b
    r.snr = snr
    r.ngood = ngood
    return r


def fftfit_full(template: np.ndarray, profile: np.ndarray,
                code: str = "classic") -> FFTFITResult:
    """Align a template to a profile and return the full match result.

    Parameters
    ----------
    template : numpy.ndarray
        The template profile.
    profile : numpy.ndarray
        The observed profile.
    code : {"classic", "aarchiba"}, optional
        Which algorithm to use (see :func:`fftfit`).

    Returns
    -------
    FFTFITResult
        Result whose ``shift`` (in turns) satisfies ``shift(template, r.shift) ~ profile``,
        along with ``uncertainty`` (turns) and, depending on the method, ``scale``,
        ``offset``, ``snr``, and ``ngood``.

    Raises
    ------
    ValueError
        If ``code`` is not ``"classic"`` or ``"aarchiba"``.
    """
    if code == "aarchiba":
        return _aarchiba.fftfit_full(template, profile)
    elif code == "classic":
        return _classic_full(template, profile)
    raise ValueError("Unrecognized FFTFIT method %r" % code)


def fftfit_basic(template: np.ndarray, profile: np.ndarray,
                 code: str = "classic") -> float:
    """Return just the phase shift to align a template with a profile.

    Parameters
    ----------
    template : numpy.ndarray
        The template profile.
    profile : numpy.ndarray
        The observed profile.
    code : {"classic", "aarchiba"}, optional
        Which algorithm to use (see :func:`fftfit`).

    Returns
    -------
    float
        The shift in turns (wrapped to ``[-0.5, 0.5)``) such that
        ``shift(template, result)`` best matches ``profile``.
    """
    if code == "aarchiba":
        return _aarchiba.fftfit_basic(template, profile)
    return fftfit_full(template, profile, code=code).shift

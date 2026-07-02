"""Anne Archibald's independent FFTFIT algorithm (the "aarchiba" method).

Adopted into PRESTO from Anne Archibald's unmerged PINT work:
  https://github.com/nanograv/PINT/pull/777  (branch `fftfit_new` on aarchiba/PINT),
  files src/pint/profile/fftfit_aarchiba.py and src/pint/profile/__init__.py.

This is an independent reimplementation of the profile-alignment ("FFTFIT") idea,
NOT a port of Taylor's Fortran: it forms the cross-spectrum of template and profile,
inverse-transforms an upsampled CCF to bracket the peak, refines the shift by bounded
scalar minimization of an interpolated CCF, and recovers scale/offset (and their
uncertainty) by linear least squares. Compared to the classic PRESTO algorithm it also
handles near-symmetric / very-low-harmonic profiles (which the Fortran bails on) and
templates/profiles of different lengths.

Conventions differ from PRESTO's `fftfit`: `fftfit_basic` returns the shift in *turns*
(phase, wrapped to [-0.5, 0.5)); `fftfit_full` returns an FFTFITResult with `.shift`
(turns), `.scale`, `.offset`, `.uncertainty` (turns), and `.std`. A separate adapter
presents these in PRESTO's `(shift_bins, eshift, snr, esnr, b, errb, ngood)` form.

Known limitation: the reported `uncertainty` is only approximately calibrated (it tends
to under-cover, more so for broad profiles) -- see ROADMAP.md "FFTFIT uncertainty
calibration". The shift itself is reliable.

------------------------------------------------------------------------------
Original code Copyright (c) 2014-2025, PINT developers. Redistributed under the
3-clause BSD license (PINT's LICENSE.md), which is compatible with PRESTO's GPL-2.0.
The BSD copyright notice and disclaimer are retained here as required by that license.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import scipy.optimize
import scipy.linalg
import scipy.stats


class FFTFITResult:
    """Container for the results of an FFTFIT operation.

    Which attributes are present depends on the options passed to :func:`fftfit_full`.

    Attributes
    ----------
    shift : float
        Phase shift, in turns (wrapped to ``[-0.5, 0.5)``), such that
        ``shift(template, shift)`` best matches the profile.
    scale : float
        Multiplicative factor relating the template to the profile.
    offset : float
        Additive baseline relating the template to the profile.
    uncertainty : float
        Estimated one-sigma uncertainty in ``shift`` (turns; approximately calibrated).
    std : float
        Estimated per-bin noise standard deviation of the profile.
    cov : numpy.ndarray
        2x2 covariance matrix of the (scale, offset) least-squares fit.
    """

    pass


def wrap(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap a phase to the range ``[-0.5, 0.5)``.

    Parameters
    ----------
    a : float or numpy.ndarray
        Phase(s) in turns.

    Returns
    -------
    float or numpy.ndarray
        The wrapped phase(s).
    """
    return (a + 0.5) % 1 - 0.5


def shift(profile: np.ndarray, phase: float) -> np.ndarray:
    """Shift a profile towards later phases by a fractional number of turns.

    Uses Fourier interpolation; for even-length profiles the Nyquist component is left
    unmodified.

    Parameters
    ----------
    profile : numpy.ndarray
        The profile to shift.
    phase : float
        The shift, in turns.

    Returns
    -------
    numpy.ndarray
        The shifted profile (same length as the input).
    """
    c = np.fft.rfft(profile)
    if len(profile) % 2:
        c *= np.exp(-2.0j * np.pi * phase * np.arange(len(c)))
    else:
        c[:-1] *= np.exp(-2.0j * np.pi * phase * np.arange(len(c) - 1))
    return np.fft.irfft(c, len(profile))


def irfft_value(c: np.ndarray, phase: np.ndarray | float, n: int | None = None) -> np.ndarray:
    """Evaluate the inverse real FFT of ``c`` at arbitrary (non-grid) phase(s).

    Parameters
    ----------
    c : numpy.ndarray
        Real-FFT coefficients (as from :func:`numpy.fft.rfft`).
    phase : float or numpy.ndarray
        Phase(s) at which to evaluate, in turns.
    n : int, optional
        Effective length of the implied time series. Defaults to the natural length
        ``2 * (len(c) - 1)``.

    Returns
    -------
    numpy.ndarray
        The interpolated value(s), with the same shape as ``phase``.
    """
    natural_n = (len(c) - 1) * 2
    if n is None:
        n = natural_n
    phase = np.asarray(phase)
    s = phase.shape
    phase = np.atleast_1d(phase)
    c = np.array(c)
    c[0] /= 2
    if n == natural_n:
        c[-1] /= 2
    return (
        (c[:, None] * np.exp(2.0j * np.pi * phase[None, :] * np.arange(len(c))[:, None]))
        .sum(axis=0)
        .real
        * 2
        / n
    ).reshape(s)


def fftfit_full(
    template: np.ndarray,
    profile: np.ndarray,
    compute_scale: bool = True,
    compute_uncertainty: bool = True,
    std: float | None = None,
) -> FFTFITResult:
    """Align a template to a profile in the Fourier domain (Archibald's algorithm).

    Forms the cross-spectrum of ``template`` and ``profile``, inverse-transforms an
    upsampled cross-correlation to bracket the peak, refines the shift by bounded scalar
    minimization of the interpolated CCF, and (optionally) recovers the scale, offset,
    and shift uncertainty by linear least squares.

    Parameters
    ----------
    template : numpy.ndarray
        The template representing the ideal pulse profile.
    profile : numpy.ndarray
        The observed profile to align the template with.
    compute_scale : bool, optional
        If True, also fit the scale and offset relating template to profile.
    compute_uncertainty : bool, optional
        If True, also estimate the shift uncertainty (implies ``compute_scale``).
    std : float, optional
        Known per-bin noise standard deviation; if omitted it is estimated from the
        fit residuals.

    Returns
    -------
    FFTFITResult
        The match result. ``shift`` (turns) is always set; ``scale``/``offset`` are set
        when ``compute_scale`` is True; ``uncertainty``/``std``/``cov`` when
        ``compute_uncertainty`` is True.

    Raises
    ------
    ValueError
        If the bounded minimization fails to converge.
    """
    upsample = 8
    t_c = np.fft.rfft(template)
    if len(template) % 2 == 0:
        t_c[-1] = 0
    p_c = np.fft.rfft(profile)
    if len(profile) % 2 == 0:
        p_c[-1] = 0
    n_c = min(len(t_c), len(p_c))
    t_c = t_c[:n_c]
    p_c = p_c[:n_c]

    ccf_c = np.conj(t_c).copy()
    ccf_c *= p_c
    ccf_c[0] = 0
    n_long = 2 ** int(np.ceil(np.log2(2 * (n_c - 1) * upsample)))
    ccf = np.fft.irfft(ccf_c, n_long)
    i = np.argmax(ccf)
    x = i / len(ccf)
    lo, hi = x - 1 / len(ccf), x + 1 / len(ccf)

    def gof(x):
        return -irfft_value(ccf_c, x, n_long)

    res = scipy.optimize.minimize_scalar(
        gof, bounds=(lo, hi), method="Bounded", options=dict(xatol=1e-5 / n_c)
    )
    if not res.success:
        raise ValueError("FFTFIT failed: %s" % res.message)
    r = FFTFITResult()
    r.shift = wrap(res.x)

    if compute_scale or compute_uncertainty:
        s_c = t_c * np.exp(-2j * np.pi * np.arange(len(t_c)) * r.shift)
        n_data = 2 * len(s_c) - 1
        a = np.zeros((n_data, 2))
        b = np.zeros(n_data)
        a[0, 1] = len(template)
        a[0, 0] = s_c[0].real
        b[0] = p_c[0].real
        b[1 : len(p_c)] = p_c[1:].real
        b[len(p_c) :] = p_c[1:].imag
        a[1 : len(s_c), 0] = s_c[1:].real
        a[len(s_c) :, 0] = s_c[1:].imag
        lin_x, _res, _rk, _s = scipy.linalg.lstsq(a, b)
        r.scale = lin_x[0]
        r.offset = lin_x[1]

        if compute_uncertainty:
            if std is None:
                resid = r.scale * shift(template, r.shift) + r.offset - profile
                std = np.sqrt(np.mean(resid ** 2))
            r.std = std
            J = np.zeros((2 * len(s_c) - 2, 2))
            J[: len(s_c) - 1, 0] = -r.scale * 2 * np.pi * s_c[1:].imag * np.arange(1, len(s_c))
            J[len(s_c) - 1 :, 0] = r.scale * 2 * np.pi * s_c[1:].real * np.arange(1, len(s_c))
            J[: len(s_c) - 1, 1] = s_c[1:].real
            J[len(s_c) - 1 :, 1] = s_c[1:].imag
            cov = scipy.linalg.inv(np.dot(J.T, J))
            # FIXME (from the original): std is per data point, not per real/imag entry
            # in s_c; the sqrt(len(profile)/2) conversion is the suspected miscalibration.
            r.uncertainty = std * np.sqrt(len(profile) * cov[0, 0] / 2)
            r.cov = cov
    return r


def fftfit_basic(template: np.ndarray, profile: np.ndarray) -> float:
    """Return just the shift needed to align a template with a profile.

    Parameters
    ----------
    template : numpy.ndarray
        The template representing the ideal pulse profile.
    profile : numpy.ndarray
        The observed profile to align the template with.

    Returns
    -------
    float
        The shift in turns (wrapped to ``[-0.5, 0.5)``).
    """
    r = fftfit_full(template, profile, compute_scale=False, compute_uncertainty=False)
    return r.shift

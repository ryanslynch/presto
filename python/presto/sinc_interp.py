from __future__ import annotations

import numpy as np
import numpy.fft as FFT
from scipy.special import i0


def kaiser_window(xs: np.ndarray, halfwidth: float, alpha: float) -> np.ndarray:
    """
    Return the Kaiser window function.

    Parameters
    ----------
    xs : numpy.ndarray
        The values at which to evaluate the window.
    halfwidth : float
        The half-width of the window.
    alpha : float
        The roll-off parameter. Some particularly interesting values:

        ===== =========================================
        alpha behaviour
        ===== =========================================
        0     Rectangular window
        5     Similar to the Hamming window
        6     Similar to the Hanning window
        8.6   Almost identical to the Blackman window
        ===== =========================================

    Returns
    -------
    numpy.ndarray
        The Kaiser window evaluated at `xs`, zero outside +/- `halfwidth`.
    """
    win = i0(alpha * np.sqrt(1.0 - (xs / halfwidth) ** 2.0)) / i0(alpha)
    return np.where(np.fabs(xs) <= halfwidth, win, 0.0)


def hanning_window(xs: np.ndarray, halfwidth: float) -> np.ndarray:
    """
    Return the Hanning window function.

    Parameters
    ----------
    xs : numpy.ndarray
        The values at which to evaluate the window.
    halfwidth : float
        The half-width of the window.

    Returns
    -------
    numpy.ndarray
        The Hanning window evaluated at `xs`, zero outside +/- `halfwidth`.
    """
    win = 0.5 + 0.5 * np.cos(np.pi * xs / halfwidth)
    return np.where(np.fabs(xs) <= halfwidth, win, 0.0)


def hamming_window(xs: np.ndarray, halfwidth: float) -> np.ndarray:
    """
    Return the Hamming window function.

    Parameters
    ----------
    xs : numpy.ndarray
        The values at which to evaluate the window.
    halfwidth : float
        The half-width of the window.

    Returns
    -------
    numpy.ndarray
        The Hamming window evaluated at `xs`, zero outside +/- `halfwidth`.
    """
    win = 0.54 + 0.46 * np.cos(np.pi * xs / halfwidth)
    return np.where(np.fabs(xs) <= halfwidth, win, 0.0)


def blackman_window(xs: np.ndarray, halfwidth: float) -> np.ndarray:
    """
    Return the Blackman window function.

    Parameters
    ----------
    xs : numpy.ndarray
        The values at which to evaluate the window.
    halfwidth : float
        The half-width of the window.

    Returns
    -------
    numpy.ndarray
        The Blackman window evaluated at `xs`, zero outside +/- `halfwidth`.
    """
    rat = np.pi * xs / halfwidth
    win = 0.42 + 0.5 * np.cos(rat) + 0.08 * np.cos(2.0 * rat)
    return np.where(np.fabs(xs) <= halfwidth, win, 0.0)


def rectangular_window(xs: np.ndarray, halfwidth: float) -> np.ndarray:
    """
    Return a rectangular window function.

    Parameters
    ----------
    xs : numpy.ndarray
        The values at which to evaluate the window.
    halfwidth : float
        The half-width of the window.

    Returns
    -------
    numpy.ndarray
        Ones within +/- `halfwidth` of zero, and zero elsewhere.
    """
    return np.where(np.fabs(xs) <= halfwidth, 1.0, 0.0)


_window_function = {
    "rectangular": rectangular_window,
    "none": rectangular_window,
    "hanning": hanning_window,
    "hamming": hamming_window,
    "blackman": blackman_window,
    "kaiser": kaiser_window,
}


def windowed_sinc_interp(
    data: np.ndarray,
    newx: float,
    halfwidth: int | None = None,
    window: str = "hanning",
    alpha: float = 6.0,
) -> float:
    """
    Return a single windowed-sinc-interpolated point from the data.

    Parameters
    ----------
    data : numpy.ndarray
        The data to interpolate.
    newx : float
        The (fractional) index at which to interpolate.
    halfwidth : int, optional
        The half-width of the interpolation kernel in samples. If None
        (default), the largest half-width that fits within `data` is used.
    window : str, optional
        The window function to use, one of the keys of `_window_function`
        (default "hanning").
    alpha : float, optional
        The roll-off parameter for the Kaiser window (default 6.0). Only
        used when `window` is "kaiser".

    Returns
    -------
    float
        The interpolated value at `newx`.
    """
    if np.fabs(round(newx) - newx) < 1e-5:
        return data[int(round(newx))]
    num_pts = (int(np.floor(newx)), len(data) - int(np.ceil(newx)) - 1)
    if halfwidth is None:
        halfwidth = min(num_pts)
    lo_pt = int(np.floor(newx)) - halfwidth
    if lo_pt < 0:
        lo_pt = 0
        print("Warning:  trying to access below the lowest index!")
    hi_pt = lo_pt + 2 * halfwidth
    if hi_pt >= len(data):
        hi_pt = len(data) - 1
        print("Warning:  trying to access above the highest index!")
    halfwidth = (hi_pt - lo_pt) // 2
    pts = np.arange(2 * halfwidth) + lo_pt
    xs = newx - pts
    if window.lower() == "kaiser":
        win = _window_function[window](xs, len(data) // 2, alpha)
    else:
        win = _window_function[window](xs, len(data) // 2)
    return np.add.reduce(np.take(data, pts) * win * np.sinc(xs))


def periodic_interp(
    data: np.ndarray, zoomfact: int, window: str = "hanning", alpha: float = 6.0
) -> np.ndarray:
    """
    Return a periodic, windowed, sinc-interpolation of the data.

    The data is oversampled by a factor of `zoomfact`.

    Parameters
    ----------
    data : numpy.ndarray
        The data to interpolate.
    zoomfact : int
        The oversampling factor. Must be >= 1.
    window : str, optional
        The window function to use, one of the keys of `_window_function`
        (default "hanning").
    alpha : float, optional
        The roll-off parameter for the Kaiser window (default 6.0). Only
        used when `window` is "kaiser".

    Returns
    -------
    numpy.ndarray
        The interpolated data, `zoomfact` times longer than `data`. The
        original `data` is returned unchanged if ``zoomfact == 1``, and 0.0
        is returned if ``zoomfact < 1``.
    """
    zoomfact = int(zoomfact)
    if zoomfact < 1:
        print("zoomfact must be >= 1.")
        return 0.0
    elif zoomfact == 1:
        return data
    newN = len(data) * zoomfact
    # Space out the data
    comb = np.zeros((zoomfact, len(data)), dtype="d")
    comb[0] += data
    comb = np.reshape(np.transpose(comb), (newN,))
    # Compute the offsets
    xs = np.zeros(newN, dtype="d")
    xs[: newN // 2 + 1] = np.arange(newN // 2 + 1, dtype="d") / zoomfact
    xs[-newN // 2 :] = xs[::-1][newN // 2 - 1 : -1]
    # Calculate the sinc times window for the kernel
    if window.lower() == "kaiser":
        win = _window_function[window](xs, len(data) // 2, alpha)
    else:
        win = _window_function[window](xs, len(data) // 2)
    kernel = win * np.sinc(xs)
    if 0:
        plotxy(np.sinc(xs), color="yellow")
        plotxy(win)
        plotxy(kernel, color="red")
        closeplot()
    return FFT.irfft(FFT.rfft(kernel) * FFT.rfft(comb))


if __name__ == "__main__":
    from presto.psr_utils import *
    from presto.Pgplot import *
    from numpy.random import normal

    # from spline import *

    fwhm = 0.01
    ctr_phase = 0.505
    noise_sigma = 0.2

    # The theoretical profile with noise
    Ntheo = 1000
    theo = gaussian_profile(Ntheo, ctr_phase, fwhm) + normal(0.0, noise_sigma, Ntheo)
    theo_phases = np.arange(Ntheo, dtype="d") / Ntheo

    # The "sampled" data
    Ndata = 100
    data = theo[:: Ntheo // Ndata]
    data_phases = theo_phases[:: Ntheo // Ndata]

    # The values to interpolate
    Ncalc = 30
    lo_calc = ctr_phase - 0.05
    hi_calc = ctr_phase + 0.05
    calc_phases = span(lo_calc, hi_calc, Ncalc)
    plotxy(theo, theo_phases, rangex=[lo_calc - 0.2, hi_calc + 0.2])
    plotxy(data, data_phases, line=None, symbol=3, color="green")

    # Do the interpolation one point at a time
    halfwidth = Ndata // 2 - 5
    calc_vals = []
    for phs in calc_phases:
        calc_vals.append(windowed_sinc_interp(data, phs * len(data), halfwidth))
    plotxy(calc_vals, calc_phases, line=None, symbol=3, color="red")

    # Interpolate the full profile using convolution
    zoomfact = 10
    newvals = periodic_interp(data, 10)
    new_phases = np.arange(Ndata * zoomfact, dtype="d") / (Ndata * zoomfact)
    plotxy(newvals, new_phases, line=1, symbol=None, color="yellow")

    # Interpolate using cubic splines
    if 0:
        sdata = interpolate.splrep(data, data_phases, s=0)
        svals = interpolate.splrep(new_phases, sdata, der=0)
        plotxy(svals, new_phases, line=1, symbol=None, color="cyan")
    elif 0:
        sdata = Spline(data_phases, data)
        svals = sdata(new_phases)
        plotxy(svals, new_phases, line=1, symbol=None, color="cyan")

    closeplot()

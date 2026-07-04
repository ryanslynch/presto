from __future__ import annotations

import bisect

import numpy as np
from scipy.integrate import quad
from scipy.special import iv, chdtri, ndtr, ndtri

from presto.psr_constants import PI, TWOPI, PIBYTWO
from presto.simple_roots import newton_raphson
from presto.cosine_rand import cosine_rand


def sine_events(pulsed_frac: float, Nevents: int, phase: float = 0.0) -> np.ndarray:
    """
    Simulate event phases from a sinusoidal folded profile.

    Parameters
    ----------
    pulsed_frac : float
        The pulsed fraction of the profile.
    Nevents : int
        The total number of events to generate.
    phase : float, optional
        A phase offset applied to the pulsed events (default 0.0).

    Returns
    -------
    numpy.ndarray
        An array of `Nevents` phase values in [0, 1) simulating a folded
        profile with a sinusoidal pulse.
    """
    Nsrc = int(pulsed_frac * Nevents + 0.5)
    Nbak = Nevents - Nsrc
    phases = np.zeros(Nevents, dtype=float)
    phases[:Nsrc] += cosine_rand(Nsrc) + phase
    phases[Nsrc:] += np.random.random(Nbak)
    phases = np.fmod(phases, 1.0)
    phases[phases < 0.0] += 1.0
    return phases


def gaussian_events(
    pulsed_frac: float, Nevents: int, fwhm: float, phase: float = 0.0
) -> np.ndarray:
    """
    Simulate event phases from a Gaussian folded profile.

    Parameters
    ----------
    pulsed_frac : float
        The pulsed fraction of the profile.
    Nevents : int
        The total number of events to generate.
    fwhm : float
        The full width at half-max of the Gaussian pulse.
    phase : float, optional
        A phase offset applied to the pulsed events (default 0.0).

    Returns
    -------
    numpy.ndarray
        An array of `Nevents` phase values in [0, 1) simulating a folded
        profile with a Gaussian pulse of width `fwhm`.
    """
    sigma = fwhm / 2.35482
    Nsrc = int(pulsed_frac * Nevents + 0.5)
    Nbak = Nevents - Nsrc
    phases = np.zeros(Nevents, dtype=float)
    phases[:Nsrc] += np.random.standard_normal(Nsrc) * sigma + phase
    phases[Nsrc:] += np.random.random(Nbak)
    phases = np.fmod(phases, 1.0)
    phases[phases < 0.0] += 1.0
    return phases


def harm_to_sum(fwhm: float) -> int:
    """
    Return the optimal number of harmonics to sum incoherently.

    For an MVMD (Modified Von Mises Distribution) profile of width `fwhm`.

    Parameters
    ----------
    fwhm : float
        The pulse full width at half-max (0-1).

    Returns
    -------
    int
        The optimal number of harmonics to sum incoherently.
    """
    # fmt: off
    fwhms = [0.0108, 0.0110, 0.0113, 0.0117, 0.0119, 0.0124, 0.0127, 0.0132,
             0.0134, 0.0140, 0.0145, 0.0151, 0.0154, 0.0160, 0.0167, 0.0173,
             0.0180, 0.0191, 0.0199, 0.0207, 0.0220, 0.0228, 0.0242, 0.0257,
             0.0273, 0.0295, 0.0313, 0.0338, 0.0366, 0.0396, 0.0437, 0.0482,
             0.0542, 0.0622, 0.0714, 0.0836, 0.1037, 0.1313, 0.1799, 0.2883]
    # fmt: on
    return len(fwhms) - bisect.bisect(fwhms, fwhm) + 1


def DFTexact(times: np.ndarray, f: float, maxnumharms: int = 20) -> np.ndarray:
    """
    Return the exact (unbinned) DFT amplitudes at the harmonics of a frequency.

    Parameters
    ----------
    times : numpy.ndarray
        The event times in seconds.
    f : float
        The fundamental frequency in Hz.
    maxnumharms : int, optional
        The number of harmonics to compute (default 20).

    Returns
    -------
    numpy.ndarray
        An array of `maxnumharms` complex amplitudes corresponding to the
        harmonics of `times` with a fundamental at frequency `f`.
    """
    const = -TWOPI * (np.arange(maxnumharms, dtype=float) + 1.0) * f * complex(0.0, 1.0)
    return np.add.reduce(np.exp(np.outer(const, times)), axis=1)


def incoherent_sum(amps: np.ndarray) -> np.ndarray:
    """
    Return the incoherent sum of an array of complex Fourier amplitudes.

    Usually these correspond to the complex harmonics of a periodic signal.

    Parameters
    ----------
    amps : numpy.ndarray
        A series of complex Fourier amplitudes.

    Returns
    -------
    numpy.ndarray
        The accumulated incoherently-summed powers.
    """
    return np.add.accumulate(np.abs(amps) ** 2.0)


def coherent_sum(amps: np.ndarray) -> np.ndarray:
    """
    Return the coherent sum of an array of complex Fourier amplitudes.

    Includes phase information. Usually these correspond to the complex
    harmonics of a periodic signal.

    Parameters
    ----------
    amps : numpy.ndarray
        A series of complex Fourier amplitudes.

    Returns
    -------
    numpy.ndarray
        The accumulated coherently-summed powers.
    """
    phss = np.arctan2(amps.imag, amps.real)
    phs0 = phss[0]
    phscorr = phs0 - np.fmod(np.arange(1.0, len(amps) + 1, dtype=float) * phs0, TWOPI)
    sumamps = np.add.accumulate(amps * np.exp(complex(0.0, 1.0) * phscorr))
    return np.abs(sumamps) ** 2.0


def Htest_exact(
    phases: np.ndarray, maxnumharms: int = 20, weights: np.ndarray | None = None
) -> tuple[float, int]:
    """
    Return an exactly computed (unbinned) H-test statistic for periodicity.

    Returns the Leahy-normalized H-statistic and the best number of
    harmonics summed. If `weights` are set to fractional photon weights,
    the weighted H-test is returned (see Kerr 2011,
    http://arxiv.org/pdf/1103.2128.pdf).

    Parameters
    ----------
    phases : numpy.ndarray
        The folded event phases in [0, 1).
    maxnumharms : int, optional
        The maximum number of harmonics to consider (default 20).
    weights : numpy.ndarray, optional
        Fractional photon weights. If None, the unweighted H-test is used.

    Returns
    -------
    hstat : float
        The H-test statistic.
    harmnum : int
        The best number of harmonics summed.
    """
    N = len(phases)
    Zm2s = np.zeros(maxnumharms, dtype=float)
    rad_phases = 2.0 * np.pi * phases
    weightfact = 1.0 / (np.sum(weights**2.0) / N) if weights is not None else 1.0
    for harmnum in range(1, maxnumharms + 1):
        phss = harmnum * rad_phases
        Zm2s[harmnum - 1] = (
            2.0
            / N
            * (np.add.reduce(np.sin(phss)) ** 2.0 + np.add.reduce(np.cos(phss)) ** 2.0)
        )
        Zm2s[harmnum - 1] *= weightfact
    hs = np.add.accumulate(Zm2s) - 4.0 * np.arange(1.0, maxnumharms + 1) + 4.0
    bestharm = hs.argmax()
    return (hs[bestharm], bestharm + 1)


def Hstat_prob(h: float) -> float:
    """
    Return the probability associated with an H-test statistic.

    Uses the de Jager & Busching 2010 result.

    Parameters
    ----------
    h : float
        The H-test statistic.

    Returns
    -------
    float
        The probability associated with an H-test statistic of value `h`.
    """
    return np.exp(-0.4 * h)


def gauss_sigma_to_prob(sigma: float) -> float:
    """
    Return the upper-tail probability of the Gaussian distribution.

    This is the area under the Gaussian probability density function
    integrated from `sigma` to infinity.

    Parameters
    ----------
    sigma : float
        The Gaussian sigma (lower integration limit).

    Returns
    -------
    float
        The upper-tail probability. For sigma >= 5, the asymptotic series
        from A&S page 932, eqn 26.2.12 for Q(x) is used.
    """
    if sigma < 5.0:
        return 1.0 - ndtr(sigma)
    else:
        # From A&S page 932, eqn 26.2.12 for Q(x)
        x = sigma
        Z = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * x * x)
        series = np.sum(
            np.asarray(
                [
                    1.0,
                    -1.0 / (x * x),
                    3.0 / (x**4.0),
                    -15.0 / (x**6.0),
                    105.0 / (x**8.0),
                ]
            )
        )
        return Z / x * series


def prob_to_gauss_sigma(prob: float) -> float:
    """
    Return the Gaussian sigma corresponding to a cumulative probability.

    This is the sigma for which the area under the Gaussian probability
    density function (integrated from minus infinity to sigma) equals
    `prob`.

    Parameters
    ----------
    prob : float
        The cumulative Gaussian probability.

    Returns
    -------
    float
        The corresponding Gaussian sigma.
    """
    return ndtri(prob)


def xray_time_to_detect(
    ctrate: float,
    pfract: float,
    dt: float,
    fpsr: float,
    bins: int = 0,
    confidence: float = 0.99,
    detectfract: float = 0.99,
) -> float | None:
    """
    Return the observation duration required to detect X-ray pulsations.

    Assumes no breaks and a sinusoidal pulse profile.

    Parameters
    ----------
    ctrate : float
        The total expected count rate.
    pfract : float
        The expected pulsed fraction.
    dt : float
        The bin duration in sec.
    fpsr : float
        The pulsar frequency in Hz.
    bins : int, optional
        The number of Fourier bins to search. The default of 0 means all
        bins will be examined.
    confidence : float, optional
        The confidence level that the signal is not caused by noise
        (default 0.99).
    detectfract : float, optional
        The fraction of the time you want this observation to result in a
        detection (default 0.99). For example, 0.5 means 50% of
        observations of this duration would detect the specified signal at
        the `confidence` level.

    Returns
    -------
    float or None
        The required observation duration in sec.

    Notes
    -----
    Based on para 1, sect 3.3, of Ransom, Gaensler, and Slane, 2002. This
    routine is currently incomplete: the general (``bins == 0``) case is not
    yet implemented and no duration is returned.
    """
    nyquist_freq = 0.5 / dt
    factor = binning_factor(fpsr, nyquist_freq) ** 2.0
    A = pfract * ctrate  # Signal ct rate
    if bins:
        P_detect = max_noise_power(bins, confidence=confidence)
        power_required = required_signal_power(P_detect, confidence=detectfract)
    # The following is from para 1, sect 3.3, of Ransom, Gaensler, and Slane, 2002
    # return  (power_required - 1.0)          4 * ctrate * dt**2.0 / (A**2.0 * factor) *
    else:
        print("Not implemented yet...I think we need to iterate.")


# The following routines are based on the method of signal
# estimation described by Vaughan et al., 1994, ApJ, 435, p362.
# The math comes from Groth, 1975, ApJS, 29, p285.


def power_average(signal_power: float, n: int = 1) -> float:
    """
    Return the expectation value of the measured power.

    From equation 14 in Groth, 1975.

    Parameters
    ----------
    signal_power : float
        The intrinsic signal power.
    n : int, optional
        The number of summed powers (default 1).

    Returns
    -------
    float
        The expectation value of the measured power.
    """
    return signal_power + n


def power_variance(signal_power: float, n: int = 1) -> float:
    """
    Return the variance of the measured power.

    From equation 14 in Groth, 1975.

    Parameters
    ----------
    signal_power : float
        The intrinsic signal power.
    n : int, optional
        The number of summed powers (default 1).

    Returns
    -------
    float
        The variance of the measured power.
    """
    return 2.0 * signal_power + n


def power_sigma(signal_power: float, n: int = 1) -> float:
    """
    Return the standard deviation of the measured power.

    From equation 14 in Groth, 1975.

    Parameters
    ----------
    signal_power : float
        The intrinsic signal power.
    n : int, optional
        The number of summed powers (default 1).

    Returns
    -------
    float
        The standard deviation of the measured power.
    """
    return np.sqrt(power_variance(signal_power, n))


def log_fact_table(maxn: int) -> np.ndarray:
    """
    Return a table of the natural logs of the first maxn+1 factorials.

    Parameters
    ----------
    maxn : int
        The largest factorial to include.

    Returns
    -------
    numpy.ndarray
        The natural logarithms of the first `maxn`+1 factorials.
    """
    table = np.arange(maxn + 1, dtype="d")
    table[0] = 1.0
    return np.add.accumulate(np.log(table))


def binning_factor(freq: float | np.ndarray, nyquist_freq: float) -> float | np.ndarray:
    """
    Return the amplitude reduction factor due to binning of events.

    High-frequency Fourier amplitudes are decreased when the time series is
    made of binned events. Square this for a power spectrum adjustment.

    Parameters
    ----------
    freq : float or numpy.ndarray
        The frequency (or frequencies) of interest.
    nyquist_freq : float
        The Nyquist frequency, which can be defined as N/(2*T).

    Returns
    -------
    float or numpy.ndarray
        The amplitude reduction factor.
    """
    x = 0.5 * np.asarray(freq) / nyquist_freq
    return np.sinc(x)  # numpy sinc is defined with pi


def max_noise_power(bins: float, n: int = 1, confidence: float = 0.99) -> float:
    """
    Return the power level unlikely to be caused by spectral noise.

    This is P_detect in Vaughan et al., 1994, also known as P_threshold.

    Parameters
    ----------
    bins : float
        The total number of independent frequencies searched.
    n : int, optional
        The number of summed powers (default 1).
    confidence : float, optional
        The confidence that noise could not cause this level (default 0.99).

    Returns
    -------
    float
        The power level giving the specified `confidence` that spectral
        noise could not cause it.
    """
    if n == 1:
        return -np.log((1.0 - confidence) / bins)
    else:
        return 0.5 * chdtri(2.0 * n, (1.0 - confidence) / bins)


def prob_power_series(
    power: float, signal_power: float, n: int = 1, TOL: float = 1.0e-14
) -> float:
    """
    Return the integrated detection probability using an infinite sum.

    Returns the integrated probability from P=0 to `power` that a signal
    with theoretical power `signal_power` will show up in a power spectrum
    with power `power`. This evaluates the integral using an infinite sum
    and is equation 16 in Groth, 1975.

    Parameters
    ----------
    power : float
        The upper limit of the power integration.
    signal_power : float
        The theoretical signal power.
    n : int, optional
        The number of summed powers (default 1).
    TOL : float, optional
        The convergence tolerance for the sum (default 1.0e-14).

    Returns
    -------
    float
        The integrated probability.
    """
    fact = np.exp(-(power + signal_power))
    lf = log_fact_table((power + signal_power) * 5)
    lp, lps = np.log(power), np.log(signal_power)
    sum = 0.0
    term = 1.0
    m = 0
    while 1:
        kmax = m + n
        term = fact * np.add.reduce(
            np.exp((np.arange(kmax) * lp + m * lps) - (lf[0:kmax] + lf[m]))
        )
        sum = sum + term
        if m > signal_power and term < TOL:
            break
        m = m + 1
    return 1.0 - sum


def prob_power_integral(power: float, signal_power: float, n: int = 1) -> float:
    """
    Return the integrated detection probability using numerical integration.

    Returns the integrated probability from P=0 to `power` that a signal
    with theoretical power `signal_power` will show up in a power spectrum
    with power `power`. This evaluates the integral numerically and is
    equation 18 in Groth, 1975.

    Parameters
    ----------
    power : float
        The upper limit of the power integration.
    signal_power : float
        The theoretical signal power.
    n : int, optional
        The number of summed powers (default 1).

    Returns
    -------
    float
        The integrated probability.
    """

    def integrand(theta, p, ps, n):
        t1 = 2 * n * theta
        t2 = np.sin(2.0 * theta)
        A = t1 + ps * t2
        B = t1 + (ps - p) * t2
        sintheta = np.sin(theta)
        sin2theta = sintheta**2.0
        return (
            np.exp(-2.0 * ps * sin2theta)
            * (np.sin(A - theta) - np.exp(-2.0 * p * sin2theta) * np.sin(B - theta))
            / sintheta
        )

    (val, err) = quad(integrand, 0.0, PIBYTWO, (power, signal_power, n))
    return val / PI


def power_probability(power: float, signal_power: float, n: int = 1) -> float:
    """
    Return the probability density of a measured power.

    Returns the probability of a signal with power `signal_power` actually
    showing up with power `power` in a power spectrum. This is equation 12
    in Groth, 1975 and is the integrand of the prob_power_* functions
    (which integrate it from 0 to P).

    Parameters
    ----------
    power : float
        The measured power.
    signal_power : float
        The theoretical signal power.
    n : int, optional
        The number of summed powers (default 1).

    Returns
    -------
    float
        The probability density.
    """
    return (
        (power / signal_power) ** (0.5 * (n - 1))
        * np.exp(-(power + signal_power))
        * iv(n - 1.0, 2 * np.sqrt(power * signal_power))
    )


def required_signal_power(power: float, n: int = 1, confidence: float = 0.99) -> float:
    """
    Return the signal power required to produce a given measured power.

    Returns the required power of a signal that will cause at least a power
    `power` in a power spectrum a fraction `confidence` of the time. This is
    the inverse of equation 16 in Groth, 1975, solved for P_signal. If
    called with ``power = P_detect`` the result is the search sensitivity;
    if called with ``power = P_max``, the result is the upper limit on the
    signal power in the power spectrum.

    Parameters
    ----------
    power : float
        The measured power level.
    n : int, optional
        The number of summed powers (default 1).
    confidence : float, optional
        The fractional confidence (default 0.99).

    Returns
    -------
    float
        The required signal power.
    """
    prob = 1.0 - confidence

    def func(x, power=power, prob=prob, n=n):
        return prob_power_series(power, x, n) - prob

    def dfunc(x, power=power, n=n):
        return power_probability(power, x, n)

    P_signal = newton_raphson(func, dfunc, 0.0001, 100.0)
    return P_signal


def fft_sensitivity(
    N: int, bins: float = 0, n: int = 1, confidence: float = 0.99
) -> float:
    """
    Return the weakest signal power detectable in an FFT search.

    This calculation does not include the correction to sensitivity due to
    binning effects. Based on the Vaughan et al 1994 paper; computes P_sens.

    Parameters
    ----------
    N : int
        The number of bins in the time series (the number of frequency bins
        searched is usually N/2).
    bins : float, optional
        The number of independent frequencies searched, if different from
        N/2 (e.g. for an acceleration search). The default of 0 uses N/2.
    n : int, optional
        The number of summed powers (default 1).
    confidence : float, optional
        The fractional confidence in the result (default 0.99).

    Returns
    -------
    float
        The weakest confidently detectable signal power (P_sens).
    """
    if not (bins):
        bins = N / 2
    P_threshold = max_noise_power(bins, n, confidence)
    return required_signal_power(P_threshold, n, confidence)


def rzw_sensitivity(
    N: int,
    zlo: float = -100.0,
    zhi: float = 100.0,
    n: int = 1,
    confidence: float = 0.99,
) -> float:
    """
    Return the weakest signal power detectable in an RZW acceleration search.

    This calculation does not include the correction to sensitivity due to
    binning effects. Based on the Vaughan et al 1994 paper; computes P_sens.

    Parameters
    ----------
    N : int
        The number of bins in the time series.
    zlo : float, optional
        The low acceleration (z) value searched (default -100.0).
    zhi : float, optional
        The high acceleration (z) value searched (default 100.0).
    n : int, optional
        The number of summed powers (default 1).
    confidence : float, optional
        The fractional confidence in the result (default 0.99).

    Returns
    -------
    float
        The weakest confidently detectable signal power (P_sens).
    """
    bins = N / 2.0 * (zhi - zlo + 1.0) / 6.95
    P_threshold = max_noise_power(bins, n, confidence)
    return required_signal_power(P_threshold, n, confidence)


def binned_fft_sensitivity(
    N: int,
    dt: float,
    freq: float,
    bins: float = 0,
    n: int = 1,
    confidence: float = 0.99,
) -> float:
    """
    Return the weakest detectable signal power in a binned FFT search.

    Includes the correction to sensitivity due to binning effects. Based on
    the Vaughan et al 1994 paper; computes P_sens.

    Parameters
    ----------
    N : int
        The number of bins in the time series (the number of frequency bins
        searched is usually N/2), each binned into `dt` sec bins.
    dt : float
        The bin duration in sec.
    freq : float
        The signal frequency in Hz.
    bins : float, optional
        The number of independent frequencies searched, if different from
        N/2 (e.g. for an acceleration search). The default of 0 uses N/2.
    n : int, optional
        The number of summed powers (default 1).
    confidence : float, optional
        The fractional confidence in the result (default 0.99).

    Returns
    -------
    float
        The weakest confidently detectable signal power (P_sens) at `freq`.
    """
    nyquist_freq = 0.5 / dt
    factor = binning_factor(freq, nyquist_freq) ** 2.0
    return fft_sensitivity(N, bins, n, confidence) / factor


def binned_rzw_sensitivity(
    N: int,
    dt: float,
    freq: float,
    zlo: float = -100.0,
    zhi: float = 100.0,
    n: int = 1,
    confidence: float = 0.99,
) -> float:
    """
    Return the weakest detectable signal power in a binned RZW search.

    Includes the correction to sensitivity due to binning effects. Based on
    the Vaughan et al 1994 paper; computes P_sens.

    Parameters
    ----------
    N : int
        The number of bins in the time series, each binned into `dt` sec
        bins.
    dt : float
        The bin duration in sec.
    freq : float
        The signal frequency in Hz.
    zlo : float, optional
        The low acceleration (z) value searched (default -100.0).
    zhi : float, optional
        The high acceleration (z) value searched (default 100.0).
    n : int, optional
        The number of summed powers (default 1).
    confidence : float, optional
        The fractional confidence in the result (default 0.99).

    Returns
    -------
    float
        The weakest confidently detectable signal power (P_sens) at `freq`.
    """
    bins = N / 2.0 * (zhi - zlo + 1.0) / 6.95
    nyquist_freq = 0.5 / dt
    factor = binning_factor(freq, nyquist_freq) ** 2.0
    return fft_sensitivity(N, bins, n, confidence) / factor


def pulsed_fraction_limit(Nphot: float, Pow: float) -> float:
    """
    Return an observational upper limit to the pulsed fraction of a signal.

    This is an *observational* (i.e. not intrinsic) upper limit to the
    pulsed fraction of a signal that is in the data but was not detected.
    "Observational" means that some of the unpulsed events do not come from
    the source you are searching for pulsations in. For the *intrinsic*
    pulsed fraction, divide the returned value by the fraction of `Nphot`
    that actually comes from the source (i.e. the NS).

    Parameters
    ----------
    Nphot : float
        The total number of photons in the data.
    Pow : float
        The largest measured power (or P_sens as calculated using the
        ``*_sensitivity`` functions in this module).

    Returns
    -------
    float
        The observational upper limit to the pulsed fraction.
    """
    return np.sqrt(4.0 * (Pow - 1.0) / Nphot)


if __name__ == "__main__":
    import presto.psr_utils as pu
    import presto.presto as pp
    from presto.Pgplot import plotxy, closeplot

    prof = pu.expcos_profile(128, 0.0, 0.1) + np.random.standard_normal(128)
    plotxy(prof)
    closeplot()
    fprof = pp.rfft(prof)
    fprof = fprof / np.sqrt(fprof[0].real)
    pows = pp.spectralpower(fprof)
    tcsum = np.add.accumulate(np.sqrt(pows[1:10])) ** 2.0
    csum = coherent_sum(fprof[1:10])
    isum = incoherent_sum(fprof[1:10])
    print(isum)
    print(csum)
    print(tcsum)
    for ii in range(len(csum)):
        print(
            pp.candidate_sigma(float(isum[ii]), ii + 1, 1),
            pp.candidate_sigma(csum[ii] / (ii + 1), 1, 1),
        )

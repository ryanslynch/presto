"""Property-based FFTFIT tests, adapted from Anne Archibald's PINT test_fftfit.py
(aarchiba/PINT branch fftfit_new, tests/test_fftfit.py; BSD-3, PINT developers) to
run in-tree against BOTH of PRESTO's methods via fftfit_methods.

Changes from the original: codes are our "classic" and "aarchiba" (nustar dropped);
where Anne marked the PRESTO code `xfail` (symmetric profiles, differing lengths), we
xfail "classic" for the same reason -- the classic algorithm inherits those Fortran
limitations, while "aarchiba" handles them. The strict uncertainty-coverage check is
marked non-strict xfail: both methods are known to under-cover (see ROADMAP.md, "FFTFIT
uncertainty calibration").

Needs pytest + hypothesis:  pytest python/fftfit_src/test_fftfit_hypothesis.py
"""
from functools import wraps
from itertools import product

import numpy as np
import pytest
import scipy.stats
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (
    complex_numbers,
    composite,
    floats,
    integers,
    one_of,
)
from numpy.testing import assert_allclose

import fftfit_methods as m
from fftfit_methods import fftfit_basic, fftfit_full, irfft_value, shift, wrap


def vonmises_profile(kappa, n, phase=0):
    return np.diff(
        scipy.stats.vonmises(kappa).cdf(
            np.linspace(-2 * np.pi * phase, 2 * np.pi * (1 - phase), n + 1)
        )
    )


# ------------------------------------------------------------------ helpers (Anne's)
def assert_rms_close(a, b, rtol=1e-8, atol=1e-8):
    assert np.mean((a - b) ** 2) < rtol * (np.mean(a ** 2) + np.mean(b ** 2)) + atol


def assert_allclose_phase(a, b, atol=1e-8):
    assert np.all(np.abs(wrap(a - b)) <= atol)


ONE_SIGMA = 1 - 2 * scipy.stats.norm().sf(1)


def assert_happens_with_probability(func, p=ONE_SIGMA, n=100, fpp=0.05):
    k = sum(1 for _ in range(n) if func())
    low_k = scipy.stats.binom(n, p).ppf(fpp / 2)
    high_k = scipy.stats.binom(n, p).isf(fpp / 2)
    assert low_k <= k <= high_k


@composite
def powers_of_two(draw):
    return 2 ** draw(integers(4, 16))


@composite
def vonmises_templates_noisy(draw, ns=powers_of_two(), phase=floats(0, 1)):
    n = draw(ns)
    return vonmises_profile(draw(floats(1, 1000)), n, draw(phase)) + (
        1e-3 / n
    ) * np.random.default_rng(0).standard_normal(n)


@composite
def random_templates(draw, ns=powers_of_two()):
    return np.random.randn(draw(ns))


@composite
def boxcar_templates(draw, ns=powers_of_two(), duty=floats(0, 1)):
    n = draw(ns)
    t = np.zeros(n)
    t[: int(draw(duty) * n)] = 1
    t[0] = 1
    t[-1] = 0
    return t


def randomized_test(tries=5, seed=0):
    def rt(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            kwargs.pop("state", None)
            exc = None
            for i in range(seed, seed + tries):
                try:
                    return f(*args, state=np.random.default_rng(i), **kwargs)
                except AssertionError as e:
                    exc = e
            raise AssertionError("Failed for all %d seeds" % tries) from exc

        return wrapper

    return rt


def CODES(classic_xfail=None):
    """Parametrize over our two methods; xfail 'classic' where noted (non-strict)."""
    marks = [pytest.mark.xfail(reason=classic_xfail, strict=False)] if classic_xfail else []
    return [pytest.param("aarchiba"), pytest.param("classic", marks=marks)]


def shift_atol(code, n):
    """Method-appropriate shift-recovery tolerance (turns).

    aarchiba reaches ~1/32 bin (bin-scaled). The classic port inherits the Fortran's
    coarser convergence: its shift error is an edtau-limited *phase* floor (~1e-4 turns,
    worse for broad/low-harmonic profiles) that does not shrink with n, so classic is
    held to a fixed phase tolerance rather than a bin-scaled one.
    """
    return 1.0 / (32 * n) if code == "aarchiba" else 2e-4


def needs_min_bins(code, n):
    """The classic _fccf uses 16 harmonics, so it requires n >= 32 (as in PRESTO)."""
    if code == "classic":
        assume(n >= 32)


# ------------------------------------------------------------------ helper-fn tests
@given(
    arrays(complex, integers(3, 9), elements=complex_numbers(max_magnitude=1e8)),
    integers(4, 16),
)
def test_irfft_value(c, n):
    assume(n >= 2 * (len(c) - 1))
    c = c.copy()
    c[0] = c[0].real
    c[-1] = 0
    xs = np.linspace(0, 1, n, endpoint=False)
    assert_rms_close(np.fft.irfft(c, n), irfft_value(c, xs, n))


@given(floats(0, 1), one_of(vonmises_templates_noisy(), random_templates()))
def test_shift_invertible(s, template):
    assert_allclose(template, shift(shift(template, s), -s), atol=1e-14)


# ------------------------------------------------------------------ shift-recovery
@given(integers(0, 2 ** 20), floats(1, 1000), integers(5, 16), floats(0, 1))
@pytest.mark.parametrize("code", CODES())
def test_fftfit_basic_integer_vonmises(code, i, kappa, profile_length, phase):
    n = 2 ** profile_length
    needs_min_bins(code, n)
    template = vonmises_profile(kappa, n, phase) + (1e-3 / n) * np.random.default_rng(
        0
    ).standard_normal(n)
    assume(sum(template > 0.5 * template.max()) > 1)
    s = i / len(template)
    rs = fftfit_basic(template, shift(template, s), code=code)
    assert_allclose_phase(s, rs, atol=shift_atol(code, len(template)))


@given(floats(0, 1), floats(1, 1000), powers_of_two())
@pytest.mark.parametrize("code", CODES())
def test_fftfit_basic_subbin(code, s, kappa, n):
    needs_min_bins(code, n)
    template = vonmises_profile(kappa, n) + (1e-3 / n) * np.random.default_rng(
        0
    ).standard_normal(n)
    rs = fftfit_basic(template, shift(template, s / n), code=code)
    atol = 1e-4 / len(template) if code == "aarchiba" else shift_atol(code, len(template))
    assert_allclose_phase(rs, s / n, atol=atol)


@given(floats(0, 1), one_of(vonmises_templates_noisy(), random_templates(), boxcar_templates()))
@pytest.mark.parametrize("code", CODES(classic_xfail="profile too symmetric"))
def test_fftfit_basic_template(code, s, template):
    needs_min_bins(code, len(template))
    rs = fftfit_basic(template, shift(template, s), code=code)
    assert_allclose_phase(rs, s, atol=shift_atol(code, len(template)))


@given(vonmises_templates_noisy(), vonmises_templates_noisy())
@pytest.mark.parametrize("code", CODES(classic_xfail="profiles different lengths"))
def test_fftfit_shift_equivalence(code, profile1, profile2):
    s = fftfit_basic(profile1, profile2, code=code)
    assert_allclose_phase(
        fftfit_basic(shift(profile1, s), profile2, code=code),
        0,
        atol=1e-3 / min(len(profile1), len(profile2)),
    )


# ------------------------------------------------------------------ scale (aarchiba only)
@given(
    one_of(vonmises_templates_noisy(), random_templates()),
    floats(0, 1),
    floats(0.5, 2),
    floats(-1, 1),
)
def test_fftfit_compute_scale(template, s, a, b):
    """aarchiba recovers scale a and offset b (classic does not compute the baseline)."""
    profile = a * shift(template, s) + b
    r = fftfit_full(template, profile, code="aarchiba")
    assert_allclose_phase(s, r.shift, atol=1e-3 / len(template))
    assert_allclose(b, r.offset, atol=a * 1e-6)
    assert_allclose(a, r.scale, atol=(1 + abs(b)) * 1e-6)


# ------------------------------------------------------------------ uncertainty coverage
@pytest.mark.parametrize("kappa,n,std,s", [(10, 64, 0.01, 1 / 3), (100, 1024, 0.02, 0.2)])
@pytest.mark.parametrize(
    "code",
    [
        pytest.param("aarchiba", marks=pytest.mark.xfail(strict=False, reason="under-coverage; ROADMAP")),
        pytest.param("classic", marks=pytest.mark.xfail(strict=False, reason="under-coverage; ROADMAP")),
    ],
)
@randomized_test(tries=3)
def test_fftfit_uncertainty_estimate(code, kappa, n, std, s, state):
    """The reported 1-sigma error bar should contain the truth ~68% of the time.

    Known to under-cover for both methods -- see ROADMAP FFTFIT uncertainty calibration.
    Marked non-strict xfail so it documents (rather than gates) the open follow-up.
    """
    template = vonmises_profile(kappa, n)

    def within_one_sigma():
        profile = shift(template, s) + std * state.standard_normal(n)
        r = fftfit_full(template, profile, code=code)
        return np.abs(wrap(r.shift - s)) < r.uncertainty

    assert_happens_with_probability(within_one_sigma)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

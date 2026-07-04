#!/usr/bin/env python
from __future__ import annotations

import numpy as np


def get_frac_harmonic(
    num: int | float, denom: int | float, spectrum: np.ndarray
) -> np.ndarray:
    """
    Return spectrum values corresponding to the harmonic num/denom.

    Parameters
    ----------
    num : int or float
        Numerator of the fractional harmonic.
    denom : int or float
        Denominator of the fractional harmonic.
    spectrum : numpy.ndarray
        The spectrum to harmonic-sum.

    Returns
    -------
    numpy.ndarray
        The selected elements of `spectrum` for the harmonic num/denom
        (same dtype as `spectrum`).
    """
    inds = np.arange(len(spectrum), dtype=np.float64)
    # get the closest indices for the fractional harmonics
    # they need to be integers to use them as indices
    bins = np.round(inds * num / denom).astype(np.int64)
    return spectrum[bins]


def harmonic_sum(
    numharm: int,
    spectrum: np.ndarray,
    partial: np.ndarray | None = None,
    partialN: int = 1,
) -> np.ndarray:
    """
    Perform a top-down harmonic sum of a spectrum.

    Parameters
    ----------
    numharm : int
        The number of harmonics to sum (2, 4, 8, 16, 32, or 64).
    spectrum : numpy.ndarray
        The spectrum to harmonic-sum.
    partial : numpy.ndarray, optional
        A partially harmonic-summed spectrum to continue from (default None,
        which starts a fresh sum).
    partialN : int, optional
        The number of harmonics already summed in `partial` (default 1).

    Returns
    -------
    numpy.ndarray
        The harmonic-summed spectrum (same dtype as `spectrum`).
    """
    if partialN == 1:
        partial = spectrum.copy()  # This is the high harmonic
    if numharm > partialN and partialN < 2:
        np.add(partial, get_frac_harmonic(1, 2, spectrum), partial)
        partialN = 2
    if numharm > partialN and partialN < 4:
        # Note that the 1/2 == 2/4 has already been added
        np.add(partial, get_frac_harmonic(1, 4, spectrum), partial)
        np.add(partial, get_frac_harmonic(3, 4, spectrum), partial)
        partialN = 4
    if numharm > partialN and partialN < 8:
        # 2/8, 4/8, and 6/8 have already been added
        for ii in [1, 3, 5, 7]:
            np.add(partial, get_frac_harmonic(ii, 8, spectrum), partial)
        partialN = 8
    if numharm > partialN and partialN < 16:
        # [even]/16 have all been added
        for ii in np.arange(1, 16, 2):
            np.add(partial, get_frac_harmonic(ii, 16, spectrum), partial)
        partialN = 16
    if numharm > partialN and partialN < 32:
        # [even]/32 have all been added
        for ii in np.arange(1, 32, 2):
            np.add(partial, get_frac_harmonic(ii, 32, spectrum), partial)
        partialN = 32
    if numharm > partialN and partialN < 64:
        # [even]/64 have all been added
        for ii in np.arange(1, 64, 2):
            np.add(partial, get_frac_harmonic(ii, 64, spectrum), partial)
        partialN = 64
    return partial


if __name__ == "__main__":
    s = np.arange(16)  # This shows that indices are correct
    # s = np.ones(16)   # This shows that the sums are correct
    print("Original:")
    print(s)
    print("2harm:")
    p2 = harmonic_sum(2, s)
    print(p2)
    print("4harm: (following two should be the same)")
    print(harmonic_sum(4, s))
    p4 = harmonic_sum(4, s, partial=p2, partialN=2)
    print(p4)  # should be the same as the last one
    print("8harm: (following two should be the same)")
    print(harmonic_sum(8, s))
    p8 = harmonic_sum(8, s, partial=p4, partialN=4)
    print(p8)  # should be the same as the last one

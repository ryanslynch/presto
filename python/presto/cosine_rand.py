from __future__ import annotations

import json
from importlib.resources import files

import numpy as np


# Number of points in the inverse-CDF lookup table (which has n+1 entries).
n = 1000

# The precomputed inverse-CDF lookup table for a sinusoid peaked at phase 0,
# loaded as package data from cosine_rand.json.
xs = json.loads((files("presto") / "cosine_rand.json").read_text())


def regenerate_xs() -> list[float]:
    """
    Regenerate the inverse-CDF lookup table used by :func:`cosine_rand`.

    Solves ``sin(2*pi*x) / (2*pi) + x = r`` for x at ``n+1`` evenly spaced
    values of r in [0, 1]. This inverts the cumulative distribution function
    of a sinusoid peaked at phase 0, so that linear interpolation into the
    resulting table maps uniform deviates onto that distribution. The result
    is what is stored (as JSON) in ``cosine_rand.json``.

    Returns
    -------
    list of float
        The ``n+1`` lookup-table values. Dump this to ``cosine_rand.json``
        (e.g. with ``json.dump``) to update the shipped table.
    """
    from presto.simple_roots import newton_raphson

    rs = np.arange(n + 1, dtype=float) / n
    new_xs = np.zeros(n + 1, dtype=float)
    for ii, rval in enumerate(rs[:-1]):
        if ii == n // 2:
            new_xs[ii] = 0.5
        else:

            def func(x, rval=rval):
                return np.sin(2.0 * np.pi * x) / (2.0 * np.pi) + x - rval

            def dfunc(x):
                return np.cos(2.0 * np.pi * x) + 1

            new_xs[ii] = newton_raphson(func, dfunc, 0.0, 1.0)
    new_xs[0] = 0.0
    new_xs[n] = 1.0
    return new_xs.tolist()


def cosine_rand(num: int) -> np.ndarray:
    """
    Return `num` random phases distributed as a sinusoid peaked at phase 0.

    Parameters
    ----------
    num : int
        The number of random phases to generate.

    Returns
    -------
    numpy.ndarray
        `num` phases in [0, 1), distributed as a sinusoid with its maximum
        at phase 0, drawn by inverse-CDF sampling of the precomputed lookup
        table `xs`.
    """
    rands = n * np.random.random(num)
    indices = rands.astype(int)
    fracts = rands - indices
    lo = np.take(xs, indices)
    hi = np.take(xs, indices + 1)
    return fracts * (hi - lo) + lo


if __name__ == "__main__":
    from presto.psr_utils import hist
    from presto.Pgplot import plotxy, closeplot

    rs = np.arange(n + 1, dtype=float) / n
    plotxy(xs, rs)
    closeplot()
    hist(cosine_rand(10000), 100, color="red")
    closeplot()

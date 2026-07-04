"""
'Safe' Newton-Raphson, secant, and bisection methods for numerical root-finding.

Written by Scott M. Ransom <sransom@nrao.edu>
"""

from __future__ import annotations

from collections.abc import Callable


def bisect(
    func: Callable[[float], float],
    lox: float,
    hix: float,
    TOL: float = 1e-14,
    MAXIT: int = 200,
) -> float:
    """
    Find a root by bisection of a bracketing interval.

    Parameters
    ----------
    func : callable
        A function of one variable whose root is sought.
    lox : float
        The lower end of the bracketing interval.
    hix : float
        The upper end of the bracketing interval.
    TOL : float, optional
        The absolute tolerance on the root (default 1e-14).
    MAXIT : int, optional
        The maximum number of iterations (default 200).

    Returns
    -------
    float
        The estimated root, or 0.0 if the interval does not bracket a root
        or the iteration limit is exceeded.
    """
    f = func(lox)
    fmid = func(hix)
    if f * fmid >= 0.0:
        print("Root must be bracketed in bisect()!")
        return 0.0
    if f < 0.0:
        dx, rtb = hix - lox, lox
    else:
        dx, rtb = lox - hix, hix
    for i in range(MAXIT):
        dx = dx * 0.5
        xmid = rtb + dx
        fmid = func(xmid)
        if fmid <= 0.0:
            rtb = xmid
        if abs(dx) < TOL or fmid == 0.0:
            return rtb
    print("Too many bisections in bisect()!")
    return 0.0


def secant(
    func: Callable[[float], float], oldx: float, x: float, TOL: float = 1e-14
) -> float:
    """
    Find a root by the secant method.

    Similar to Newton's method, but the derivative is estimated by a divided
    difference using only function calls. A root is estimated by
    ``x = x - f(x) (x - oldx) / (f(x) - f(oldx))``, where ``oldx = x[i-1]``
    and ``x = x[i]``.

    Parameters
    ----------
    func : callable
        A function of one variable whose root is sought.
    oldx : float
        The first initial estimate of the root.
    x : float
        The second initial estimate of the root.
    TOL : float, optional
        The relative tolerance on the root (default 1e-14).

    Returns
    -------
    float
        The estimated root.
    """
    oldf, f = func(oldx), func(x)
    if abs(f) > abs(oldf):
        oldx, x = x, oldx
        oldf, f = f, oldf
    count = 0
    while 1:
        dx = f * (x - oldx) / float(f - oldf)
        if abs(dx) < TOL * (1 + abs(x)):
            return x - dx
        oldx, x = x, x - dx
        oldf, f = f, func(x)
        count = count + 1
        # print "secant(%d): x=%s, f(x)=%s" % (count, x, f)


def newton_raphson(
    func: Callable[[float], float],
    dfunc: Callable[[float], float],
    lox: float,
    hix: float,
    TOL: float = 1.0e-14,
) -> float:
    """
    Find a root by a safe Newton-Raphson method.

    Finds the root of `func`, which is bracketed by `lox` and `hix`, to an
    accuracy of +/- `TOL`. The algorithm is a safe version of Newton-Raphson
    (see page 366 of Numerical Recipes in C, 2nd ed.).

    Parameters
    ----------
    func : callable
        A function of one variable whose root is sought.
    dfunc : callable
        The derivative of `func`.
    lox : float
        The lower end of the bracketing interval.
    hix : float
        The upper end of the bracketing interval.
    TOL : float, optional
        The absolute tolerance on the root (default 1.0e-14).

    Returns
    -------
    float
        The estimated root, or 0.0 if the interval does not bracket a root
        or the iteration limit is exceeded.
    """
    maxit = 500
    fl, fh = func(lox), func(hix)
    if (fl > 0.0 and fh > 0.0) or (fl < 0.0 and fh < 0.0):
        print("Root must be bracketed in newtonRaphson()")
        return 0.0
    if fl == 0.0:
        return lox
    if fh == 0.0:
        return hix
    if fl < 0.0:
        xl = lox
        xh = hix
    else:
        xh = lox
        xl = hix
    rts = 0.5 * (lox + hix)
    dxold = abs(hix - lox)
    dx = dxold
    f, df = func(rts), dfunc(rts)
    for j in range(maxit):
        if (((rts - xh) * df - f) * ((rts - xl) * df - f) > 0.0) or (
            abs(2.0 * f) > abs(dxold * df)
        ):
            dxold = dx
            dx = 0.5 * (xh - xl)
            rts = xl + dx
            if xl == rts:
                return rts
        else:
            dxold = dx
            dx = f / df
            temp = rts
            rts = rts - dx
            if temp == rts:
                return rts
        if abs(dx) < TOL:
            return rts
        f, df = func(rts), dfunc(rts)
        if f < 0.0:
            xl = rts
        else:
            xh = rts
    print("Maximum number of iterations exceeded in newton_raphson()")
    return 0.0


# Test code

if __name__ == "__main__":
    import numpy as np

    def func(x):
        return np.sin(x) - x + np.pi / 4.0

    def dfunc(x):
        return np.cos(x) - 1

    nr = newton_raphson(func, dfunc, 0.0, 3.0)
    s = secant(func, 0.0, 3.0)
    bs = bisect(func, 0.0, 3.0)
    theo = 1.766340286602865756325301235707
    print("")
    print("Finding the root (between 0.0 and 3.0) of:")
    print("    x - sin(x) = pi/4")
    print("")
    print("         Newton-Raphson gives (default accuracy) = %15.14f" % nr)
    print("          Secant method gives (default accuracy) = %15.14f" % s)
    print("       Bisection method gives (default accuracy) = %15.14f" % bs)
    print("Theoretical result (correct to all shown digits) = %15.14f" % theo)
    print("")

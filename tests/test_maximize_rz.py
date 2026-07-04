"""Correctness harness for the f-fdot optimizers max_rz_arr() and
max_rz_arr_harmonics() (src/maximize_rz.c), exposed as presto.maximize_rz and
presto.maximize_rz_harmonics.

Two ways to use this file:

* As a test (``pytest tests/test_maximize_rz.py`` or ``python
  tests/test_maximize_rz.py``): injects signals at known Fourier (r, z), runs
  the optimizers from perturbed starting points, and asserts that they recover
  the true peak location and power.  These are physical tolerances, so they are
  robust to the (small, FFTW_MEASURE-driven) run-to-run jitter in the FFT
  results and do not need FFTW wisdom to be present.

* As an A/B golden harness for swapping the optimizer's solver (this file was
  written to validate replacing the Numerical-Recipes amoeba() downhill simplex
  with GSL's nmsimplex2):

      python tests/test_maximize_rz.py record  before.json   # old solver build
      python tests/test_maximize_rz.py compare before.json   # new solver build

  The record/compare diff is only meaningful at full precision if the FFTs are
  deterministic across processes -- generate FFTW wisdom first (`makewisdom`),
  otherwise expect ~1e-3-bin FFTW_MEASURE noise between runs of even the *same*
  binary.
"""

import sys
import json

import numpy as np

from presto import presto

N = 2**14                       # length of the injected time series
R_RZ = N / 4.0                  # fundamental r for the single-harmonic battery
R_HARM = N / 16.0               # fundamental r for harmonics (4th still < Nyquist)
THEO_POW = N ** 2.0 / 4.0       # theoretical power of one full-strength harmonic

# Deterministic (dr, dz) start offsets from the true peak.
RZ_OFFSETS = [(0.0, 0.0), (0.12, -0.8), (-0.2, 1.3), (0.3, 2.0), (-0.35, -2.5)]
HARM_OFFSETS = [(0.0, 0.0), (0.15, 1.0), (-0.25, -1.5)]
Z_VALUES = [0.0, 1.0, 2.5, 5.0, -5.0, 10.0, -10.0, 30.0, -30.0, 55.0]
HARM_Z_VALUES = [0.0, 5.0, -8.0, 20.0]
NOISE_CASES = [(0.0, 0), (1.0, 12345)]   # (amplitude, seed)

# Recovery tolerances, split by whether noise was injected (noise shifts the
# true peak by a fraction of a bin).  Comfortably above the observed GSL errors
# (noiseless: <0.003 bins; noisy: <0.11 bins) yet far below the ~1.6-bin / 2.8x
# error of a solver that gets stuck (which is exactly how the old amoeba()
# failed when started right on a peak) -- so these still catch a real regression.
TOL = {
    #                 |dr|   |dz|   powfrac_lo  powfrac_hi
    0.0: dict(dr=0.01, dz=0.03, plo=0.99, phi=1.01),
    1.0: dict(dr=0.05, dz=0.20, plo=0.90, phi=1.12),
}


def make_signal(r, z, noise_amp=0.0, seed=0):
    """rfft of one sinusoid at Fourier frequency r and f-dot z."""
    us = np.arange(N, dtype=np.float64) / N
    r0 = r - 0.5 * z
    phss = 2.0 * np.pi * (us * (us * (z / 2.0) + r0))
    sig = np.cos(phss)
    if noise_amp:
        sig = sig + noise_amp * np.random.default_rng(seed).standard_normal(N)
    return presto.rfft(sig, -1)


def make_harmonic_signal(r, z, numharm, noise_amp=0.0, seed=0):
    """rfft of a sum of numharm harmonics at (h*r, h*z), h = 1..numharm."""
    us = np.arange(N, dtype=np.float64) / N
    sig = np.zeros(N, dtype=np.float64)
    for h in range(1, numharm + 1):
        rh, zh = r * h, z * h
        phss = 2.0 * np.pi * (us * (us * (zh / 2.0) + (rh - 0.5 * zh)))
        sig = sig + np.cos(phss)
    if noise_amp:
        sig = sig + noise_amp * np.random.default_rng(seed).standard_normal(N)
    return presto.rfft(sig, -1)


def _derivs_dict(rd):
    return {k: getattr(rd, k) for k in
            ("pow", "phs", "dpow", "dphs", "d2pow", "d2phs", "locpow")}


def run_cases():
    """Run the full deterministic battery; return a list of result records."""
    results = []
    for noise_amp, seed in NOISE_CASES:
        for z in Z_VALUES:
            ft = make_signal(R_RZ, z, noise_amp, seed)
            for dr, dz in RZ_OFFSETS:
                maxpow, rmax, zmax, rd = presto.maximize_rz(
                    ft, R_RZ + dr, z + dz, norm=1.0)
                results.append(dict(
                    kind="rz", r_true=R_RZ, z_true=z, theo_pow=THEO_POW,
                    noise_amp=noise_amp, seed=seed,
                    r_start=R_RZ + dr, z_start=z + dz,
                    maxpow=maxpow, rmax=rmax, zmax=zmax,
                    derivs=_derivs_dict(rd)))

    for numharm in (2, 3, 4):
        for noise_amp, seed in NOISE_CASES:
            for z in HARM_Z_VALUES:
                ft = make_harmonic_signal(R_HARM, z, numharm, noise_amp, seed)
                for dr, dz in HARM_OFFSETS:
                    maxpow, rmax, zmax, rds = presto.maximize_rz_harmonics(
                        ft, R_HARM + dr, z + dz, numharm, norm=1.0)
                    results.append(dict(
                        kind="rz_harmonics", numharm=numharm,
                        r_true=R_HARM, z_true=z, theo_pow=numharm * THEO_POW,
                        noise_amp=noise_amp, seed=seed,
                        r_start=R_HARM + dr, z_start=z + dz,
                        maxpow=maxpow, rmax=rmax, zmax=zmax,
                        derivs=[_derivs_dict(rd) for rd in rds]))
    return results


def _check_recovery(results):
    """Return a list of human-readable failure strings (empty == all good)."""
    failures = []
    for i, c in enumerate(results):
        tol = TOL[c["noise_amp"]]
        adr = abs(c["rmax"] - c["r_true"])
        adz = abs(c["zmax"] - c["z_true"])
        pf = c["maxpow"] / c["theo_pow"]
        tag = c["kind"] + (f"[nh{c['numharm']}]" if "numharm" in c else "")
        if adr > tol["dr"] or adz > tol["dz"] or not (tol["plo"] <= pf <= tol["phi"]):
            failures.append(
                f"case {i} {tag} z={c['z_true']} na={c['noise_amp']}: "
                f"|dr|={adr:.3g} (<{tol['dr']}) |dz|={adz:.3g} (<{tol['dz']}) "
                f"powfrac={pf:.4f} ([{tol['plo']},{tol['phi']}])")
    return failures


# --------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------
def test_maximize_rz_recovers_injected_peaks():
    """Both optimizers recover the injected (r, z) and power within tolerance."""
    failures = _check_recovery(run_cases())
    assert not failures, "\n".join([f"{len(failures)} case(s) failed:"] + failures)


# --------------------------------------------------------------------------
# Golden record/compare (manual A/B for solver swaps)
# --------------------------------------------------------------------------
def _leaves(rec):
    out = {"maxpow": rec["maxpow"], "rmax": rec["rmax"], "zmax": rec["zmax"]}
    d = rec["derivs"]
    items = enumerate(d) if isinstance(d, list) else [(None, d)]
    for idx, dd in items:
        for k, v in dd.items():
            out[f"derivs{'' if idx is None else f'[{idx}]'}.{k}"] = v
    return out


def compare(golden, current):
    if len(golden) != len(current):
        print(f"CASE COUNT MISMATCH: {len(golden)} vs {len(current)}")
        return False
    dr = [abs(g["rmax"] - c["rmax"]) for g, c in zip(golden, current)]
    dz = [abs(g["zmax"] - c["zmax"]) for g, c in zip(golden, current)]
    rp = [abs(g["maxpow"] - c["maxpow"]) / (abs(g["maxpow"]) + 1e-300)
          for g, c in zip(golden, current)]
    for name, arr in (("|dr| ", dr), ("|dz| ", dz), ("|dpow|/pow", rp)):
        a = np.asarray(arr)
        print(f"  {name}: median={np.median(a):.2e} p90={np.percentile(a,90):.2e} "
              f"max={a.max():.2e}")
    worst = max(range(len(golden)), key=lambda i: dz[i])
    print(f"  worst-|dz| case {worst}: golden z={golden[worst]['zmax']:.6f} "
          f"current z={current[worst]['zmax']:.6f}")
    return True


def main(argv):
    if len(argv) >= 2 and argv[1] in ("record", "compare"):
        results = run_cases()
        path = argv[2] if len(argv) > 2 else "rz_golden.json"
        if argv[1] == "record":
            with open(path, "w") as f:
                json.dump(results, f, indent=1)
            print(f"recorded {len(results)} cases -> {path}")
        else:
            with open(path) as f:
                compare(json.load(f), results)
        return 0
    # default: run the recovery check like a plain script
    failures = _check_recovery(run_cases())
    if failures:
        print("\n".join([f"FAILED ({len(failures)} cases):"] + failures))
        return 1
    print("maximize_rz peak-recovery checks: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

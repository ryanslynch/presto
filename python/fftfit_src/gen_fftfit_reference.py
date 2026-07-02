#!/usr/bin/env python
"""Phase 0 of the FFTFIT pure-Python rewrite: freeze a golden reference dataset.

This drives the *current* Fortran FFTFIT extension (`presto.fftfit`) over a broad
battery of inputs and stores the inputs together with all outputs. The resulting
`.npz` file is the primary regression oracle against which the pure-Python port is
validated (see ROADMAP.md, "FFTFIT pure-Python rewrite").

Two families of cases are generated:

  1. Synthetic  -- analytic profiles (Gaussian / double-Gaussian / von Mises) with
     known true shifts, over a range of nbins, shifts, and S/N (including the
     low-S/N regime where the Fortran bails out with eshift=999). Noise is drawn
     from a seeded RNG, but the *actual* profile arrays are stored, so validation
     does not depend on reproducing any particular RNG stream.

  2. Real data  -- profiles pulled straight from a prepfold fold in
     tests/prepfold/goodfolds/, exercising FFTFIT the way get_TOAs.py does
     (a high-S/N summed template vs. per-part sub-profiles), but WITHOUT pulling in
     TEMPO/polycos: we freeze the raw FFTFIT shift/eshift numbers only.

Run from the repo top level (needs $PRESTO data + a built `presto.fftfit`):

    python python/fftfit_src/gen_fftfit_reference.py

Output: python/fftfit_src/fftfit_reference.npz
"""

import os
import sys
import numpy as np

from presto.fftfit import cprof, fftfit
from presto.psr_utils import gaussian_profile

TWOPI = 2.0 * np.pi

# Repo top level, so the script works regardless of CWD.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUTFILE = os.path.join(HERE, "fftfit_reference.npz")
PFD_FILE = os.path.join(
    REPO, "tests", "prepfold", "goodfolds",
    "Ter5_080912_short2bits_PSR_1748-2446A.pfd",
)


# --------------------------------------------------------------------------- #
# Profile shapes.  Each returns a noiseless, unit-ish profile of length nbins
# with its peak at fractional phase `phase` (0-1), wrapping in phase.
# --------------------------------------------------------------------------- #
def prof_gaussian(nbins, phase, fwhm=0.06):
    return gaussian_profile(nbins, phase, fwhm)


def prof_double(nbins, phase, fwhm=0.04):
    # Main pulse + a trailing interpulse-ish component at 0.35 phase later.
    return (gaussian_profile(nbins, phase, fwhm)
            + 0.4 * gaussian_profile(nbins, (phase + 0.35) % 1.0, fwhm * 1.6))


def prof_vonmises(nbins, phase, kappa=40.0):
    x = np.arange(nbins, dtype=np.float64) / nbins
    y = np.exp(kappa * np.cos(TWOPI * (x - phase)))
    return y / y.max()


def prof_sine(nbins, phase):
    # Near-sinusoid: essentially all power in the fundamental (-> ngood ~ 1).
    x = np.arange(nbins, dtype=np.float64) / nbins
    return 0.5 * (1.0 + np.cos(TWOPI * (x - phase)))


def prof_wide(nbins, phase):
    # Very wide, fundamental-dominated, but with a few small higher harmonics so
    # it is still fittable. Note the algorithm bails (eshift=999) for ngood < 4,
    # because nsum0 = min(16, ngood/4) collapses to 0; a 4-harmonic sum lands at
    # ngood ~ 4-7, exercising the smallest fittable / low-harmonic path.
    theta = TWOPI * (np.arange(nbins, dtype=np.float64) / nbins - phase)
    y = sum(np.cos(k * theta) / k for k in range(1, 5))
    y -= y.min()
    return y / y.max()


def prof_narrow(nbins, phase):
    # ~1.5-bin-wide spike, bin-scaled so it stays narrow at every nbins
    # (-> power across nearly all harmonics, large ngood).
    return gaussian_profile(nbins, phase, 1.5 / nbins)


SHAPES = {
    "gaussian": prof_gaussian,
    "double": prof_double,
    "vonmises": prof_vonmises,
    "sine": prof_sine,
    "wide": prof_wide,
    "narrow": prof_narrow,
}


def measure(profile, template, rotate_prof=True):
    """Exactly the FFTFIT call sequence used by get_TOAs.py / sum_profiles.py."""
    c, amp, pha = cprof(template)
    if rotate_prof:
        pha = np.fmod(pha - np.arange(1, len(pha) + 1) * pha[0], TWOPI)
    out = fftfit(profile, amp, pha)  # shift,eshift,snr,esnr,b,errb,ngood
    return amp, pha, out


def add_case(store, tag, template, profile, meta):
    """Append one case: store the arrays under indexed keys + a meta row."""
    idx = len(store["meta"])
    amp, pha, out = measure(profile, template)
    store["arrays"][f"templ_{idx}"] = np.asarray(template, dtype=np.float64)
    store["arrays"][f"prof_{idx}"] = np.asarray(profile, dtype=np.float64)
    store["arrays"][f"amp_{idx}"] = np.asarray(amp, dtype=np.float64)
    store["arrays"][f"pha_{idx}"] = np.asarray(pha, dtype=np.float64)
    shift, eshift, snr, esnr, b, errb, ngood = out
    row = dict(meta)
    row.update(idx=idx, tag=tag, nbins=len(profile),
               shift=shift, eshift=eshift, snr=snr, esnr=esnr,
               b=b, errb=errb, ngood=int(ngood))
    store["meta"].append(row)


def build_synthetic(store):
    rng = np.random.default_rng(20260701)
    nbins_list = [64, 128, 256, 512, 1024]
    shifts = [0.0, 0.13, 0.25, 0.5, 0.67, 0.9, 0.499]  # incl. near half-bin
    # amp = peak-signal scale; noise sigma is fixed at 1.0 per bin.
    # strong -> clean fit; weak -> noisy; tiny -> expected bailout (eshift=999).
    amps = {"strong": 30.0, "weak": 4.0, "tiny": 0.5}
    n_noise = 2

    for shape_name, shapefn in SHAPES.items():
        for nbins in nbins_list:
            # Reference template: noiseless, peak at phase 0.5, high fidelity.
            template = shapefn(nbins, 0.5)
            template = template / template.max()
            for true_shift in shifts:
                clean = shapefn(nbins, (0.5 + true_shift) % 1.0)
                clean = clean / clean.max()
                for amp_name, amp in amps.items():
                    for k in range(n_noise):
                        noise = rng.standard_normal(nbins)
                        profile = amp * clean + noise
                        add_case(store, "synthetic", template, profile, dict(
                            shape=shape_name, amp_name=amp_name, amp=amp,
                            true_shift=true_shift, true_shift_bins=true_shift * nbins,
                            noise_seed_idx=k))


def build_realdata(store):
    if not os.path.exists(PFD_FILE):
        sys.stderr.write(f"WARNING: {PFD_FILE} not found; skipping real-data cases.\n")
        return
    # NOTE: these are raw fold counts with large DC offsets (template mean ~1e7,
    # per-part ~1e5). Consequently the scale factor `b` and `snr` outputs are tiny
    # and scale-dependent -- that is expected, not a bug. The primary comparison
    # targets for these cases are `shift`, `eshift`, and `ngood`. The huge
    # magnitudes also stress the Fortran's single-precision (real*4) arithmetic,
    # so the double-precision port may legitimately be *more* accurate here
    # ("match, but aim to improve").
    from presto.prepfold import pfd
    p = pfd(PFD_FILE)
    p.dedisperse()
    # profs shape: (npart, nsub, proflen). Template = full summed profile (high S/N).
    template = p.profs.sum(0).sum(0)
    # Observed profiles = each time part, summed over subbands (what get_TOAs folds).
    parts = p.profs.sum(1)
    for ipart in range(parts.shape[0]):
        add_case(store, "realdata", template, parts[ipart], dict(
            pfd=os.path.basename(PFD_FILE), ipart=ipart))


def main():
    store = {"meta": [], "arrays": {}}
    build_synthetic(store)
    build_realdata(store)

    # Flatten meta (list of dicts with varying keys) into a compact object array.
    meta = np.array(store["meta"], dtype=object)
    np.savez_compressed(OUTFILE, meta=meta, **store["arrays"])

    n = len(store["meta"])
    nbail = sum(1 for r in store["meta"] if r["eshift"] >= 999.0)
    nreal = sum(1 for r in store["meta"] if r["tag"] == "realdata")
    print(f"Wrote {OUTFILE}")
    print(f"  {n} cases total ({n - nreal} synthetic, {nreal} real-data)")
    print(f"  {nbail} cases hit the FFTFIT bailout (eshift>=999)")


if __name__ == "__main__":
    main()

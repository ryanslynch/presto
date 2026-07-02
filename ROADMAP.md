# PRESTO Roadmap

Rough notes on significant changes planned for the coming weeks/months. This is a living,
aspirational document — items may change, be reordered, or be dropped. For stable "how to work
in this repo" guidance, see [CLAUDE.md](CLAUDE.md); for tracked work, see GitHub issues.

## Primary goal: get PRESTO into conda-forge

Package all of PRESTO — compiled code **and** Python modules — for conda-forge, so it can be
installed trivially with `conda`/`pixi`. Most of the items below are prerequisites or
simplifications that make clean packaging possible.

### Remove environment-variable requirements

Stop requiring `PRESTO` (and other) environment variables to be set. These currently locate
runtime data such as the pulsar database and the FFTW wisdom file, among other things. For a
conda-forge package these paths need to be discoverable without user-set env vars.

### Reduce/remove the Fortran dependency

- Remove Fortran from the main compiled binaries, replacing it with GSL code where possible.
- Convert the original **FFTFIT** Fortran code and its Python module to pure Python
  (NumPy + SciPy). This can be done independently of the rest and should get thorough tests.
  See the dedicated plan below.

#### FFTFIT pure-Python rewrite (first Fortran-removal target)

FFTFIT (Taylor 1992) measures pulsar times-of-arrival by fitting a shift/scale between an
observed profile and a template, entirely in the Fourier domain. Current implementation:
`python/fftfit_src/*.f` (`fftfit`, `cprof`, `fccf`, `brent`, `ffft`) wrapped via f2py; the
public API used by `bin/get_TOAs.py` and `bin/sum_profiles.py` is just `cprof(template)` and
`fftfit(profile, amp, pha)`. Because this feeds high-precision timing, the replacement must be
validated very carefully against the original.

**Validation philosophy:** "match, but aim to improve" — bulk agreement with the Fortran must be
negligible relative to the reported `eshift`; cases where the port is *more* accurate or stable
(low-SNR, the `eshift=999` bailout, near-half-bin shifts) count as improvements, not failures.
Note the Fortran is single-precision (`real*4`) with truncated constants, so a double-precision
port will not be bit-identical by design.

**Oracles:** (1) a frozen golden dataset generated from the current Fortran [primary]; (2) Anne
Archibald's independent pure-Python FFTFIT [independent cross-check — see below]; (3) real-data
TOAs, kept lightweight by reusing the existing prepfold fixtures in `tests/prepfold/goodfolds/*.pfd`
rather than adding new data.

**Two implementations, selectable by method (decided):** ship both behind a method selector.
  * `"classic"` — the faithful NumPy/SciPy port of Taylor's Fortran. **Default**, for
    backward-compatible TOAs / reproducibility. Validated to ~0.28 sigma vs. the Fortran.
  * `"aarchiba"` — Anne Archibald's independent algorithm (cross-spectrum + upsampled `irfft` +
    bounded minimization + least-squares scale/uncertainty), adopted from her unmerged PINT
    branch. Opt-in improvement: handles the symmetric-profile / very-low-harmonic case the
    Fortran bails on, and different-length profiles.

**Adopting Anne's code (`aarchiba/PINT` branch `fftfit_new`, BSD-3-clause → GPL-compatible,
retain attribution):** her implementation appears abandoned upstream (she has left the field; the
PR, nanograv/PINT#777, was never merged). PRESTO can give it a maintained home. Her 844-line
`hypothesis` test suite is excellent and should be adopted/adapted. Note her API returns `shift`
in *turns* plus `scale`/`offset`/`uncertainty`; a thin adapter maps to PRESTO's
`(shift_bins, eshift, snr, esnr, b, errb, ngood)` tuple.

**Open question — uncertainty calibration:** Anne's likely reason for never finalizing was
uncertainty estimation, not the shift (which is solid). Her source carries a `FIXME` about the
noise-std normalization, and her statistical coverage tests (does the 1-sigma bar contain truth
~68% of the time?) are the hard part; she also marks *PRESTO's* uncertainties `xfail(reason=
"bug?")` in places. Plan: port her test suite, run it against both implementations, see where the
uncertainties actually fail, then decide how much to invest in fixing calibration.

**Phased approach:**
0. [done] Golden-reference harness: freeze inputs+outputs of the current Fortran over a broad
   synthetic battery (`fftfit_reference.npz`), plus real-data cases from a `goodfolds` fold.
1. [done] Bottom-up "classic" port (`cprof` via NumPy `rfft`; `fftfit` with `scipy.optimize.brentq`
   for `brent.f` and NumPy for `ffft.f`, letting both Fortran files be dropped). Validated vs the
   frozen oracle: shift within 0.28 sigma, bailouts exact.
2. [done] Independent cross-check: vendored Anne's algorithm agrees with the port to <=0.23 bins on
   all well-determined fits (disagreements only at snr~1.5).
3. [in progress] Adopt Anne's code + test suite as the `"aarchiba"` method; build the method
   selector and the shift/uncertainty adapter; resolve the uncertainty-calibration question above.
4. Assemble behind the public API (drop-in for `get_TOAs.py`, `sum_profiles.py`) with `"classic"`
   default; validate against real-data TOAs from `goodfolds`.
5. Swap the build: replace the f2py target in `python/fftfit_src/meson.build` with a pure-Python
   install, remove the Fortran, update `test_fftfit.py`.

### Remove the TEMPO dependency

- Barycentering: use **ERFA** instead of TEMPO.
- Polyco creation: use **tempo2** with tempo1-style predictors.

### Adopt ERFA/SOFA for astronomy/time routines

Replace hand-rolled code with ERFA/SOFA where possible: `mjd2cal`, `cal2jd`, certain routines in
`misc_utils.c`, and especially the barycenter↔topocenter calculations. This will require:
- a database of observatory positions (from PINT or TEMPO), and
- possibly a leap-seconds list.

### Replace numerical routines with GSL

Replace the in-tree median and statistics calls with well-tested GSL equivalents.

### Drop support for retired pulsar backends

Remove all code for old backends (Spigot, BCPM, WAPP, and others). This is believed to affect
only the `readfile` code path.

## Other known issues to fix

### `prepfold_multi` generated command-line files

`prepfold_multi` currently requires **manual** edits to its auto-generated
`src/prepfold_multi_cmd.[ch]` after they are generated from `clig/prepfold_multi_cmd.cli` — the
generator (`pyclig`) output is not yet correct/complete on its own. Fix so `prepfold_multi`
regenerates cleanly from its `.cli` like every other tool. Until then, do not blindly overwrite
the hand-edits when regenerating.

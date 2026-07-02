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
- **[DONE]** Convert the original **FFTFIT** Fortran code and its Python module to pure Python
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

**Uncertainty calibration [decided: ship now, fix later].** Measured 1-sigma coverage (does the
error bar contain the truth ~68% of the time?) shows *both* methods under-cover by ~5-15%, worse
for broad profiles (e.g. kappa=1: aarchiba 0.656, classic 0.625 vs target 0.683; multi-sigma over
N=2000). This is a pre-existing FFTFIT trait inherited faithfully by the classic port, not a
regression; the shift itself is solid for both. Decision: proceed with the rewrite, document the
uncertainties as approximate/slightly optimistic, and treat calibration as a separate follow-up
(see below) rather than a blocker.

**Follow-up (tracked, non-blocking): FFTFIT uncertainty calibration.** Get 1-sigma coverage to
~0.683 across regimes. aarchiba is the better starting point (marginally better calibrated, and
its least-squares/covariance framework is amenable to a principled fix). Candidate fixes: resolve
Anne's `std`-normalization `FIXME`; investigate linearization bias for broad profiles; possibly an
empirical inflation factor validated against the coverage tests.

**Status: DONE** (except the non-blocking uncertainty-calibration follow-up above). The pure-Python
FFTFIT is the installed implementation (`python/presto/fftfit.py` + `fftfit_aarchiba.py`); the
Fortran and its f2py build are removed. `get_TOAs.py` gained `-A/--fftfit {classic,aarchiba}`
(default classic). Regression + `hypothesis` tests live in `python/fftfit_src/` and drive the
installed package; the frozen oracle (`fftfit_reference.npz`) is retained so tests survive the
Fortran's removal. End-to-end verified on two topocentric `goodfolds` folds: classic reproduces the
prior TOAs exactly; aarchiba agrees within ~0.1 sigma.

**Phased approach:**
0. [done] Golden-reference harness: freeze inputs+outputs of the current Fortran over a broad
   synthetic battery (`fftfit_reference.npz`), plus real-data cases from a `goodfolds` fold.
1. [done] Bottom-up "classic" port (`cprof` via NumPy `rfft`; `fftfit` with `scipy.optimize.brentq`
   for `brent.f` and NumPy for `ffft.f`, letting both Fortran files be dropped). Validated vs the
   frozen oracle: shift within 0.28 sigma, bailouts exact.
2. [done] Independent cross-check: vendored Anne's algorithm agrees with the port to <=0.23 bins on
   all well-determined fits (disagreements only at snr~1.5).
3. [done] Adopt Anne's code + test suite as the `"aarchiba"` method; build the method selector and
   the shift/uncertainty adapter. (Uncertainty calibration deferred as a tracked follow-up.)
4. [done] Assemble behind the public API (drop-in for `get_TOAs.py`, `sum_profiles.py`) with
   `"classic"` default and a `--fftfit` selector; validated against real-data TOAs from `goodfolds`.
5. [done] Swap the build: replace the f2py target with a pure-Python install, remove the Fortran,
   update the tests.

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

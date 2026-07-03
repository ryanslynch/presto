# PRESTO Roadmap

Rough notes on significant changes planned for the coming weeks/months. This is a living,
aspirational document — items may change, be reordered, or be dropped. For stable "how to work
in this repo" guidance, see [CLAUDE.md](CLAUDE.md); for tracked work, see GitHub issues.

## Primary goal: get PRESTO into conda-forge

Package all of PRESTO — compiled code **and** Python modules — for conda-forge, so it can be
installed trivially with `conda`/`pixi`. Most of the items below are prerequisites or
simplifications that make clean packaging possible.

**ERFA packaging wrinkle:** conda-forge has **no C-library ERFA package** (verified 2026-07:
no `erfa`/`liberfa` feedstock exists, and `pyerfa` only ships a private Python extension — no
`erfa.h` or `liberfa.so` to link against; Meson WrapDB has no erfa entry either). PRESTO
handles this generally via `subprojects/erfa.wrap`: meson uses system ERFA when present and
otherwise downloads/builds the release tarball automatically (offline: pre-place the tarball
in `subprojects/packagecache/`). Since conda-forge builds have no network access, the PRESTO
recipe should either (a) list the ERFA tarball as an extra source unpacked into
`subprojects/packagecache/` (works today, but statically vendors it), or (b) — cleaner —
submit a trivial `liberfa` feedstock to conda-forge first (autotools or meson build, ~30-line
recipe) and depend on it. Prefer (b).

### Remove environment-variable requirements

Stop requiring `PRESTO` (and other) environment variables to be set. These currently locate
runtime data such as the pulsar database and the FFTW wisdom file, among other things. For a
conda-forge package these paths need to be discoverable without user-set env vars.

### Reduce/remove the Fortran dependency

- Remove Fortran from the main compiled binaries, replacing it with GSL code where possible.
- **[DONE]** Convert the original **FFTFIT** Fortran code and its Python module to pure Python
  (NumPy + SciPy). This can be done independently of the rest and should get thorough tests.
  See the dedicated plan below.
- **[DONE]** Replace the vendored LAPACK/BLAS `dgels` least-squares solver (`src/least_squares.f`,
  the last Fortran source) with GSL's `gsl_multifit_linear` in `bary2topo()`. This drops
  `least_squares.f`, `include/f77.h`, and the Fortran language from the meson build, so
  `libpresto` and all compiled tools are now Fortran-free (PGPLOT remains an external Fortran
  dependency). Validated: GSL vs the old Fortran agree to ~4e-14 relative on the raw fit
  outputs, and end-to-end prepfold topocentric timing parameters differ by ~1e-10 to 1e-9 of
  their quoted uncertainties (QR-vs-SVD roundoff, negligible for timing).

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

- **[DONE]** Barycentering: use **ERFA** instead of TEMPO. `barycenter()` in
  `src/barycenter.c` now computes UTC→TT→TDB (ERFA's built-in leap seconds + `eraDtdb`),
  the Roemer delay (`eraEpv00` analytic ephemeris + `eraPvtob`/`eraC2i06a` for the
  observatory, from a new internal ITRF table in `src/observatories.c`), and the solar
  Shapiro delay, all in-process with **no external files**. v/c comes from a central
  difference of the total delay, matching TEMPO's `femit/fobs - 1` definition (positive
  = moving away). Two conventions matter: PRESTO MJDs count elapsed-seconds/86400 from
  UTC midnight (not ERFA's "stretched" leap-day quasi-JD — up to 1 s different on a
  leap-second day), and UT1≈UTC is used (≤1.4 μs). Validated vs TEMPO/DE405 over 14
  observatories × 7 sky positions × 8 epochs (1995–2023, incl. leap-second days):
  absolute agreement ~14 μs max (the eraEpv00-vs-DE405 ephemeris difference), ~1.3 μs
  differential drift over a dense 12-hr series (mostly TEMPO's measured UT1 vs our
  UT1≈UTC), and |Δ(v/c)| ≲ 3e-9 (vs typical |v/c| ~1e-4). The old code is retained as
  `barycenter_tempo()` and `src/check_bary_erfa.c` re-runs the full comparison anytime
  (needs TEMPO); `tests/test_barycenter.py` checks against frozen TEMPO values
  (`tests/bary_tempo_reference.txt`) with no TEMPO needed. The `ephem` argument is now
  ignored.
- **[DONE]** Polyco creation: use **tempo2** (which is on conda-forge and sets `$TEMPO2`
  automatically) instead of TEMPO. `make_polycos()` in `src/polycos.c` and
  `polycos.create_polycos()` now run `tempo2 -tempo1 -f par -polyco "mjd1 mjd2 nspan
  ncoeff maxha sitename freq"`, which writes TEMPO1-format polycos that the existing
  `getpoly()`/`phcalc()` readers consume (one reader fix: the line-2 observatory field is
  now a site name like "gbt", so it is parsed as a string). Sites are passed as tempo2
  observatory *names* — tempo2's tempo1-code alias table is conflicting ('k' is e-Merlin,
  'y' is aliased to Jodrell), and the old TEMPO codes had two latent bugs anyway (ATA 's'
  was SHAO 65m; KAT-7 'k' was FAST). Validated vs TEMPO with `src/check_polycos_t2.c`
  (needs both programs): constant phase offsets 0.001–0.023 μs, differential drift
  ≤ 3e-6 turns, |Δf|/f ≤ 4e-11; all polyco-based prepfold regression folds pass against
  the *unmodified* goodfolds references and `get_TOAs.py` TOAs are identical. The old
  generator is retained as `make_polycos_tempo1()`. **With this plus ERFA barycentering,
  PRESTO no longer needs TEMPO at all.** (tempo2 caveat: no conda-forge osx-arm64 build.)

### Adopt ERFA/SOFA for astronomy/time routines

Replace hand-rolled code with ERFA/SOFA where possible: `mjd2cal`, `cal2jd`, and certain routines
in `misc_utils.c`. The barycenter↔topocenter calculations are done (see above), which also
provided the two prerequisites listed here: the observatory-position database now lives in
`src/observatories.c` (coordinates from TEMPO's obsys.dat, with PINT/tempo2 filling gaps), and
leap seconds come from ERFA's built-in table (`eraDat` — note this means very new leap seconds,
should they ever resume, require an ERFA update).

### Replace numerical routines with GSL

Replace the in-tree median and statistics calls with well-tested GSL equivalents.

### Drop support for retired pulsar backends

Remove all code for old backends (Spigot, BCPM, WAPP, and others). This is believed to affect
only the `readfile` code path.

## Nice-to-haves

### Revitalize CI

The old Travis CI setup (`.travis/`, `.travis.yml`) was stale for years and has been removed
(2026-07). Set up modern CI — presumably GitHub Actions — that builds the meson project and
Python package and runs the test suite: `tests/test_barycenter.py`, `tests/test_presto_python.py`
(now TEMPO-free), the fftfit tests in `python/fftfit_src/`, and ideally the prepfold suite
(`tests/prepfold/prepfold_tests.sh`, which downloads its ~100 MB fixture and needs TEMPO for
the polyco-based folds). A linux + macOS matrix would also catch the recurring macOS
build/rpath issues early.

## Other known issues to fix

### `prepfold_multi` generated command-line files

`prepfold_multi` currently requires **manual** edits to its auto-generated
`src/prepfold_multi_cmd.[ch]` after they are generated from `clig/prepfold_multi_cmd.cli` — the
generator (`pyclig`) output is not yet correct/complete on its own. Fix so `prepfold_multi`
regenerates cleanly from its `.cli` like every other tool. Until then, do not blindly overwrite
the hand-edits when regenerating.

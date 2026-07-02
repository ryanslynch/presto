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

**Oracles:** (1) a frozen golden dataset generated from the current Fortran [primary]; (2) PINT's
pure-Python `fftfit` [independent cross-check]; (3) real-data TOAs, kept lightweight by reusing
the existing prepfold fixtures in `tests/prepfold/goodfolds/*.pfd` rather than adding new data.

**Phased approach:**
0. Golden-reference harness: freeze inputs+outputs of the current Fortran over a broad synthetic
   battery, plus frozen reference TOAs from a `goodfolds` fold via `get_TOAs.py`.
1. Bottom-up port with per-routine tests (`cprof` → `fccf` → `dchisqr` → error formulas →
   continuation loop). Use NumPy `rfft`/`irfft` for `ffft.f` and `scipy.optimize.brentq` for
   `brent.f` (scipy is already a dependency), letting those two files be dropped.
2. Assemble behind the identical public API (drop-in for `get_TOAs.py`, `sum_profiles.py`).
3. Validate against all three oracles with tiered tolerances.
4. Swap the build: replace the f2py target in `python/fftfit_src/meson.build` with a pure-Python
   install, remove the Fortran, update `test_fftfit.py`.

**Nice-to-have improvement (not required):** the Fortran bails out (`eshift=999`) for profiles
with fewer than 4 significant harmonics, because `nsum0 = min(16, ngood/4)` collapses to 0. In
particular a pure sinusoid (`ngood=1`) cannot be fit. An improved algorithm should ideally handle
the near-sinusoidal / very-low-harmonic case gracefully. This has not been needed in practice, so
it is a bonus goal, not a requirement — but a good one to aim for.

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

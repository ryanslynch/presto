# PRESTO Multi-Candidate Folding

## Project Goal

Enhance `prepfold` to fold multiple pulsar candidates in a single pass over a raw data file, reducing cost from `O(N × file_read)` to `O(1 × file_read + N × fold)`.

## Key Files

| File | Role |
|------|------|
| `src/prepfold.c` | ~1,856-line monolithic `main()` — target for refactoring |
| `src/fold.c` | Core `fold()` / `foldfile()` — well-isolated, do not modify |
| `src/prepfold_utils.c` | `correct_subbands_for_DM()`, `normalize_stats()`, etc. |
| `src/dispersion.c` | DM delay calculations |
| `include/prepfold.h` | `prepfoldinfo` struct — single-candidate state |

## Implementation Plan

### Phase 1 — Extract `fold_candidate()` from `prepfold.c`

- Extract lines ~800–1690 (folding + optimization loops) into:
  ```c
  int fold_candidate(
      spectra_info *s,        // shared read-only raw data state
      prepfoldinfo *search,   // per-candidate: rawfolds, stats, grids
      Cmdline *cmd,           // per-candidate: period, pdot, dm, etc.
      float *rawdata_cache,
      long long nsamples
  );
  ```
- `main()` becomes: parse args → identify data → load metadata → call `fold_candidate()` → output
- Zero behavior change; all existing use cases must continue to work
- Only `src/prepfold.c` is touched in this phase

### Phase 2 — New `src/prepfold_multi.c` executable

- Parse candidate list file: plain-text `(P0, Pdot, DM)` tuples, comment/blank-line support
- Read raw data once (or chunk-stream for files too large for RAM)
- Sort candidates by DM; reuse dedispersed time series within a configurable `dm_tolerance`
- Allocate per-candidate `prepfoldinfo`; call `fold_candidate()` for each
- Write per-candidate `.pfd` and `.inf` output files with unique filenames
- Produce aggregate summary table of best-fit parameters
- Link existing object files (`fold.o`, `dispersion.o`, `prepfold_utils.o`, etc.) — no logic duplication
- Add `prepfold_multi` target to `Makefile`

### Phase 3 — OpenMP Parallelism in `prepfold_multi.c`

- Outer loop over candidates:
  ```c
  #pragma omp parallel for schedule(dynamic) num_threads(ncpus)
  for (int ic = 0; ic < ncands; ic++) {
      fold_candidate(&s, &searches[ic], ...);
  }
  ```
- Per-candidate data (`prepfoldinfo`, buffers, stats) must be thread-private
- Shared read-only data (raw float cache, `spectra_info`) is safe
- Audit `fold()`, `correct_subbands_for_DM()`, `prepfold_utils.c` for thread-safety
- Runtime selection: Level 1 (candidate-level, >20 candidates) vs Level 2 (subband-level, <5 candidates) via `-ncpus`; optional `--nested` flag for both

### Phase 4 — Output & Integration

- Thread-safe per-candidate `.pfd`/`.inf` file writing
- `-noplots` flag for batch runs (deferred/suppressed plotting)
- Optional Python helper: generate candidate list from `accelsearch` output

## Candidate List Format

```
# P0(s)          Pdot(s/s)       DM(pc/cm^3)
0.00575212       1.56e-19        71.02
0.033592         -4.2e-18        47.8
```

Usage: `prepfold_multi -candlist candidates.txt rawfile.fits`

## Memory Budget

- Per `prepfoldinfo`: `rawfolds` ≈ 1 MB (32 × 64 × 64 × 8 bytes at defaults)
- 1,000 candidates ≈ ~1 GB in flight; batch if RAM-constrained
- Large raw files (e.g. 1-hour PSRFITS @ 256ch / 64 µs ≈ 10 GB): use existing `WORKLEN`-chunk streaming loop adapted for multi-candidate

## Key Decisions

- **Refactor first** — without `fold_candidate()`, multi-candidate requires dangerous duplication
- **New executable** `prepfold_multi` — backward-compatible, cleaner separation
- **DM-sorted deduplication** is the single biggest speedup (10–50× in dedispersion cost)
- **OpenMP Level 1** (candidate outer loop) for large candidate counts; Level 2 (subband inner loop, already exists at `prepfold.c:1360`) for small counts

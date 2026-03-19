#!/usr/bin/env python3
"""accel_to_candlist.py — Convert accelsearch ACCEL_* files to a prepfold_multi
candidate list.

Usage
-----
  accel_to_candlist.py [options] ACCEL_file [ACCEL_file ...]

The script reads one or more PRESTO accelsearch output files (named like
``basename_DM47.80_ACCEL_50``), optionally sieves them (deduplication, harmonic
removal), converts Fourier drift *z* to ``pdot``, and writes a candidate list
in the format expected by ``prepfold_multi``:

  # P0(s)          Pdot(s/s)       DM(pc/cm^3)   label
  0.00575212       1.56e-19        71.02          J0437_DM71.02_cand1

The output can then be passed directly to ``prepfold_multi``::

  prepfold_multi -candlist candidates.txt -ncpus 8 rawfile.fits

Pdot derivation
---------------
``accelsearch`` outputs the Fourier drift *z* (total drift in frequency bins
across the observation of length *T*):

    fdot  = z / T^2          [Hz/s]
    pdot  = -fdot * p^2      [s/s]

For a non-accelerated search (``ACCEL_0`` files) *z* is always 0 and pdot = 0.

Options
-------
  -o / --output FILE   Write candidate list to FILE (default: stdout)
  --sigma THRESH       Minimum candidate sigma (default: 6.0)
  --no-sift            Skip deduplication and harmonic removal; keep every
                       candidate above the sigma threshold
  --min-dms N          Minimum number of DMs in which a candidate must appear
                       to be kept (default: 1, i.e. no DM filtering)
  --low-dm VAL         Lowest DM considered real (default: 2.0 pc/cm^3)
  --max-cands N        Keep only the top N candidates sorted by sigma (0=all)
  -v / --verbose       Print summary to stderr
"""

from __future__ import print_function

import argparse
import os
import re
import sys

try:
    import presto.sifting as sifting
except ImportError:
    sys.exit("Error: cannot import presto.sifting.  "
             "Make sure the PRESTO Python package is on your PYTHONPATH.")


# ---------------------------------------------------------------------------
# Pdot calculation
# ---------------------------------------------------------------------------

def z_to_pdot(z, p, T):
    """Convert Fourier drift z to pdot (s/s).

    Parameters
    ----------
    z : float
        Fourier drift (bins drifted across observation).
    p : float
        Barycentric spin period (s).
    T : float
        Observation length (s).

    Returns
    -------
    float
        Period derivative in s/s (negative for spin-down, per convention).
    """
    if T == 0.0:
        return 0.0
    fdot = z / (T * T)        # Hz/s
    return -fdot * p * p      # s/s


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

_DM_re = re.compile(r"DM(\d+\.\d+)")

def _candidate_label(cand):
    """Build a short, unique label for a candidate suitable for filenames."""
    base = os.path.basename(cand.filename)
    # Strip trailing ACCEL_NN suffix so label stays compact
    base = re.sub(r"_ACCEL_\d+$", "", base)
    # Strip DM portion; we'll add it explicitly
    base = re.sub(r"_DM[\d.]+", "", base)
    return "{}_DM{:.2f}_c{:04d}".format(base, cand.DM, cand.candnum)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accel_files", nargs="+", metavar="ACCEL_file",
                   help="One or more accelsearch ACCEL_* output files")
    p.add_argument("-o", "--output", default=None, metavar="FILE",
                   help="Write candidate list to FILE (default: stdout)")
    p.add_argument("--sigma", type=float, default=6.0, metavar="THRESH",
                   help="Minimum candidate sigma (default: %(default)s)")
    p.add_argument("--no-sift", action="store_true",
                   help="Skip deduplication and harmonic removal")
    p.add_argument("--min-dms", type=int, default=1, metavar="N",
                   help="Min DMs a candidate must appear in (default: %(default)s)")
    p.add_argument("--low-dm", type=float, default=2.0, metavar="VAL",
                   help="Lowest DM considered real (default: %(default)s pc/cm^3)")
    p.add_argument("--max-cands", type=int, default=0, metavar="N",
                   help="Keep top N candidates by sigma (0 = keep all)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print summary to stderr")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Set sifting thresholds before reading
    sifting.sigma_threshold = args.sigma

    # ------------------------------------------------------------------
    # Read candidates from ACCEL files
    # ------------------------------------------------------------------
    # prelim_reject applies sifting.sigma_threshold and period limits;
    # skip it when --no-sift is given so we can apply our own threshold.
    candlist = sifting.read_candidates(args.accel_files,
                                       prelim_reject=(not args.no_sift))

    if args.verbose:
        print("Read {:d} candidates from {:d} file(s)".format(
            len(candlist), len(args.accel_files)), file=sys.stderr)

    if not len(candlist):
        print("Warning: no candidates found.", file=sys.stderr)
        return 0

    # ------------------------------------------------------------------
    # Sifting (optional)
    # ------------------------------------------------------------------
    if not args.no_sift:
        # Collect unique DM strings from the input filenames for DM-problem check
        dm_matches = [_DM_re.search(f) for f in args.accel_files]
        dmstrs = sorted(set(m.group(1) for m in dm_matches if m))

        if len(candlist):
            candlist = sifting.remove_duplicate_candidates(candlist)
        if len(candlist) and args.min_dms > 1 and dmstrs:
            candlist = sifting.remove_DM_problems(
                candlist, args.min_dms, dmstrs, args.low_dm)
        if len(candlist):
            candlist = sifting.remove_harmonics(candlist)

        if args.verbose:
            print("After sifting: {:d} candidates".format(len(candlist)),
                  file=sys.stderr)
    else:
        # Manual sigma filter when sifting is off
        cands_raw = [c for c in candlist.cands if c.sigma >= args.sigma]
        candlist.cands = cands_raw
        if args.verbose:
            print("After sigma >= {:.1f} filter: {:d} candidates".format(
                args.sigma, len(candlist)), file=sys.stderr)

    if not len(candlist):
        print("Warning: no candidates survived filtering.", file=sys.stderr)
        return 0

    # ------------------------------------------------------------------
    # Sort by sigma descending; apply --max-cands limit
    # ------------------------------------------------------------------
    from operator import attrgetter
    candlist.cands.sort(key=attrgetter('sigma'), reverse=True)

    if args.max_cands > 0:
        candlist.cands = candlist.cands[:args.max_cands]
        if args.verbose:
            print("Keeping top {:d} candidates".format(args.max_cands),
                  file=sys.stderr)

    # ------------------------------------------------------------------
    # Write candidate list
    # ------------------------------------------------------------------
    out = open(args.output, "w") if args.output else sys.stdout

    out.write("# Candidate list generated by accel_to_candlist.py\n")
    out.write("# Source files: {}\n".format(", ".join(args.accel_files)))
    out.write("# sigma >= {:.1f}  candidates: {:d}\n".format(
        args.sigma, len(candlist.cands)))
    out.write("#\n")
    out.write("# {:<16s}  {:<16s}  {:<14s}  {}\n".format(
        "P0(s)", "Pdot(s/s)", "DM(pc/cm^3)", "label"))
    out.write("# " + "-" * 70 + "\n")

    for cand in candlist.cands:
        pdot = z_to_pdot(cand.z, cand.p, cand.T)
        label = _candidate_label(cand)
        out.write("{:<18.10g}  {:<18.6g}  {:<14.4f}  {}\n".format(
            cand.p, pdot, cand.DM, label))

    if args.output:
        out.close()
        if args.verbose:
            print("Wrote {:d} candidates to '{}'".format(
                len(candlist.cands), args.output), file=sys.stderr)
    else:
        if args.verbose:
            print("Wrote {:d} candidates to stdout".format(
                len(candlist.cands)), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

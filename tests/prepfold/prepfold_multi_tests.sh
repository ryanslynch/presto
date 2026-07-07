#!/bin/sh

# Equivalence + read-once test for prepfold_multi.
#
# The defining correctness property of prepfold_multi is "multi == many singles":
# folding N candidates (each with its own DM) in a SINGLE pass over the raw data
# must reproduce, field for field, what N independent prepfold runs produce.  This
# script proves that for both supported inputs -- raw PSRFITS and PRESTO subbands --
# by folding the candidate files with prepfold_multi and comparing each candidate's
# .bestprof against the matching single prepfold run via compare_bestprof.py.
#
# It also measures wall-clock for the one multi run vs the sum of the single runs,
# turning the structural "the raw data is read once" argument into a measured one
# (the multi run pays the disk read once, the N singles pay it N times).
#
# Run from this directory (it reuses the data downloaded by prepfold_tests.sh and
# creates the subbands itself if they are not already present).

FAILED=0
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG=multi_output.txt
: > "$LOG"

# The binaries under test.  Without this a fresh shell finds no prepfold at
# all: every fold dies with "command not found" into $LOG and the compares
# then silently "pass" on stale output files from an earlier build.
export PATH="${_SCRIPT_DIR}/../../build/src:$PATH"
command -v prepfold_multi > /dev/null 2>&1 && command -v prepfold > /dev/null 2>&1 || {
    echo "ERROR: prepfold/prepfold_multi not found (build with: ninja -C build)"
    exit 2
}

fold() {        # run a fold command, logging output and trapping failures
    "$@" >> "$LOG" 2>&1 || {
        echo "  ERROR: '$1' exited nonzero (see $LOG)"
        FAILED=1
    }
}

# Spin parameters shared by every candidate (must match the *_cands.txt files).
P=8.66430621957513e-3
PD=-5.01154755640048e-11
COMMON="-noxwin -noscales -nooffsets -nosearch -fine -n 64"

# elapsed SECONDS.NS between two `date +%s.%N` stamps, via awk (no bc dependency).
elapsed() { awk "BEGIN{printf \"%.2f\", $2 - $1}"; }

compare() {     # $1 = multi .bestprof, $2 = single .bestprof
    if python3 "${_SCRIPT_DIR}/compare_bestprof.py" "$1" "$2" >> "$LOG" 2>&1; then
        echo "  PASS: $1"
    else
        echo "  FAIL: $1  (vs $2)"
        FAILED=1
    fi
}

# --- Test data (mirror prepfold_tests.sh) ------------------------------------
echo "Getting test data (if needed)."
curl -C - -o Ter5_080912_short2bits.fits \
    https://www.cv.nrao.edu/~sransom/Ter5_080912_short2bits.fits >> "$LOG" 2>&1
if [ ! -f Ter5_080912_DM242.30.sub00 ]; then
    echo "Making PRESTO subbands (DM 242.3)."
    prepsubband -sub -nsub 32 -subdm 242.3 -nobary -o Ter5_080912 \
        Ter5_080912_short2bits.fits >> "$LOG" 2>&1
fi
DATA=Ter5_080912_short2bits.fits
SUBS="Ter5_080912_DM242.30.sub??"

# --- RAWDATA: prepfold_multi (one pass) vs 3 single prepfold runs -------------
echo "RAWDATA equivalence (PSRFITS, 3 DMs in one pass):"
t0=$(date +%s.%N)
fold prepfold_multi $COMMON -nsub 64 -candfile "${_SCRIPT_DIR}/multi_cands.txt" \
    $DATA
t1=$(date +%s.%N)
multi_raw=$(elapsed "$t0" "$t1")

single_raw=0
for pair in 238.30:candDM238 240.30:candDM240 242.30:candDM242; do
    dm=${pair%%:*}; nm=${pair##*:}
    s0=$(date +%s.%N)
    fold prepfold $COMMON -nsub 64 -p $P -pd $PD -dm $dm -o m_single_$nm \
        $DATA
    s1=$(date +%s.%N)
    single_raw=$(awk "BEGIN{printf \"%.2f\", $single_raw + ($s1 - $s0)}")
    compare "Ter5_080912_short2bits_${nm}.pfd.bestprof" \
            "m_single_${nm}_8.66ms_Cand.pfd.bestprof"
done
echo "  read-once wall-time: multi=${multi_raw}s  sum-of-singles=${single_raw}s"
awk "BEGIN{ if ($multi_raw < $single_raw) print \"  READ-ONCE OK: multi run was faster than N separate reads\"; else print \"  READ-ONCE WARN: multi was not faster -- check raw I/O is shared\" }"

# --- insubs: prepfold_multi (one pass) vs 3 single prepfold runs --------------
echo "insubs equivalence (PRESTO subbands, 3 residual DMs in one pass):"
fold prepfold_multi $COMMON -candfile "${_SCRIPT_DIR}/multi_cands_sub.txt" \
    $SUBS
for pair in 241.30:subDM241 242.30:subDM242 243.30:subDM243; do
    dm=${pair%%:*}; nm=${pair##*:}
    fold prepfold $COMMON -p $P -pd $PD -dm $dm -o m_single_$nm $SUBS
    compare "Ter5_080912_DM242.30_${nm}.pfd.bestprof" \
            "m_single_${nm}_8.66ms_Cand.pfd.bestprof"
done

echo "Finished"
[ "${FAILED}" -eq 0 ] && echo "All prepfold_multi equivalence tests PASSED." \
    || { echo "One or more prepfold_multi equivalence tests FAILED."; exit 1; }

# To clean up temp files, do:
# rm m_single_* Ter5_080912_short2bits_candDM*.pfd* Ter5_080912_DM242.30_subDM*.pfd* multi_output.txt

"""Regression test for PRESTO's ERFA-based barycentering.

Compares ``presto.presto.barycenter`` against frozen reference values
generated with the original TEMPO-based implementation (DE405), so no
TEMPO installation is needed at test time.  The reference file was
created with::

    ./build/src/check_bary_erfa -dump > tests/bary_tempo_reference.txt

The thresholds match those in src/check_bary_erfa.c and reflect the
expected level of agreement:  the absolute time offsets are dominated
by the difference between ERFA's built-in analytical ephemeris
(eraEpv00) and JPL's DE405, plus TEMPO's observatory clock corrections;
the *differential* agreement (what matters when folding or searching)
is far tighter and is exercised by the paired times in the battery.
"""

import os

import numpy as np
import pytest

from presto import presto

SECPERDAY = 86400.0
REFFILE = os.path.join(os.path.dirname(__file__), "bary_tempo_reference.txt")

# Thresholds (see src/check_bary_erfa.c and ROADMAP.md)
MAX_ABS_DIFF_S = 50e-6
MAX_DIFFERENTIAL_S = 1e-6
MAX_VOVERC_DIFF = 5e-8


def load_reference() -> list[dict]:
    """Parse the frozen TEMPO reference battery.

    Returns
    -------
    list of dict
        One entry per (obs, ra, dec) battery case, each holding the
        arrays of topocentric UTC MJDs, barycentric TDB MJDs, and v/c.
    """
    cases: dict[tuple, dict] = {}
    with open(REFFILE) as reffile:
        for line in reffile:
            if line.startswith("#") or not line.strip():
                continue
            obs = line[:2]  # the code may contain a significant space
            ra, dec, topo, bary, voverc = line[2:].split()
            key = (obs, ra, dec)
            case = cases.setdefault(
                key,
                {
                    "obs": obs,
                    "ra": ra,
                    "dec": dec,
                    "topo": [],
                    "bary": [],
                    "voverc": [],
                },
            )
            case["topo"].append(float(topo))
            case["bary"].append(float(bary))
            case["voverc"].append(float(voverc))
    return list(cases.values())


@pytest.mark.parametrize(
    "case", load_reference(), ids=lambda c: f"{c['obs'].strip()}_{c['ra']}_{c['dec']}"
)
def test_barycenter_vs_tempo_reference(case: dict) -> None:
    """ERFA barycentering must agree with the frozen TEMPO results."""
    topo = np.asarray(case["topo"])
    reft = np.asarray(case["bary"])
    refv = np.asarray(case["voverc"])
    bary = np.zeros_like(topo)
    voverc = np.zeros_like(topo)
    presto.barycenter(topo, bary, voverc, case["ra"], case["dec"], case["obs"], "DE405")
    dt = (bary - reft) * SECPERDAY
    assert np.max(np.abs(dt)) < MAX_ABS_DIFF_S
    assert np.max(np.abs(voverc - refv)) < MAX_VOVERC_DIFF
    # The battery times come in pairs 20 min apart:  the differential
    # error across each pair must be far below the absolute offset.
    assert np.max(np.abs(dt[1::2] - dt[0::2])) < MAX_DIFFERENTIAL_S


def test_barycenter_roundtrip_sanity() -> None:
    """Basic sanity:  Roemer delay is within +-8.3 min and v/c is small."""
    topo = np.array([60300.25, 60300.35])
    bary = np.zeros_like(topo)
    voverc = np.zeros_like(topo)
    presto.barycenter(topo, bary, voverc, "17:48:04.85", "-24:46:45.0", "GB", "DE405")
    assert np.all(np.abs(bary - topo) * SECPERDAY < 600.0)
    assert np.all(np.abs(voverc) < 1.2e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

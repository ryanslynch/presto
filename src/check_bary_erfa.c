#include <presto.h>

/* Compare the ERFA-based barycenter() against the original TEMPO-based  */
/* barycenter_tempo() over a battery of observatories, sky positions,    */
/* and epochs.  Requires the 'tempo' executable (with $TEMPO set); the   */
/* comparison is skipped (exit 77) if TEMPO is not available.            */
/*                                                                       */
/* Reported/checked quantities:                                          */
/*   - absolute time difference |dt|:  dominated by the eraEpv00 vs JPL  */
/*     ephemeris difference and TEMPO's clock handling (expect ~us)      */
/*   - differential |dt| drift across a 12 hr observation:  this is     */
/*     what actually matters for folding/searching (expect << 1 us)      */
/*   - |d(v/c)|:  affects Doppler corrections (expect ~1e-9)             */

/* Observatories to test: present in both observatories.c and the        */
/* standard TEMPO obsys.dat.                                             */
static char *testobs[] = { "GB", "AO", "PK", "JB", "EF", "NC", "VL",
    "FA", "MK", "GM", "CH", "MW", "SR", "LF"
};

/* Sky positions: Ter5, high/low dec, equatorial, near the ecliptic      */
/* (i.e. close to the Sun at some epochs, to exercise the Shapiro        */
/* delay and maximal v/c), and near the ecliptic pole.                   */
static char *testras[] = { "17:48:04.85", "23:23:26.0", "05:34:31.97",
    "12:00:00.0", "00:00:00.0", "18:00:00.0", "06:45:08.9"
};
static char *testdecs[] = { "-24:46:45.0", "58:48:42.0", "22:00:52.06",
    "00:00:00.0", "00:00:00.0", "66:33:39.0", "-16:42:58.0"
};

/* Epochs (UTC MJDs), including dates close to (but not straddling)      */
/* recent leap seconds (2015-06-30 = MJD 57203, 2016-12-31 = 57753).     */
static double testmjds[] = { 50000.3, 52987.1, 55555.8, 57203.5, 57204.5,
    57753.9, 58849.2, 60300.6
};

static void compare(long N, double *topo, char *ra, char *dec, char *obs,
                    double *maxabs, double *maxvel, double *maxdiff)
/* Run both implementations on the times in topo and update the running  */
/* maxima:  absolute time diff (s), v/c diff, and the spread of the time */
/* diffs across the N times (s).                                         */
{
    double *berfa, *verfa, *btempo, *vtempo;
    double dt, mindt = 1e100, maxdt = -1e100;
    long ii;

    berfa = gen_dvect(N);
    verfa = gen_dvect(N);
    btempo = gen_dvect(N);
    vtempo = gen_dvect(N);
    barycenter(topo, berfa, verfa, N, ra, dec, obs, "DE405");
    barycenter_tempo(topo, btempo, vtempo, N, ra, dec, obs, "DE405");
    for (ii = 0; ii < N; ii++) {
        dt = (berfa[ii] - btempo[ii]) * SECPERDAY;
        if (fabs(dt) > *maxabs)
            *maxabs = fabs(dt);
        if (dt < mindt)
            mindt = dt;
        if (dt > maxdt)
            maxdt = dt;
        if (fabs(verfa[ii] - vtempo[ii]) > *maxvel)
            *maxvel = fabs(verfa[ii] - vtempo[ii]);
    }
    if (maxdt - mindt > *maxdiff)
        *maxdiff = maxdt - mindt;
    vect_free(berfa);
    vect_free(verfa);
    vect_free(btempo);
    vect_free(vtempo);
}

static void dump_reference(void)
/* Print full-precision TEMPO barycentering results for a compact       */
/* battery, for freezing into tests/ so that the ERFA implementation    */
/* can be regression-tested without TEMPO (see tests/test_barycenter.py). */
{
    char *obss[] = { "GB", "AO", "PK", "FA", "MK", "0 " };
    int pos[] = { 0, 2, 5 }, mjds[] = { 1, 3, 6, 7 };
    double topo[2], bary[2], vel[2];
    int oo, pp, mm, ii;

    printf("# obs ra dec topomjd_utc barymjd_tdb voverc   (TEMPO DE405 reference)\n");
    for (oo = 0; oo < 6; oo++) {
        for (pp = 0; pp < 3; pp++) {
            for (mm = 0; mm < 4; mm++) {
                topo[0] = testmjds[mjds[mm]];
                topo[1] = topo[0] + 1200.0 / SECPERDAY;
                barycenter_tempo(topo, bary, vel, 2, testras[pos[pp]],
                                 testdecs[pos[pp]], obss[oo], "DE405");
                for (ii = 0; ii < 2; ii++)
                    printf("%s %s %s %.12f %.12f %.14e\n",
                           obss[oo], testras[pos[pp]], testdecs[pos[pp]],
                           topo[ii], bary[ii], vel[ii]);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    double topo[721], xyz[3];
    double maxabs = 0.0, maxvel = 0.0, maxdiff = 0.0;
    double densabs = 0.0, densvel = 0.0, densdiff = 0.0;
    int numobs = sizeof(testobs) / sizeof(testobs[0]);
    int numpos = sizeof(testras) / sizeof(testras[0]);
    int nummjd = sizeof(testmjds) / sizeof(testmjds[0]);
    int oo, pp, mm, retval = 0;
    long ii;

    if (system("which tempo > /dev/null 2>&1") != 0 || getenv("TEMPO") == NULL) {
        printf("TEMPO is not available:  skipping the ERFA vs TEMPO "
               "barycentering comparison.\n");
        return 77;
    }

    if (argc > 1 && strcmp(argv[1], "-dump") == 0) {
        dump_reference();
        return 0;
    }

    printf("Comparing ERFA and TEMPO barycentering over %d observatories,\n"
           "%d sky positions, and %d epochs...\n\n", numobs, numpos, nummjd);

    /* Sparse battery: pairs of times 20 min apart at each epoch */
    for (oo = 0; oo < numobs; oo++) {
        double obsmaxabs = 0.0, obsmaxvel = 0.0, obsmaxdiff = 0.0;
        if (!obs_coords(testobs[oo], xyz, NULL)) {
            printf("Unknown observatory '%s'!\n", testobs[oo]);
            return 1;
        }
        for (pp = 0; pp < numpos; pp++) {
            for (mm = 0; mm < nummjd; mm++) {
                topo[0] = testmjds[mm];
                topo[1] = testmjds[mm] + 1200.0 / SECPERDAY;
                compare(2, topo, testras[pp], testdecs[pp], testobs[oo],
                        &obsmaxabs, &obsmaxvel, &obsmaxdiff);
            }
        }
        printf("  %s:  max |dt| = %7.3f us   max |d(v/c)| = %10.3e\n",
               testobs[oo], obsmaxabs * 1e6, obsmaxvel);
        if (obsmaxabs > maxabs)
            maxabs = obsmaxabs;
        if (obsmaxvel > maxvel)
            maxvel = obsmaxvel;
        if (obsmaxdiff > maxdiff)
            maxdiff = obsmaxdiff;
    }

    /* Dense battery: 12 hr of 1-min spaced points (the prepdata/prepfold */
    /* usage pattern), at two epochs, for GBT and Parkes on Ter5.         */
    for (mm = 0; mm < 2; mm++) {
        double mjd0 = (mm == 0) ? 55555.75 : 60300.25;
        for (ii = 0; ii < 721; ii++)
            topo[ii] = mjd0 + ii * 60.0 / SECPERDAY;
        compare(721, topo, testras[0], testdecs[0], (mm == 0) ? "GB" : "PK",
                &densabs, &densvel, &densdiff);
    }
    printf("\nDense 12-hr series (721 points):\n"
           "  max |dt| = %.3f us   differential drift = %.4f us   "
           "max |d(v/c)| = %.3e\n", densabs * 1e6, densdiff * 1e6, densvel);

    /* Thresholds:  see ROADMAP.md for the measured values these are */
    /* based on.                                                     */
    printf("\nOverall:  max |dt| = %.3f us   max |d(v/c)| = %.3e\n",
           maxabs > densabs ? maxabs * 1e6 : densabs * 1e6,
           maxvel > densvel ? maxvel : densvel);
    if (maxabs > 50e-6 || densabs > 50e-6) {
        printf("FAILURE:  absolute time differences exceed 50 us!\n");
        retval = 1;
    }
    if (densdiff > 2e-6) {
        /* Mostly TEMPO's use of measured UT1 vs our UT1 ~ UTC */
        printf("FAILURE:  differential time drift exceeds 2 us / 12 hr!\n");
        retval = 1;
    }
    if (maxvel > 5e-8 || densvel > 5e-8) {
        printf("FAILURE:  v/c differences exceed 5e-8!\n");
        retval = 1;
    }
    if (retval == 0)
        printf("\nAll comparisons passed.\n");
    return retval;
}

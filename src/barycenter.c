#include <presto.h>
#include <erfa.h>
#include <erfam.h>

double doppler(double freq_observed, double voverc)
/* This routine returns the frequency emitted by a pulsar */
/* (in MHz) given that we observe the pulsar at frequency */
/* freq_observed (MHz) while moving with radial velocity  */
/* (in units of v/c) of voverc wrt the pulsar.            */
{
    return freq_observed * (1.0 + voverc);
}


static double bary_delay(double mjd_tt, double mjd_ut1, double nhat[3],
                         double rc2i[3][3], double sp, int geocentric,
                         double elong, double phi, double hm,
                         double u_km, double v_km)
/* Return the total delay (in seconds) between a topocentric TT MJD  */
/* and the corresponding infinite-frequency barycentric TDB MJD:     */
/* Einstein (TDB-TT) + Roemer + solar Shapiro.  The observatory is   */
/* described by its WGS84 geodetic coordinates (elong/phi/hm), its   */
/* distance from the Earth's spin axis (u_km) and from the equator   */
/* plane (v_km), or by geocentric=1 for the geocenter.  rc2i is the  */
/* GCRS-to-CIRS (precession-nutation) matrix and sp the TIO locator, */
/* both slowly-varying and so evaluated once per epoch by the caller. */
{
    double pvh[2][3], pvb[2][3], pvc[2][3], pvg[2][3];
    double robs_au[3], rsun_au[3], rsun;
    double era, dtdb, roemer, shapiro;
    int ii;

    /* Heliocentric and barycentric Earth state (AU, AU/day).  Using TT */
    /* as the TDB time argument here is insignificant (< 1 ns).         */
    eraEpv00(ERFA_DJM0, mjd_tt, pvh, pvb);

    /* Geocentric position of the observatory in the GCRS (m) */
    if (geocentric) {
        for (ii = 0; ii < 3; ii++)
            pvg[0][ii] = 0.0;
    } else {
        era = eraEra00(ERFA_DJM0, mjd_ut1);
        eraPvtob(elong, phi, hm, 0.0, 0.0, sp, era, pvc);       /* CIRS */
        eraTrxpv(rc2i, pvc, pvg);                               /* -> GCRS */
    }

    /* Barycentric and heliocentric observatory positions (AU) */
    for (ii = 0; ii < 3; ii++) {
        robs_au[ii] = pvb[0][ii] + pvg[0][ii] / ERFA_DAU;
        rsun_au[ii] = pvh[0][ii] + pvg[0][ii] / ERFA_DAU;
    }

    /* Einstein delay:  TDB - TT (s), including topocentric terms */
    dtdb = eraDtdb(ERFA_DJM0, mjd_tt, fmod(mjd_ut1, 1.0), elong, u_km, v_km);

    /* Roemer delay (s):  nhat . robs / c */
    roemer = eraPdp(nhat, robs_au) * ERFA_AULT;

    /* Solar Shapiro delay (s):  2 GMsun/c^3 = ERFA_SRS * ERFA_AULT.    */
    /* The signal is retarded by -2GM/c^3 * ln(rsun + rsun.nhat) (+ a   */
    /* constant absorbed into the pulsar's epoch), so correcting to the */
    /* barycenter *adds* the logarithm.                                 */
    rsun = eraPm(rsun_au);
    shapiro = ERFA_SRS * ERFA_AULT * log(rsun + eraPdp(nhat, rsun_au));

    return dtdb + roemer + shapiro;
}


void barycenter(double *topotimes, double *barytimes,
                double *voverc, long N, char *ra, char *dec, char *obs, char *ephem)
/* This routine uses the ERFA library to correct a vector of           */
/* topocentric times (in *topotimes) to barycentric times              */
/* (in *barytimes) assuming an infinite observation                    */
/* frequency.  The routine also returns values for the                 */
/* radial velocity of the observation site (in units of                */
/* v/c) at the barycentric times.  All three vectors must              */
/* be initialized prior to calling.  The vector length for             */
/* all the vectors is 'N' points.  The topocentric times               */
/* must be UTC MJDs; the barycentric times returned are TDB            */
/* MJDs (as TEMPO has always returned).  The RA and DEC                */
/* (J2000) of the observed object are passed as strings in             */
/* the following format: "hh:mm:ss.ssss" for RA and                    */
/* "dd:mm:ss.ssss" for DEC.  The observatory site is passed            */
/* as a 2 letter ITOA code (see observatories.c).  The                 */
/* ephemeris argument is accepted for backwards compatibility          */
/* but is ignored:  ERFA uses its built-in analytical                  */
/* ephemeris (eraEpv00), which agrees with the JPL DE                  */
/* ephemerides at the few-km (i.e. ~10 microsec) level.                */
{
    long ii;
    int h_or_d, m, geocentric, status;
    double s, rar, decr, nhat[3], obs_xyz[3];
    double elong = 0.0, phi = 0.0, hm = 0.0, u_km = 0.0, v_km = 0.0;
    char obsname[32], radec[40];
    static int firsttime = 1;
    const double dt = 60.0;     /* baseline (s) for the velocity derivative */

    /* Unit vector pointing towards the source.  Parse copies since */
    /* ra_dec_from_string() modifies the string in place.           */
    snprintf(radec, 40, "%s", ra);
    ra_dec_from_string(radec, &h_or_d, &m, &s);
    rar = hms2rad(h_or_d, m, s);
    snprintf(radec, 40, "%s", dec);
    ra_dec_from_string(radec, &h_or_d, &m, &s);
    decr = dms2rad(h_or_d, m, s);
    eraS2c(rar, decr, nhat);

    /* Observatory position */
    if (!obs_coords(obs, obs_xyz, obsname)) {
        fprintf(stderr,
                "\nError:  unrecognized observatory code '%s' in barycenter().\n"
                "        Please add your observatory to src/observatories.c.\n",
                obs);
        exit(1);
    }
    geocentric = (obs_xyz[0] == 0.0 && obs_xyz[1] == 0.0 && obs_xyz[2] == 0.0);
    if (!geocentric) {
        eraGc2gd(ERFA_WGS84, obs_xyz, &elong, &phi, &hm);
        u_km = sqrt(obs_xyz[0] * obs_xyz[0] + obs_xyz[1] * obs_xyz[1]) / 1000.0;
        v_km = obs_xyz[2] / 1000.0;
    }

    if (firsttime && ephem != NULL && ephem[0] != '\0') {
        printf("Barycentering in-process using ERFA "
               "(the '%s' ephemeris request is ignored).\n", ephem);
        firsttime = 0;
    }

    for (ii = 0; ii < N; ii++) {
        double fd, dat, mjd_tt, mjd_ut1, d0, dp, dm;
        double ddt = dt / SECPERDAY;
        int iy, im, id;

        /* UTC -> TT.  PRESTO's topocentric MJDs (from backend headers)  */
        /* count elapsed seconds since UTC midnight divided by 86400, so */
        /* apply the day's TAI-UTC directly rather than using ERFA's     */
        /* "stretched" quasi-JD convention (they only differ during a    */
        /* day that ends with a leap second, but then by up to 1 s).     */
        /* UT1 ~ UTC is fine for the Earth rotation angle (|dUT1| <      */
        /* 0.9 s gives < 1.4 us of Roemer delay).                        */
        status = eraJd2cal(ERFA_DJM0, topotimes[ii], &iy, &im, &id, &fd);
        if (status) {
            fprintf(stderr,
                    "\nError:  MJD %.12f is not a valid UTC in barycenter().\n",
                    topotimes[ii]);
            exit(1);
        }
        status = eraDat(iy, im, id, fd, &dat);
        if (status < 0) {
            fprintf(stderr,
                    "\nError:  cannot get TAI-UTC for MJD %.12f in barycenter().\n",
                    topotimes[ii]);
            exit(1);
        }
        mjd_tt = topotimes[ii] + (dat + 32.184) / SECPERDAY;
        mjd_ut1 = topotimes[ii];

        /* The precession-nutation matrix and TIO locator change very   */
        /* slowly:  evaluate them once per epoch and share them between */
        /* the three delay evaluations below.                           */
        double rc2i[3][3], sp;
        eraC2i06a(ERFA_DJM0, mjd_tt, rc2i);
        sp = eraSp00(ERFA_DJM0, mjd_tt);

        d0 = bary_delay(mjd_tt, mjd_ut1, nhat, rc2i, sp, geocentric,
                        elong, phi, hm, u_km, v_km);
        barytimes[ii] = mjd_tt + d0 / SECPERDAY;

        /* v/c comes from the rate of change of the total delay:         */
        /* femit/fobs = dt_topo/dt_bary, so voverc = femit/fobs - 1      */
        /* ~ -d(delay)/dt (TEMPO's convention: positive v/c means the    */
        /* observatory moves *away* from the source).  This includes the */
        /* Einstein-rate terms, just as TEMPO's did.  The UTC->TT offset */
        /* is held fixed across the central difference.                  */
        dp = bary_delay(mjd_tt + ddt, mjd_ut1 + ddt, nhat, rc2i, sp,
                        geocentric, elong, phi, hm, u_km, v_km);
        dm = bary_delay(mjd_tt - ddt, mjd_ut1 - ddt, nhat, rc2i, sp,
                        geocentric, elong, phi, hm, u_km, v_km);
        voverc[ii] = (dm - dp) / (2.0 * dt);
    }
}


/*                                                                       */
/* The original TEMPO-based implementation follows.  It is retained so   */
/* that the ERFA implementation above can always be validated against    */
/* it (see check_bary_erfa.c) -- it is not used by any PRESTO programs   */
/* and requires the external 'tempo' executable when called.             */
/*                                                                       */

int read_resid_rec(FILE * file, double *toa, double *obsf)
/* This routine reads a single record (i.e. 1 TOA) from */
/* the file resid2.tmp which is written by TEMPO.       */
/* It returns 1 if successful, 0 if unsuccessful.       */
{
    static int firsttime = 1, use_ints = 0;
    static double d[9];

    // The default Fortran binary block marker has changed
    // several times in recent versions of g77 and gfortran.
    // g77 used 4 bytes, gfortran 4.0 and 4.1 used 8 bytes
    // and gfortrans 4.2 and higher use 4 bytes again.
    // So here we try to auto-detect what is going on.
    // The current version should be OK on 32- and 64-bit systems

    if (firsttime) {
        int ii;
        long long ll;
        double dd;

        chkfread(&ll, sizeof(long long), 1, file);
        chkfread(&dd, sizeof(double), 1, file);
        if (0)
            printf("(long long) index = %lld  (MJD = %17.10f)\n", ll, dd);
        if (ll != 72 || dd < 40000.0 || dd > 70000.0) { // 9 * doubles
            rewind(file);
            chkfread(&ii, sizeof(int), 1, file);
            chkfread(&dd, sizeof(double), 1, file);
            if (0)
                printf("(int) index = %d    (MJD = %17.10f)\n", ii, dd);
            if (ii == 72 && (dd > 40000.0 && dd < 70000.0)) {
                use_ints = 1;
            } else {
                fprintf(stderr,
                        "\nError:  Can't read the TEMPO residuals correctly!\n");
                exit(1);
            }
        }
        rewind(file);
        firsttime = 0;
    }
    if (use_ints) {
        int ii;
        chkfread(&ii, sizeof(int), 1, file);
    } else {
        long long ll;
        chkfread(&ll, sizeof(long long), 1, file);
    }
    //  Now read the rest of the binary record
    chkfread(&d, sizeof(double), 9, file);
    if (0) {                    // For debugging
        printf("Barycentric TOA = %17.10f\n", d[0]);
        printf("Postfit residual (pulse phase) = %g\n", d[1]);
        printf("Postfit residual (seconds) = %g\n", d[2]);
        printf("Orbital phase = %g\n", d[3]);
        printf("Barycentric Observing freq = %g\n", d[4]);
        printf("Weight of point in the fit = %g\n", d[5]);
        printf("Timing uncertainty = %g\n", d[6]);
        printf("Prefit residual (seconds) = %g\n", d[7]);
        printf("??? = %g\n\n", d[8]);
    }
    *toa = d[0];
    *obsf = d[4];
    if (use_ints) {
        int ii;
        return chkfread(&ii, sizeof(int), 1, file);
    } else {
        long long ll;
        return chkfread(&ll, sizeof(long long), 1, file);
    }
}

void barycenter_tempo(double *topotimes, double *barytimes,
                      double *voverc, long N, char *ra, char *dec,
                      char *obs, char *ephem)
/* This routine uses TEMPO to correct a vector of           */
/* topocentric times (in *topotimes) to barycentric times   */
/* (in *barytimes) assuming an infinite observation         */
/* frequency.  The routine also returns values for the      */
/* radial velocity of the observation site (in units of     */
/* v/c) at the barycentric times.  All three vectors must   */
/* be initialized prior to calling.  The vector length for  */
/* all the vectors is 'N' points.  The RA and DEC (J2000)   */
/* of the observed object are passed as strings in the      */
/* following format: "hh:mm:ss.ssss" for RA and             */
/* "dd:mm:ss.ssss" for DEC.  The observatory site is passed */
/* as a 2 letter ITOA code.  This observatory code must be  */
/* found in obsys.dat (in the TEMPO paths).  The ephemeris  */
/* is the full name of an ephemeris supported by TEMPO,     */
/* examples include DE200, DE421, or DE436.                 */
{
    FILE *outfile;
    long i;
    double fobs = 1000.0, femit, dtmp;
    char command[100], temporaryfile[100];

    /* Make/chdir to a temp dir to avoid multiple prepfolds stepping on
     * each other.
     */
    char tmpdir[]  = "/tmp/prestoXXXXXX";
    if (mkdtemp(tmpdir)==NULL) {
        fprintf(stderr, "barycenter_tempo: error creating temp dir.\n");
        exit(1);
    }
    char *origdir = getcwd(NULL,0);
    chdir(tmpdir);

    /* Write the free format TEMPO file to begin barycentering */

    strcpy(temporaryfile, "bary.tmp");
    outfile = chkfopen(temporaryfile, "w");
    fprintf(outfile, "C  Header Section\n"
            "  HEAD                    \n"
            "  PSR                 bary\n"
            "  NPRNT                  2\n"
            "  P0                   1.0 1\n"
            "  P1                   0.0\n"
            "  CLK            UTC(NIST)\n"
            "  PEPOCH           %19.13f\n"
            "  COORD              J2000\n"
            "  RA                    %s\n"
            "  DEC                   %s\n"
            "  DM                   0.0\n"
            "  EPHEM                 %s\n"
            "C  TOA Section (uses ITAO Format)\n"
            "C  First 8 columns must have + or -!\n"
            "  TOA\n", topotimes[0], ra, dec, ephem);

    /* Write the TOAs for infinite frequencies */

    for (i = 0; i < N; i++) {
        fprintf(outfile, "topocen+ %19.13f  0.00     0.0000  0.000000  %s\n",
                topotimes[i], obs);
    }
    fprintf(outfile, "topocen+ %19.13f  0.00     0.0000  0.000000  %s\n",
            topotimes[N - 1] + 10.0 / SECPERDAY, obs);
    fprintf(outfile, "topocen+ %19.13f  0.00     0.0000  0.000000  %s\n",
            topotimes[N - 1] + 20.0 / SECPERDAY, obs);
    fclose(outfile);

    /* Call TEMPO */

    /* Check the TEMPO *.tmp and *.lis files for errors when done. */

    sprintf(command, "tempo bary.tmp > tempoout_times.tmp");
    if (system(command) == -1) {
        fprintf(stderr, "\nError calling TEMPO in barycenter_tempo()!\n");
        exit(1);
    }

    /* Now read the TEMPO results */

    strcpy(temporaryfile, "resid2.tmp");
    outfile = chkfopen(temporaryfile, "rb");

    /* Read the barycentric TOAs for infinite frequencies */

    for (i = 0; i < N; i++) {
        read_resid_rec(outfile, &barytimes[i], &dtmp);
    }
    fclose(outfile);

    /* Write the free format TEMPO file to begin barycentering */

    strcpy(temporaryfile, "bary.tmp");
    outfile = chkfopen(temporaryfile, "w");
    fprintf(outfile, "C  Header Section\n"
            "  HEAD                    \n"
            "  PSR                 bary\n"
            "  NPRNT                  2\n"
            "  P0                   1.0 1\n"
            "  P1                   0.0\n"
            "  CLK            UTC(NIST)\n"
            "  PEPOCH           %19.13f\n"
            "  COORD              J2000\n"
            "  RA                    %s\n"
            "  DEC                   %s\n"
            "  DM                   0.0\n"
            "  EPHEM                 %s\n"
            "C  TOA Section (uses ITAO Format)\n"
            "C  First 8 columns must have + or -!\n"
            "  TOA\n", topotimes[0], ra, dec, ephem);

    /* Write the TOAs for finite frequencies */

    for (i = 0; i < N; i++) {
        fprintf(outfile, "topocen+ %19.13f  0.00  %9.4f  0.000000  %s\n",
                topotimes[i], fobs, obs);
    }
    fprintf(outfile, "topocen+ %19.13f  0.00  %9.4f  0.000000  %s\n",
            topotimes[N - 1] + 10.0 / SECPERDAY, fobs, obs);
    fprintf(outfile, "topocen+ %19.13f  0.00  %9.4f  0.000000  %s\n",
            topotimes[N - 1] + 20.0 / SECPERDAY, fobs, obs);
    fclose(outfile);

    /* Call TEMPO */

    /* Insure you check the file tempoout.tmp for  */
    /* errors from TEMPO when complete.            */

    sprintf(command, "tempo bary.tmp > tempoout_vels.tmp");
    if (system(command) == -1) {
        fprintf(stderr, "\nError calling TEMPO in barycenter_tempo()!\n");
        exit(1);
    }

    /* Now read the TEMPO results */

    strcpy(temporaryfile, "resid2.tmp");
    outfile = chkfopen(temporaryfile, "rb");

    /* Determine the radial velocities using the emitted freq */

    for (i = 0; i < N; i++) {
        read_resid_rec(outfile, &dtmp, &femit);
        voverc[i] = femit / fobs - 1.0;
    }
    fclose(outfile);

    /* Cleanup the temp files */

    remove("tempo.lis");
    remove("tempoout_times.tmp");
    remove("tempoout_vels.tmp");
    remove("resid2.tmp");
    remove("bary.tmp");
    remove("matrix.tmp");
    remove("bary.par");

    chdir(origdir);
    free(origdir);
    rmdir(tmpdir);
}

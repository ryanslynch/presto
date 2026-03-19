/*
 * prepfold_multi.c  --  fold multiple pulsar candidates in one pass
 *
 * Usage:
 *   prepfold_multi -candlist <file> [prepfold_options] rawfiles
 *
 * The candidate list file has one candidate per line:
 *   P0(s)   Pdot(s/s)   DM(pc/cm^3)   [label]
 * Lines starting with '#' or blank lines are ignored.
 *
 * All standard prepfold options (nsub, npart, proflen, maskfile, etc.)
 * are accepted and applied to every candidate.  The per-candidate
 * parameters (p, pd, dm) come from the candidate list and override
 * anything given on the command line.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "prepfold.h"
#include "prepfold_cmd.h"
#include "backend_common.h"
#include "mask.h"

/* RAWDATA: same macro as in prepfold.c */
#define RAWDATA (cmd->filterbankP || cmd->psrfitsP)

/* ------------------------------------------------------------------ */
/* Candidate list                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    double p0;
    double pdot;
    double dm;
    char   label[128];
} cand_params;

static int cmp_by_dm(const void *a, const void *b)
{
    double da = ((const cand_params *) a)->dm;
    double db = ((const cand_params *) b)->dm;
    return (da > db) - (da < db);
}

static cand_params *parse_candlist(const char *filename, int *ncands_out)
{
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot open candidate list '%s'\n", filename);
        exit(1);
    }
    int capacity = 64, ncands = 0;
    cand_params *cands = (cand_params *) malloc(capacity * sizeof(cand_params));
    char line[512];
    int lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0')
            continue;
        cand_params c;
        c.pdot = 0.0;
        c.label[0] = '\0';
        int n = sscanf(p, "%lf %lf %lf %127s", &c.p0, &c.pdot, &c.dm, c.label);
        if (n < 3) {
            fprintf(stderr,
                    "Warning: skipping malformed line %d in '%s'\n",
                    lineno, filename);
            continue;
        }
        if (ncands == capacity) {
            capacity *= 2;
            cands = (cand_params *) realloc(cands,
                                            capacity * sizeof(cand_params));
        }
        cands[ncands++] = c;
    }
    fclose(f);
    printf("Read %d candidates from '%s'\n\n", ncands, filename);
    *ncands_out = ncands;
    return cands;
}

/* ------------------------------------------------------------------ */
/* File re-opening                                                      */
/*                                                                      */
/* fold_candidate() calls close_rawfiles(s) before returning.  We must */
/* reopen the raw data files before the next candidate.                 */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    int ic, ii;
    struct spectra_info s;
    infodata idata;
    mask obsmask;
    Cmdline *cmd;
    plotflags pflags;

    /* prepfold_multi-specific options */
    const char *candlist_file = NULL;
    int noplots = 0;
    int nested_flag = 0;    /* --nested: enable Level 1 + Level 2 simultaneously */
    double dm_tol = 0.1;    /* reserved for future DM-dedup optimisation */

    /* Shared timing/geometry state */
    double recdt = 0.0, T = 0.0, N = 0.0, startTday = 0.0;
    double tepoch_shared = 0.0, bepoch_raw = 0.0, dtmp = 0.0;
    int numchan = 1, ptsperrec = 1;
    int insubs = 0, useshorts = 0;
    int nummasked = 0, good_padvals = 0;
    int *maskchans = NULL;
    long long lorec = 0, hirec = 0, numrec = 0;
    long reads_per_part = 0, worklen = 0;
    char rastring[50], decstring[50], obs[3], ephem[6];
    char telescope_shared[40];
    int do_barycenter = 1;

    /* ----------------------------------------------------------------
     * 1.  Strip prepfold_multi-specific args; forward the rest to
     *     parseCmdline so all standard prepfold options work.
     * ---------------------------------------------------------------- */

    char **fargv = (char **) malloc((argc + 1) * sizeof(char *));
    int fargc = 0;
    for (ii = 0; ii < argc; ii++) {
        if (strcmp(argv[ii], "-candlist") == 0 && ii + 1 < argc) {
            candlist_file = argv[++ii];
        } else if (strcmp(argv[ii], "-noplots") == 0) {
            noplots = 1;
        } else if (strcmp(argv[ii], "--nested") == 0 || strcmp(argv[ii], "-nested") == 0) {
            nested_flag = 1;
        } else if (strcmp(argv[ii], "-dm_tol") == 0 && ii + 1 < argc) {
            dm_tol = atof(argv[++ii]);
            (void) dm_tol;      /* suppress unused-variable warning for now */
        } else {
            fargv[fargc++] = argv[ii];
        }
    }
    fargv[fargc] = NULL;

    if (!candlist_file || fargc < 2) {
        fprintf(stderr, "\n");
        fprintf(stderr, "        Multi-Candidate Pulsar Folding\n");
        fprintf(stderr,
                " Folds N candidates from a single raw data file.\n\n");
        fprintf(stderr,
                "Usage: prepfold_multi -candlist <file> "
                "[prepfold_options] rawfiles\n\n");
        fprintf(stderr,
                "  -candlist <file>  "
                "Candidate list (P0 Pdot DM [label] per line)\n");
        fprintf(stderr,
                "  -noplots          "
                "Skip PGPLOT output (write .pfd files only)\n");
        fprintf(stderr,
                "  --nested          "
                "Enable Level 1 (candidate) + Level 2 (subband) parallelism\n");
        fprintf(stderr,
                "  -dm_tol <val>     "
                "DM tolerance for deduplication (default %.2f)\n\n",
                dm_tol);
        exit(1);
    }

    /* ----------------------------------------------------------------
     * 2.  Parse standard prepfold options via the clig-generated parser.
     * ---------------------------------------------------------------- */

    cmd = parseCmdline(fargc, fargv);
    /* Do NOT free(fargv) here: parseCmdline sets cmd->argv = fargv+1,
     * so fargv must remain valid for the lifetime of cmd. */

    spectra_info_set_defaults(&s);
    s.filenames = cmd->argv;
    s.num_files = cmd->argc;
    if (cmd->zerodmP)
        cmd->noclipP = 1;
    s.clip_sigma = cmd->clip;
    s.apply_flipband = (cmd->invertP) ? 1 : -1;
    s.apply_weight   = (cmd->noweightsP) ? 0 : -1;
    s.apply_scale    = (cmd->noscalesP)  ? 0 : -1;
    s.apply_offset   = (cmd->nooffsetsP) ? 0 : -1;
    s.remove_zerodm  = (cmd->zerodmP)    ? 1 : 0;

#ifdef _OPENMP
    int maxcpus = omp_get_num_procs();
    int nthreads;
    if (cmd->ncpusP && cmd->ncpus > 0) {
        nthreads = (cmd->ncpus <= maxcpus) ? cmd->ncpus : maxcpus;
    } else {
        /* Default: use all available cores (unlike prepfold which defaults
         * to 1).  prepfold_multi's primary purpose is parallel folding. */
        nthreads = maxcpus;
    }
    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);
#else
    int nthreads = 1;
#endif

    if (cmd->noclipP) {
        cmd->clip = 0.0;
        s.clip_sigma = 0.0;
    }
    if (cmd->ifsP)
        s.use_poln = cmd->ifs + 1;

    obsmask.numchan = obsmask.numint = 0;

    if (cmd->timingP) {
        cmd->nosearchP  = 1;
        cmd->nopsearchP = 1;
        cmd->nopdsearchP= 1;
        cmd->nodmsearchP= 1;
        if (cmd->npart == 64) cmd->npart = 60;
        cmd->fineP = 1;
    }
    if (cmd->slowP) {
        cmd->fineP = 1;
        if (!cmd->proflen) { cmd->proflenP = 1; cmd->proflen = 100; }
    }
    if (cmd->fineP) {
        cmd->ndmfact = 1; cmd->dmstep = 1;
        cmd->npfact  = 1; cmd->pstep  = 1; cmd->pdstep = 2;
    }
    if (cmd->coarseP) {
        cmd->npfact = 4;
        cmd->pstep  = (cmd->pstep  == 1) ? 2 : 3;
        cmd->pdstep = (cmd->pdstep == 2) ? 4 : 6;
    }

    pflags.events    = cmd->eventsP;
    pflags.nosearch  = cmd->nosearchP;
    pflags.scaleparts= cmd->scalepartsP;
    pflags.justprofs = cmd->justprofsP;
    pflags.allgrey   = cmd->allgreyP;
    pflags.fixchi    = cmd->fixchiP;
    pflags.samples   = cmd->samplesP;
    pflags.showfold  = 0;

    printf("\n\n");
    printf("        Multi-Candidate Pulsar Folding\n");
    printf("  Single file pass for N candidates.\n");
    printf("           by PRESTO (Scott M. Ransom)\n\n");

    /* ----------------------------------------------------------------
     * 3.  Shared data setup  (mirrors prepfold main() lines 1346-1760)
     * ---------------------------------------------------------------- */

    /* Identify data type */
    if (RAWDATA) {
        if (cmd->filterbankP) s.datatype = SIGPROCFB;
        else if (cmd->psrfitsP) s.datatype = PSRFITS;
    } else {
        identify_psrdatatype(&s, 1);
        if      (s.datatype == SIGPROCFB) cmd->filterbankP = 1;
        else if (s.datatype == PSRFITS)   cmd->psrfitsP = 1;
        else if (s.datatype == EVENTS)    { cmd->eventsP = pflags.events = 1; }
        else if (s.datatype == SDAT)      useshorts = 1;
        else if (s.datatype == DAT)       useshorts = 0;
        else if (s.datatype == SUBBAND)   { useshorts = 1; insubs = 1; }
        else {
            fprintf(stderr,
                    "Error: cannot identify input data type. "
                    "Use -filterbank or -psrfits.\n\n");
            exit(1);
        }
    }

    if (!RAWDATA)
        s.files = (FILE **) malloc(sizeof(FILE *) * s.num_files);

    /* Open files and read metadata */
    if (RAWDATA || insubs) {
        char description[40];
        psrdatatype_description(description, s.datatype);
        if (s.num_files > 1)
            printf("Reading %s data from %d files:\n", description, s.num_files);
        else
            printf("Reading %s data from 1 file:\n", description);
        for (ii = 0; ii < s.num_files; ii++) {
            printf("  '%s'\n", cmd->argv[ii]);
            if (insubs)
                s.files[ii] = chkfopen(s.filenames[ii], "rb");
        }
        printf("\n");
        if (RAWDATA) {
            read_rawdata_files(&s);
            if (cmd->ignorechanstrP) {
                s.ignorechans = get_ignorechans(cmd->ignorechanstr, 0,
                                                s.num_channels - 1,
                                                &s.num_ignorechans,
                                                &s.ignorechans_str);
                if (s.ignorechans_str == NULL) {
                    s.ignorechans_str =
                        (char *) malloc(strlen(cmd->ignorechanstr) + 1);
                    strcpy(s.ignorechans_str, cmd->ignorechanstr);
                }
            }
            print_spectra_info_summary(&s);
            spectra_info_to_inf(&s, &idata);
            ptsperrec = s.spectra_per_subint;
            numrec    = s.N / ptsperrec;
            numchan   = s.num_channels;
            /* Auto-detect nsub from channel count */
            if (!cmd->nsubP) {
                cmd->nsub = 1;   /* flag: not yet set */
                if (numchan <= 256) {
                    if      (numchan % 32 == 0) cmd->nsub = 32;
                    else if (numchan % 30 == 0) cmd->nsub = 30;
                    else if (numchan % 25 == 0) cmd->nsub = 25;
                    else if (numchan % 20 == 0) cmd->nsub = 20;
                } else if (numchan <= 1024) {
                    if      (numchan % 8  == 0) cmd->nsub = numchan / 8;
                    else if (numchan % 10 == 0) cmd->nsub = numchan / 10;
                } else {
                    if      (numchan % 128 == 0) cmd->nsub = 128;
                    else if (numchan % 100 == 0) cmd->nsub = 100;
                }
                if (cmd->nsub == 1) {
                    fprintf(stderr,
                            "Cannot auto-determine -nsub. "
                            "Please specify it explicitly.\n\n");
                    exit(1);
                }
            }
        } else {   /* insubs */
            cmd->nsub = s.num_files;
            s.N = chkfilelen(s.files[0], sizeof(short));
            s.spectra_per_subint = ptsperrec = SUBSBLOCKLEN;
            numrec = s.N / ptsperrec;
            s.padvals = gen_fvect(s.num_files);
            for (ii = 0; ii < s.num_files; ii++) s.padvals[ii] = 0.0;
            s.start_MJD = (long double *) malloc(sizeof(long double));
            s.start_spec = (long long *) malloc(sizeof(long long));
            s.num_spec   = (long long *) malloc(sizeof(long long));
            s.num_pad    = (long long *) malloc(sizeof(long long));
            s.start_spec[0] = 0L;
            s.num_spec[0]   = s.N;
            s.num_pad[0]    = 0L;
        }
        /* Read mask if specified */
        if (cmd->maskfileP) {
            read_mask(cmd->maskfile, &obsmask);
            printf("Read mask information from '%s'\n\n", cmd->maskfile);
            if ((obsmask.numchan != idata.num_chan) ||
                (fabs(obsmask.mjd - (idata.mjd_i + idata.mjd_f)) > 1e-9)) {
                printf("WARNING: maskfile channel count or MJD mismatch! "
                       "Exiting.\n\n");
                exit(1);
            }
            good_padvals = determine_padvals(cmd->maskfile, &obsmask,
                                             s.padvals);
            maskchans = gen_ivect(obsmask.numchan);
        }
    }

    /* Non-RAWDATA: read .inf file and open data file */
    if (!RAWDATA) {
        char *root, *suffix;
        if (split_root_suffix(s.filenames[0], &root, &suffix) == 0) {
            fprintf(stderr,
                    "Error: input filename (%s) must have a suffix.\n\n",
                    s.filenames[0]);
            exit(1);
        }
        if (insubs) {
            char *tmpname;
            if (strncmp(suffix, "sub", 3) == 0) {
                tmpname = (char *) calloc(strlen(root) + 10, 1);
                sprintf(tmpname, "%s.sub", root);
                readinf(&idata, tmpname);
                free(tmpname);
                s.num_channels = numchan = idata.num_chan;
                s.start_MJD[0] = idata.mjd_i + idata.mjd_f;
                s.dt     = idata.dt;
                s.T      = s.N * s.dt;
                s.lo_freq= idata.freq;
                s.df     = idata.chan_wid;
                s.hi_freq= s.lo_freq + (s.num_channels - 1.0) * s.df;
                s.BW     = s.num_channels * s.df;
                s.fctr   = s.lo_freq - 0.5 * s.df + 0.5 * s.BW;
                s.padvals= gen_fvect(s.num_channels);
                for (ii = 0; ii < s.num_channels; ii++) s.padvals[ii] = 0.0;
                print_spectra_info_summary(&s);
                if (cmd->maskfileP) {
                    read_mask(cmd->maskfile, &obsmask);
                    printf("Read mask information from '%s'\n\n",
                           cmd->maskfile);
                    if ((obsmask.numchan != idata.num_chan) ||
                        (fabs(obsmask.mjd - (idata.mjd_i + idata.mjd_f)) > 1e-9)) {
                        printf("WARNING: maskfile mismatch! Exiting.\n\n");
                        exit(1);
                    }
                    good_padvals = determine_padvals(cmd->maskfile, &obsmask,
                                                    s.padvals);
                    maskchans = gen_ivect(obsmask.numchan);
                }
            } else {
                fprintf(stderr,
                        "Input files (%s) must be subbands (*.sub##).\n\n",
                        cmd->argv[0]);
                exit(1);
            }
        } else if (!cmd->eventsP) {
            printf("Reading input data from '%s'.\n", cmd->argv[0]);
            printf("Reading information from '%s.inf'.\n\n", root);
            readinf(&idata, root);
            cmd->nsub = 1;
            s.files[0] = chkfopen(s.filenames[0], "rb");
        }
        free(root);
        free(suffix);
    }

    /* ----------------------------------------------------------------
     * 4.  Compute shared timing quantities
     * ---------------------------------------------------------------- */

    strcpy(ephem, "DE405");
    rastring[0] = '\0'; decstring[0] = '\0';
    obs[0] = '?'; obs[1] = '\0';
    telescope_shared[0] = '\0';
    do_barycenter = 0;

    if (RAWDATA || insubs) {
        telescope_to_tempocode(idata.telescope, telescope_shared, obs);
        ra_dec_to_string(rastring,  idata.ra_h, idata.ra_m, idata.ra_s);
        ra_dec_to_string(decstring, idata.dec_d, idata.dec_m, idata.dec_s);

        recdt = idata.dt * ptsperrec;

        lorec = (long long) (cmd->startT * numrec + DBLCORRECT);
        hirec = (long long) (cmd->endT   * numrec + DBLCORRECT);
        startTday = lorec * recdt / SECPERDAY;
        numrec = hirec - lorec;

        reads_per_part = numrec / cmd->npart;
        if (numrec < cmd->npart) {
            reads_per_part = 1;
            cmd->npart = (int) numrec;
            printf("Overriding -npart to %lld (number of records).\n",
                   numrec);
        }
        numrec = reads_per_part * cmd->npart;
        T      = numrec * recdt;
        N      = (double) ((long long) numrec * ptsperrec);
        worklen = ptsperrec;

        if (idata.mjd_i) {
            tepoch_shared = idata.mjd_i + idata.mjd_f + startTday;
            if (!cmd->polycofileP && !cmd->timingP &&
                !cmd->topoP       && !cmd->parnameP) {
                double tep_copy = tepoch_shared;
                barycenter(&tep_copy, &bepoch_raw, &dtmp, 1,
                           rastring, decstring, obs, ephem);
                do_barycenter = 1;
            } else {
                bepoch_raw = tepoch_shared;
            }
        }
    } else {
        /* Float / short time series or events */
        cmd->nsub = 1;
        numchan   = 1;
        worklen   = SUBSBLOCKLEN;

        strcpy(telescope_shared, idata.telescope);

        if (!cmd->eventsP) {
            if (useshorts)
                N = (double) chkfilelen(s.files[0], sizeof(short));
            else
                N = (double) chkfilelen(s.files[0], sizeof(float));
            lorec = (long long) (cmd->startT * N + DBLCORRECT);
            hirec = (long long) (cmd->endT   * N + DBLCORRECT);
            startTday = lorec * idata.dt / SECPERDAY;
            numrec = (hirec - lorec) / worklen;
            recdt  = worklen * idata.dt;
            reads_per_part = numrec / cmd->npart;
            numrec = reads_per_part * cmd->npart;
            N = numrec * worklen;
            T = N * idata.dt;
        }

        if (idata.mjd_i) {
            tepoch_shared = idata.mjd_i + idata.mjd_f + startTday;
            bepoch_raw    = tepoch_shared;
        }
    }

    /* ----------------------------------------------------------------
     * 5.  Read candidate list and sort by DM
     * ---------------------------------------------------------------- */

    int ncands = 0;
    cand_params *cands = parse_candlist(candlist_file, &ncands);
    if (ncands == 0) {
        fprintf(stderr, "Error: no valid candidates found in '%s'\n",
                candlist_file);
        exit(1);
    }
    qsort(cands, ncands, sizeof(cand_params), cmp_by_dm);
    printf("Sorted %d candidates by DM (range %.4f -- %.4f pc/cm^3).\n\n",
           ncands, cands[0].dm, cands[ncands - 1].dm);

    /* ----------------------------------------------------------------
     * OpenMP level selection
     *
     * Level 1 (candidate-level, outer loop):
     *   Best when ncands >= 20.  Each thread reads files independently;
     *   inner subband loop runs serially (max_active_levels = 1).
     *
     * Level 2 (subband-level, inner loop):
     *   Best when ncands < 5.  Existing #pragma omp parallel for at
     *   prepfold.c:835 uses all threads within each fold_candidate call.
     *
     * --nested:  Enable both levels simultaneously.  Outer threads fold
     *   different candidates; inner threads parallelise subbands.
     * ---------------------------------------------------------------- */
#ifdef _OPENMP
    int use_parallel = 0;
    if (nthreads > 1) {
        if (nested_flag) {
            /* Level 1 + Level 2 */
            omp_set_nested(1);
            omp_set_max_active_levels(2);
            use_parallel = (ncands > 1);
            printf("OpenMP: nested mode (%d threads, outer+inner).\n\n",
                   nthreads);
        } else if (ncands < 5) {
            /* Level 2 only: serial outer loop, parallel subbands */
            use_parallel = 0;
            printf("OpenMP: Level 2 (subband parallelism, %d threads).\n\n",
                   nthreads);
        } else {
            /* Level 1: parallel outer loop, serial inner */
            omp_set_max_active_levels(1);
            use_parallel = 1;
            printf("OpenMP: Level 1 (candidate parallelism, %d threads"
                   ", %d candidates).\n\n", nthreads, ncands);
        }
    }
#endif /* _OPENMP level selection */

    /* Root name for output files */
    char *rootnm;
    {
        char *path_tmp, *filenm_tmp;
        split_path_file(cmd->argv[0], &path_tmp, &filenm_tmp);
        free(path_tmp);
        char *suf;
        char *r2;
        if (split_root_suffix(filenm_tmp, &r2, &suf) == 1) {
            free(filenm_tmp);
            rootnm = r2;
            free(suf);
        } else {
            rootnm = filenm_tmp;
        }
    }
    if (cmd->outfileP) {
        free(rootnm);
        rootnm = cmd->outfile;
    }
    if ((cmd->startT != 0.0) || (cmd->endT != 1.0)) {
        char *tmp = (char *) malloc(strlen(rootnm) + 20);
        sprintf(tmp, "%s_%4.2f-%4.2f", rootnm, cmd->startT, cmd->endT);
        if (!cmd->outfileP) free(rootnm);
        rootnm = tmp;
    }

    /* ----------------------------------------------------------------
     * 6.  Pre-compute barycentric corrections (once, shared by all candidates)
     * ---------------------------------------------------------------- */

    double *pre_barytimes = NULL, *pre_topotimes = NULL;
    int    pre_numbarypts = 0;
    double pre_avgvoverc  = 0.0;

    if ((RAWDATA || insubs) && !cmd->topoP && !cmd->polycofileP) {
        int jj;
        pre_numbarypts = (int)(T / TDT + 10);
        pre_topotimes  = gen_dvect(pre_numbarypts);
        pre_barytimes  = gen_dvect(pre_numbarypts);
        double *pre_voverc = gen_dvect(pre_numbarypts);
        for (jj = 0; jj < pre_numbarypts; jj++)
            pre_topotimes[jj] = tepoch_shared + (double) jj * TDT / SECPERDAY;
        printf("Generating barycentric corrections (shared across all candidates)...\n");
        barycenter(pre_topotimes, pre_barytimes, pre_voverc,
                   pre_numbarypts, rastring, decstring, obs, ephem);
        for (jj = 0; jj < pre_numbarypts - 1; jj++)
            pre_avgvoverc += pre_voverc[jj];
        pre_avgvoverc /= (pre_numbarypts - 1.0);
        vect_free(pre_voverc);
        printf("The average topocentric velocity is %.6g (units of c).\n\n",
               pre_avgvoverc);
    }

    /* ----------------------------------------------------------------
     * 7.  Summary storage
     * ---------------------------------------------------------------- */

    double *best_redchi = (double *) calloc(ncands, sizeof(double));
    double *best_dm     = (double *) calloc(ncands, sizeof(double));
    double *best_period = (double *) calloc(ncands, sizeof(double));

    /* ----------------------------------------------------------------
     * 7.  Per-candidate fold loop
     * ---------------------------------------------------------------- */

    printf("%-4s  %-30s  %-12s  %-12s  %-8s\n",
           "#", "Label", "P0 (s)", "DM (pc/cc)", "Status");
    printf("%.70s\n",
           "----------------------------------------------------------------------");

    /*
     * Close the files opened during shared setup.  In the loop below every
     * iteration (serial or parallel) opens its own private copy so that
     * fold_candidate can safely call close_rawfiles() at the end of each
     * call without affecting other iterations.
     */
    close_rawfiles(&s);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if(use_parallel)
#endif
    for (ic = 0; ic < ncands; ic++) {
        cand_params *c = &cands[ic];

        /* ------------------------------------------------------------------
         * Per-thread spectra_info: each thread needs independent file handles
         * so that concurrent fold_candidate() calls do not share FILE* state.
         * File opening is serialised via a critical section; everything else
         * (metadata, padvals, filenames) is read-only and safely shared.
         * ------------------------------------------------------------------ */
        struct spectra_info s_local;
#ifdef _OPENMP
#pragma omp critical(file_open)
#endif
        {
            s_local = s;     /* copy all metadata (read-only pointers) */
            if (RAWDATA) {
                /* read_rawdata_files allocates and fills s_local.fitsfiles
                 * or s_local.files, overwriting the shared (freed) pointer. */
                read_rawdata_files(&s_local);
            } else {
                /* Allocate a private files array and open handles. */
                s_local.files = (FILE **) malloc(s.num_files * sizeof(FILE *));
                if (insubs) {
                    int jj;
                    for (jj = 0; jj < s.num_files; jj++)
                        s_local.files[jj] = chkfopen(s.filenames[jj], "rb");
                } else {
                    s_local.files[0] = chkfopen(s.filenames[0], "rb");
                }
            }
        }

        /* Per-thread idata copy so idata.dm is thread-private */
        infodata idata_local = idata;
        idata_local.dm = c->dm;

        /* Per-thread maskchans: read_subbands() writes to this buffer */
        int *maskchans_local = NULL;
        if (cmd->maskfileP)
            maskchans_local = gen_ivect(obsmask.numchan);

#ifdef _OPENMP
#pragma omp critical(output)
#endif
        {
            printf("\n--- Candidate %d / %d ---\n", ic + 1, ncands);
            printf("  P0 = %.10g s  Pdot = %g s/s  DM = %.4f pc/cm^3\n",
                   c->p0, c->pdot, c->dm);
        }

        /* Build per-candidate Cmdline; leave shared options intact */
        Cmdline cmd_cand = *cmd;
        cmd_cand.pP  = 1;  cmd_cand.p  = c->p0;
        cmd_cand.pdP = 1;  cmd_cand.pd = c->pdot;
        cmd_cand.dmP = 1;  cmd_cand.dm = c->dm;
        /* Disable source-name / accel-cand / timing overrides; we use -p */
        cmd_cand.psrnameP    = 0;
        cmd_cand.parnameP    = 0;
        cmd_cand.timingP     = 0;
        cmd_cand.accelcandP  = 0;
        cmd_cand.rzwcandP    = 0;
        cmd_cand.polycofileP = 0;
        cmd_cand.binaryP     = 0;
        cmd_cand.fP          = 0;

        /* ---- Output filenames for this candidate ---- */
        char candnm[256];
        if (c->label[0]) {
            snprintf(candnm, sizeof(candnm), "%s", c->label);
        } else {
            snprintf(candnm, sizeof(candnm), "Cand%04d_DM%.2f_P%.4fms",
                     ic + 1, c->dm, c->p0 * 1000.0);
        }

        int slen = (int)(strlen(rootnm) + strlen(candnm) + 12);
        char *outfilenm  = (char *) calloc(slen, sizeof(char));
        char *plotfilenm = (char *) calloc(slen + 4, sizeof(char));
        sprintf(outfilenm,  "%s_%s.pfd",    rootnm, candnm);
        sprintf(plotfilenm, "%s_%s.pfd.ps", rootnm, candnm);

        /* ---- Initialise prepfoldinfo ---- */
        prepfoldinfo search;
        init_prepfoldinfo(&search);

        /* Fields that fold_candidate expects to already be set in *search_out */
        {
            char *pf_tmp, *fn_tmp;
            split_path_file(cmd->argv[0], &pf_tmp, &fn_tmp);
            search.filenm = fn_tmp;
            free(pf_tmp);
        }
        search.candnm = (char *) calloc(strlen(candnm) + 1, sizeof(char));
        strcpy(search.candnm, candnm);

        search.pgdev = (char *) calloc(slen + 8, sizeof(char));
        sprintf(search.pgdev, "%s/CPS", plotfilenm);

        search.telescope = (char *) calloc(strlen(telescope_shared) + 1,
                                           sizeof(char));
        strcpy(search.telescope, telescope_shared);

        search.dt     = idata_local.dt;
        search.startT = cmd->startT;
        search.endT   = cmd->endT;

        /* Per-candidate epoch */
        if (idata_local.mjd_i) {
            search.tepoch = tepoch_shared;
            if (do_barycenter) {
                search.bepoch = bepoch_raw;
                if (c->dm > 0.0) {
                    double barydispdt = delay_from_dm(
                        c->dm,
                        idata_local.freq +
                        (idata_local.num_chan - 1) * idata_local.chan_wid);
                    search.bepoch -= barydispdt / SECPERDAY;
                }
            } else {
                if (idata_local.bary)
                    search.bepoch = tepoch_shared;
                else
                    search.tepoch = tepoch_shared;
            }
        }

        /* ---- Populate fold_context ---- */
        fold_context ctx;
        ctx.s            = &s_local;   /* per-thread private file handles */
        ctx.idata        = &idata_local;
        ctx.numchan      = numchan;
        ctx.ptsperrec    = ptsperrec;
        ctx.recdt        = recdt;
        ctx.startTday    = startTday;
        ctx.lorec        = lorec;
        ctx.hirec        = hirec;
        ctx.numrec       = numrec;
        ctx.reads_per_part = reads_per_part;
        ctx.worklen      = worklen;
        ctx.insubs       = insubs;
        ctx.useshorts    = useshorts;
        ctx.good_padvals = good_padvals;
        ctx.obsmask      = &obsmask;
        ctx.maskchans    = maskchans_local;
        ctx.nummasked    = nummasked;
        strncpy(ctx.rastring,  rastring,  sizeof(ctx.rastring)  - 1);
        ctx.rastring[sizeof(ctx.rastring) - 1] = '\0';
        strncpy(ctx.decstring, decstring, sizeof(ctx.decstring) - 1);
        ctx.decstring[sizeof(ctx.decstring) - 1] = '\0';
        strncpy(ctx.obs,   obs,   sizeof(ctx.obs)   - 1);
        ctx.obs[sizeof(ctx.obs) - 1] = '\0';
        strncpy(ctx.ephem, ephem, sizeof(ctx.ephem) - 1);
        ctx.ephem[sizeof(ctx.ephem) - 1] = '\0';
        ctx.pflags       = &pflags;
        ctx.events       = NULL;
        ctx.numevents    = 0;
        ctx.T            = T;
        ctx.N            = N;
        ctx.pre_barytimes  = pre_barytimes;   /* shared; NULL → fold_candidate calls TEMPO */
        ctx.pre_topotimes  = pre_topotimes;
        ctx.pre_numbarypts = pre_numbarypts;
        ctx.pre_avgvoverc  = pre_avgvoverc;
        ctx.barytimes    = NULL;
        ctx.topotimes    = NULL;
        ctx.numbarypts   = 0;
        ctx.bestprof     = NULL;
        ctx.ppdot        = NULL;

        /* ---- Call fold_candidate ---- */
        fold_candidate(&ctx, &search, &cmd_cand, outfilenm, plotfilenm);

        /* Sync outputs from fold_context */
        double *barytimes   = ctx.barytimes;
        double *topotimes   = ctx.topotimes;
        double *bestprof    = ctx.bestprof;
        float  *ppdot       = ctx.ppdot;
        foldstats beststats;
        beststats = ctx.beststats;

        /* Store for summary table — indexed by ic, no race condition */
        best_redchi[ic] = beststats.redchi;
        best_dm[ic]     = search.bestdm;
        best_period[ic] = (search.bary.p1 > 0.0)
                          ? search.bary.p1 : search.topo.p1;

        /* ---- Write .pfd file ---- */
        write_prepfoldinfo(&search, outfilenm);

        /* ---- Plot (serialised: PGPLOT is not thread-safe) ---- */
        if (!noplots) {
#ifdef _OPENMP
#pragma omp critical(pgplot)
#endif
            prepfold_plot(&search, &pflags, 0, ppdot);
        }

#ifdef _OPENMP
#pragma omp critical(output)
#endif
        {
            printf("  Candidate %d: RedChi2=%.5f", ic + 1, beststats.redchi);
            if (cmd_cand.nsub > 1)
                printf("  BestDM=%.4f", search.bestdm);
            printf("  -> '%s'\n", outfilenm);
        }

        /* ---- Cleanup this candidate ---- */
        if (cmd_cand.nsub == 1 && !cmd_cand.searchpddP)
            vect_free(ppdot);
        delete_prepfoldinfo(&search);
        vect_free(bestprof);
        if (RAWDATA || insubs) {
            vect_free(barytimes);
            vect_free(topotimes);
        }
        if (maskchans_local)
            free(maskchans_local);
        free(outfilenm);
        free(plotfilenm);
    }

    /* Free the shared pre-computed barycentric arrays (allocated once above) */
    vect_free(pre_barytimes);
    vect_free(pre_topotimes);

    /* ----------------------------------------------------------------
     * 8.  Summary table
     * ---------------------------------------------------------------- */

    printf("\n\n=== Summary: %d candidates ===\n\n", ncands);
    printf("%-4s  %-32s  %-14s  %-12s  %-10s\n",
           "#", "Label/Name", "Best P (s)", "Best DM", "RedChi2");
    printf("%.75s\n",
           "---------------------------------------------------------------------------");
    for (ic = 0; ic < ncands; ic++) {
        char lbl[33];
        if (cands[ic].label[0])
            snprintf(lbl, sizeof(lbl), "%s", cands[ic].label);
        else
            snprintf(lbl, sizeof(lbl), "Cand%04d", ic + 1);
        printf("%-4d  %-32s  %-14.10g  %-12.4f  %-10.4f\n",
               ic + 1, lbl,
               best_period[ic], best_dm[ic], best_redchi[ic]);
    }
    printf("\n");

    /* ----------------------------------------------------------------
     * 9.  Global cleanup
     * ---------------------------------------------------------------- */

    if (cmd->maskfileP)
        free_mask(obsmask);
    if (maskchans)
        free(maskchans);
    free(cands);
    free(best_redchi);
    free(best_dm);
    free(best_period);
    if (!cmd->outfileP && (cmd->startT != 0.0 || cmd->endT != 1.0))
        free(rootnm);
    else if (!cmd->outfileP)
        free(rootnm);

    printf("Done.\n\n");
    return 0;
}

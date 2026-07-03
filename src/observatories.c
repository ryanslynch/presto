#include <string.h>
#include <stdio.h>

/* Geocentric ITRF coordinates for the radio observatories that PRESTO   */
/* knows about (see telescope_to_tempocode() in misc_utils.c), keyed by  */
/* the 2-letter ITOA code that has always been passed to barycenter().   */
/*                                                                       */
/* Coordinates are taken from TEMPO's obsys.dat where available (so that */
/* comparisons against TEMPO barycentering isolate algorithm differences */
/* rather than site differences), with PINT's observatories.json and     */
/* tempo2's observatories.dat filling the gaps (G1, LW, AT, K7).         */

typedef struct observatory {
    char code[3];               /* 2-letter ITOA code */
    char name[32];              /* Familiar name */
    double x, y, z;             /* Geocentric ITRF coordinates (m) */
} observatory;

static observatory obss[] = {
    {"GB", "GBT", 882589.289, -4924872.368, 3943729.418},
    {"AO", "Arecibo", 2390487.080, -5564731.357, 1994720.633},
    {"VL", "VLA", -1601192.0, -5041981.4, 3554871.4},
    {"PK", "Parkes", -4554231.5, 2816759.1, -3454036.3},
    {"JB", "Jodrell Bank", 3822626.04, -154105.65, 5086486.04},
    {"G1", "GB 140ft", 882872.57, -4924552.73, 3944154.92},
    {"NC", "Nancay", 4324165.81, 165927.11, 4670132.83},
    {"EF", "Effelsberg", 4033949.5, 486989.4, 4900430.8},
    {"EB", "Effelsberg", 4033949.5, 486989.4, 4900430.8},
    {"SR", "SRT", 4865182.766, 791922.689, 4035137.174},
    {"FA", "FAST", -1668557.0, 5506838.0, 2744934.0},
    {"WT", "WSRT", 3828445.659, 445223.600, 5064921.5677},
    {"WS", "WSRT", 3828445.659, 445223.600, 5064921.5677},
    {"GM", "GMRT", 1656342.30, 5797947.77, 2073243.16},
    {"CH", "CHIME", -2059166.313, -3621302.972, 4814304.113},
    {"LF", "LOFAR", 3826577.462, 461022.624, 5064892.526},
    {"LW", "LWA1", -1602196.60, -5042313.47, 3553971.51},
    {"MW", "MWA", -2559454.08, 5095372.14, -2849057.18},
    {"MK", "MeerKAT", 5109360.133, 2006852.586, -3238948.127},
    {"AT", "ATA", -2524263.18, -4123529.78, 4147966.36},
    {"K7", "KAT-7", 5109943.1050, 2003650.7359, -3239908.3195},
};

int obs_coords(const char *itoacode, double xyz[3], char *obsname)
/* Look up the geocentric ITRF coordinates (m) of an observatory  */
/* given its 2-letter ITOA code (as used by TEMPO's obsys.dat and */
/* by barycenter()).  The code "0" (or "0 ") means the geocenter. */
/* If obsname is non-NULL, the familiar name (< 32 chars) is      */
/* copied into it.  Returns 1 on success, 0 for an unknown code.  */
{
    int ii, numobs = sizeof(obss) / sizeof(observatory);

    if (itoacode[0] == '0') {   /* Geocenter */
        xyz[0] = xyz[1] = xyz[2] = 0.0;
        if (obsname)
            strcpy(obsname, "Geocenter");
        return 1;
    }
    for (ii = 0; ii < numobs; ii++) {
        if (itoacode[0] == obss[ii].code[0] && itoacode[1] == obss[ii].code[1]) {
            xyz[0] = obss[ii].x;
            xyz[1] = obss[ii].y;
            xyz[2] = obss[ii].z;
            if (obsname)
                strcpy(obsname, obss[ii].name);
            return 1;
        }
    }
    return 0;
}

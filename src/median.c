/* Fast computation of the median of an array. */
/* Note:  It messes up the order!              */

#include <gsl/gsl_statistics_float.h>

float median(float arr[], int n)
{
    /* GSL's median uses a quickselect internally, rearranges (partially */
    /* sorts) the input array, and interpolates (mean of the two central */
    /* values) for even n.                                               */
    return (float) gsl_stats_float_median(arr, 1, (size_t) n);
}

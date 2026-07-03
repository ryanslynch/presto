#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "misc_utils.h"

#if defined(__APPLE__)
#include <stdint.h>
#include <mach-o/dyld.h>
#endif

/* The absolute install data directory (<prefix>/share/presto) is compiled
 * in via -DPRESTO_DATADIR by the meson build.  Fall back to empty if, for
 * some reason, it was not defined. */
#ifndef PRESTO_DATADIR
#define PRESTO_DATADIR ""
#endif


static int file_is_readable(const char *path)
{
    return (path != NULL && access(path, R_OK) == 0);
}


static char *join_path(const char *dir, const char *sub, const char *filename)
/* Build "<dir>/<sub>/<filename>" (or "<dir>/<filename>" if sub is NULL) in a
 * freshly malloc'd string. */
{
    size_t len = strlen(dir) + strlen(filename) + 3;
    char *path;

    if (sub != NULL)
        len += strlen(sub) + 1;
    path = (char *) malloc(len);
    if (path == NULL)
        return NULL;
    if (sub != NULL)
        snprintf(path, len, "%s/%s/%s", dir, sub, filename);
    else
        snprintf(path, len, "%s/%s", dir, filename);
    return path;
}


static char *executable_dir(void)
/* Return a malloc'd path to the directory containing the running executable,
 * or NULL if it cannot be determined on this platform. */
{
    char buf[4096];
    char *slash;
    ssize_t n = -1;

#if defined(__APPLE__)
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) == 0)
        n = (ssize_t) strlen(buf);
#elif defined(__linux__)
    n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0)
        buf[n] = '\0';
#endif
    if (n <= 0)
        return NULL;
    slash = strrchr(buf, '/');
    if (slash == NULL)
        return NULL;
    *slash = '\0';
    return strdup(buf);
}


char *presto_data_path(const char *filename)
/* Return a malloc'd absolute path to a PRESTO runtime data file, searching
 * (in order): $PRESTO/lib (an optional override that keeps the source-tree
 * layout working), the compiled-in install datadir <prefix>/share/presto,
 * and a <prefix>/share/presto derived from the running executable (for
 * relocatable installs).  Returns the first location that exists; if none
 * exist, returns the compiled-in datadir path so the caller's error message
 * names the expected install location.  The caller must free() the result. */
{
    const char *presto;
    char *path, *datadir_path = NULL, *exedir;

    /* 1. $PRESTO/lib -- optional override for running from a source tree */
    presto = getenv("PRESTO");
    if (presto != NULL && presto[0] != '\0') {
        path = join_path(presto, "lib", filename);
        if (file_is_readable(path))
            return path;
        free(path);
    }

    /* 2. Compiled-in install datadir: <prefix>/share/presto */
    if (PRESTO_DATADIR[0] != '\0') {
        datadir_path = join_path(PRESTO_DATADIR, NULL, filename);
        if (file_is_readable(datadir_path))
            return datadir_path;
    }

    /* 3. <exedir>/../share/presto -- relative to the running executable, so
     *    a relocated (e.g. moved conda/pixi) environment still resolves. */
    exedir = executable_dir();
    if (exedir != NULL) {
        path = join_path(exedir, "../share/presto", filename);
        free(exedir);
        if (file_is_readable(path))
            return path;
        free(path);
    }

    /* Nothing found: hand back the compiled-in datadir path (which names the
     * expected install location), or a bare filename as a last resort. */
    if (datadir_path != NULL)
        return datadir_path;
    return join_path(".", NULL, filename);
}


char *presto_data_writepath(const char *filename)
/* Return a malloc'd absolute path naming where a PRESTO data file should be
 * *written* (e.g. by makewisdom).  Uses $PRESTO/lib when $PRESTO is set (the
 * read search order's first choice) and otherwise the compiled-in install
 * datadir <prefix>/share/presto.  The caller must free() the result. */
{
    const char *presto = getenv("PRESTO");

    if (presto != NULL && presto[0] != '\0')
        return join_path(presto, "lib", filename);
    if (PRESTO_DATADIR[0] != '\0')
        return join_path(PRESTO_DATADIR, NULL, filename);
    return join_path(".", NULL, filename);
}

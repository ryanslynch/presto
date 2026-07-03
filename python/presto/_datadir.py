"""Locate PRESTO runtime data files without requiring the ``$PRESTO``
environment variable.

Mirrors the search order used by the C resolver in ``src/datadir.c`` so the
compiled tools and the Python package agree on where shared data (the pulsar
catalog, etc.) lives:

1. ``$PRESTO/lib`` -- an optional override that keeps the source-tree layout
   working (and matches how the compiled tools behave when ``$PRESTO`` is set),
2. ``<sys.prefix>/share/presto`` -- the installed location,
3. ``<package>/../../lib`` -- the in-repo ``lib/`` directory, so an uninstalled
   source checkout works without any environment variable.
"""
import os
import sys

__all__ = ["data_path", "data_dir_candidates"]


def data_dir_candidates():
    """Return the candidate data directories, most-preferred first."""
    dirs = []
    presto = os.getenv("PRESTO")
    if presto:
        dirs.append(os.path.join(presto, "lib"))
    dirs.append(os.path.join(sys.prefix, "share", "presto"))
    # <package>/../../lib -- i.e. the repo's lib/ when running uninstalled
    here = os.path.dirname(os.path.abspath(__file__))
    dirs.append(os.path.normpath(os.path.join(here, "..", "..", "lib")))
    return dirs


def data_path(filename):
    """Return the path to a PRESTO data file (e.g. ``"psr_catalog.txt"``).

    Returns the first candidate location that actually contains ``filename``.
    If none do, the ``<sys.prefix>/share/presto`` path is returned so callers
    that are *writing* the file (or reporting an error) get the canonical
    installed location.

    Parameters
    ----------
    filename : str
        The bare name of the data file to locate.

    Returns
    -------
    str
        An absolute path to the data file.
    """
    candidates = data_dir_candidates()
    for d in candidates:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate
    # Fall back to the installed share/presto location (index 0 is $PRESTO/lib
    # only when it is set, so search for the share/presto entry explicitly).
    installed = os.path.join(sys.prefix, "share", "presto", filename)
    return installed

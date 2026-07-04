from __future__ import annotations

import numpy as np


def get_epochs(line: str) -> tuple[float, float]:
    """
    Parse an epoch line from a ``.bestprof`` file.

    Parameters
    ----------
    line : str
        A header line containing an epoch, e.g. ``"# Epoch_topo = 51234.567"``.

    Returns
    -------
    epochi : float
        The integer part of the epoch (MJD).
    epochf : float
        The fractional part of the epoch (days). If the fractional epoch is
        very close to an exact second, it is snapped to that exact second.
    """
    i, f = line.split("=")[-1].split(".")
    f = "0." + f
    epochi = float(i)
    epochf = float(f)
    # Check to see if it is very close to 1 sec
    # If it is, assume the epoch was _exactly_ at the second
    fsec = epochf * 86400.0 + 1e-10
    if np.fabs(fsec - int(fsec)) < 1e-6:
        # print "Looks like an exact second"
        epochf = float(int(fsec)) / 86400.0
    return epochi, epochf


class bestprof(object):
    """
    Parse a PRESTO ``.bestprof`` file produced by ``prepfold``.

    The header lines (beginning with ``#``) are parsed into attributes and
    the pulse profile itself is read into a list.

    Parameters
    ----------
    filenm : str
        The path to the ``.bestprof`` file to read.

    Attributes
    ----------
    datnm : str
        The name of the input data file.
    psr : str or None
        The pulsar name, or None if the candidate is not a known PSR.
    dt : float
        The sample time (T_sample) in sec.
    N : float
        The number of data points folded.
    T : float
        The total observation length in sec (``dt * N``).
    data_avg, data_std : float
        The average and standard deviation of the input data.
    prof_avg, prof_std : float
        The average and standard deviation of the profile.
    chi_sqr : float
        The reduced chi-squared of the fit.
    topo : int
        1 if a topocentric epoch was found, 0 otherwise.
    epochi, epochf : float
        The integer and fractional parts of the reference epoch (topocentric
        if available, otherwise barycentric).
    epochi_topo, epochf_topo, epochi_bary, epochf_bary : float
        The topocentric and barycentric epochs, when present.
    p0, p1, p2 : float
        The period and its first two derivatives (topocentric if available,
        otherwise barycentric), with ``p0`` in sec.
    p0err, p1err, p2err : float
        The corresponding uncertainties.
    p0_topo, p1_topo, p2_topo, p0_bary, p1_bary, p2_bary : float
        The topocentric and barycentric period parameters, when present
        (with matching ``*err_*`` uncertainties).
    profile : list of float
        The pulse profile values.
    proflen : int
        The number of bins in the profile.
    """

    def __init__(self, filenm: str):
        infile = open(filenm)
        self.topo = 0
        self.profile = []
        for line in infile.readlines():
            if line[0] == "#":
                if line.startswith("# Input file"):
                    self.datnm = line.split("=")[-1][:-1]
                    continue
                if line.startswith("# Candidate"):
                    if line.startswith("# Candidate        =  PSR_"):
                        self.psr = line.split("=")[-1].split("_")[1][:-1]
                        continue
                    else:
                        self.psr = None
                if line.startswith("# T_sample"):
                    self.dt = float(line.split("=")[-1])
                    continue
                if line.startswith("# Data Folded"):
                    self.N = float(line.split("=")[-1])
                    continue
                if line.startswith("# Data Avg"):
                    self.data_avg = float(line.split("=")[-1])
                    continue
                if line.startswith("# Data StdDev"):
                    self.data_std = float(line.split("=")[-1])
                    continue
                if line.startswith("# Profile Avg"):
                    self.prof_avg = float(line.split("=")[-1])
                    continue
                if line.startswith("# Profile StdDev"):
                    self.prof_std = float(line.split("=")[-1])
                    continue
                if line.startswith("# Reduced chi-sqr"):
                    self.chi_sqr = float(line.split("=")[-1])
                    continue
                if line.startswith("# Epoch_topo"):
                    try:
                        self.epochi, self.epochf = get_epochs(line)
                        self.epochi_topo, self.epochf_topo = self.epochi, self.epochf
                        self.topo = 1
                    except ValueError:
                        pass
                    continue
                if line.startswith("# Epoch_bary"):
                    try:
                        self.epochi_bary, self.epochf_bary = get_epochs(line)
                        if not self.topo:
                            self.epochi, self.epochf = (
                                self.epochi_bary,
                                self.epochf_bary,
                            )
                    except ValueError:
                        pass
                    continue
                if line.startswith("# P_topo"):
                    try:
                        self.p0_topo = float(line.split("=")[-1].split("+")[0]) / 1000.0
                        self.p0err_topo = (
                            float(line.split("=")[-1].split("+")[1][2:]) / 1000.0
                        )
                        if self.topo:
                            self.p0, self.p0err = self.p0_topo, self.p0err_topo
                    except Exception:
                        pass
                    continue
                if line.startswith("# P_bary"):
                    try:
                        self.p0_bary = float(line.split("=")[-1].split("+")[0]) / 1000.0
                        self.p0err_bary = (
                            float(line.split("=")[-1].split("+")[1][2:]) / 1000.0
                        )
                        if not self.topo:
                            self.p0, self.p0err = self.p0_bary, self.p0err_bary
                    except Exception:
                        pass
                    continue
                if line.startswith("# P'_topo"):
                    try:
                        self.p1_topo = float(line.split("=")[-1].split("+")[0])
                        self.p1err_topo = float(line.split("=")[-1].split("+")[1][2:])
                        if self.topo:
                            self.p1, self.p1err = self.p1_topo, self.p1err_topo
                    except Exception:
                        pass
                    continue
                if line.startswith("# P'_bary"):
                    try:
                        self.p1_bary = float(line.split("=")[-1].split("+")[0])
                        self.p1err_bary = float(line.split("=")[-1].split("+")[1][2:])
                        if not self.topo:
                            self.p1, self.p1err = self.p1_bary, self.p1err_bary
                    except Exception:
                        pass
                    continue
                if line.startswith("# P''_topo"):
                    try:
                        self.p2_topo = float(line.split("=")[-1].split("+")[0])
                        self.p2err_topo = float(line.split("=")[-1].split("+")[1][2:])
                        if self.topo:
                            self.p2, self.p2err = self.p2_topo, self.p2err_topo
                    except Exception:
                        pass
                    continue
                if line.startswith("# P''_bary"):
                    try:
                        self.p2_bary = float(line.split("=")[-1].split("+")[0])
                        self.p2err_bary = float(line.split("=")[-1].split("+")[1][2:])
                        if not self.topo:
                            self.p2, self.p2err = self.p2_bary, self.p2err_bary
                    except Exception:
                        pass
                    continue
            else:
                self.profile.append(float(line.split()[-1]))
        infile.close()
        self.T = self.dt * self.N
        self.proflen = len(self.profile)

    def normalize(self) -> np.ndarray:
        """
        Return the profile scaled to the range [0, 1].

        Returns
        -------
        numpy.ndarray
            The profile with its minimum subtracted and then divided by its
            maximum, so that values span 0 to 1.
        """
        normprof = np.asarray(self.profile)
        normprof -= min(normprof)
        normprof /= max(normprof)
        return normprof

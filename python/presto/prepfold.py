from __future__ import annotations

import sys
import copy
import random
import struct
import numbers

import numpy as np

from presto import psr_utils, infodata, polycos, Pgplot
from presto.bestprof import bestprof
from presto.presto import chi2_sigma


class pfd(object):
    """
    A folded-profile data file, as written by PRESTO's ``prepfold``.

    Parses a ``.pfd`` file (and its ``.bestprof``/``.polycos`` companions,
    if present) into the folded profiles, per-subband/interval statistics,
    and the associated fold and search parameters, and provides methods to
    dedisperse, adjust the period, combine profiles, compute reduced-chi^2,
    and plot.

    Parameters
    ----------
    filename : str
        The path to the ``.pfd`` file.
    """

    def __init__(self, filename: str):
        self.pfd_filename = filename
        infile = open(filename, "rb")
        # See if the .bestprof file is around
        try:
            self.bestprof = bestprof(filename + ".bestprof")
        except IOError:
            self.bestprof = 0
        swapchar = "<"  # this is little-endian
        data = infile.read(5 * 4)
        testswap = struct.unpack(swapchar + "i" * 5, data)
        # This is a hack to try and test the endianness of the data.
        # None of the 5 values should be a large positive number.
        if (np.fabs(np.asarray(testswap))).max() > 100000:
            swapchar = ">"  # this is big-endian
        (self.numdms, self.numperiods, self.numpdots, self.nsub, self.npart) = (
            struct.unpack(swapchar + "i" * 5, data)
        )
        (
            self.proflen,
            self.numchan,
            self.pstep,
            self.pdstep,
            self.dmstep,
            self.ndmfact,
            self.npfact,
        ) = struct.unpack(swapchar + "i" * 7, infile.read(7 * 4))

        self.filenm = infile.read(struct.unpack(swapchar + "i", infile.read(4))[0])
        self.candnm = infile.read(
            struct.unpack(swapchar + "i", infile.read(4))[0]
        ).decode("utf-8")
        self.telescope = infile.read(
            struct.unpack(swapchar + "i", infile.read(4))[0]
        ).decode("utf-8")
        self.pgdev = infile.read(struct.unpack(swapchar + "i", infile.read(4))[0])
        test = infile.read(16)
        if not test[:8] == b"Unknown" and b":" in test:
            self.rastr = test[: test.find(b"\0")]
            test = infile.read(16)
            self.decstr = test[: test.find(b"\0")]
        else:
            self.rastr = "Unknown"
            self.decstr = "Unknown"
            test = infile.read(16)
        (self.dt, self.startT) = struct.unpack(swapchar + "dd", infile.read(2 * 8))
        (
            self.endT,
            self.tepoch,
            self.bepoch,
            self.avgvoverc,
            self.lofreq,
            self.chan_wid,
            self.bestdm,
        ) = struct.unpack(swapchar + "d" * 7, infile.read(7 * 8))
        # The following "fixes" (we think) the observing frequency of the Spigot
        # based on tests done by Ingrid on 0737 (comparing it to GASP)
        # The same sorts of corrections should be made to WAPP data as well...
        # The tepoch corrections are empirically determined timing corrections
        # Note that epoch is only double precision and so the floating
        # point accuracy is ~1 us!
        if self.telescope == "GBT":
            if (
                np.fabs(np.fmod(self.dt, 8.192e-05)) < 1e-12
                and ("spigot" in filename.lower() or "guppi" not in filename.lower())
                and (self.tepoch < 54832.0)
            ):
                sys.stderr.write("Assuming SPIGOT data...\n")
                if self.chan_wid == 800.0 / 1024:  # Spigot 800 MHz mode 2
                    self.lofreq -= 0.5 * self.chan_wid
                    # original values
                    # if self.tepoch > 0.0: self.tepoch += 0.039334/86400.0
                    # if self.bestprof: self.bestprof.epochf += 0.039334/86400.0
                    # values measured with 1713+0747 wrt BCPM2 on 13 Sept 2007
                    if self.tepoch > 0.0:
                        self.tepoch += 0.039365 / 86400.0
                    if self.bestprof:
                        self.bestprof.epochf += 0.039365 / 86400.0
                elif self.chan_wid == 800.0 / 2048:
                    self.lofreq -= 0.5 * self.chan_wid
                    if self.tepoch < 53700.0:  # Spigot 800 MHz mode 16 (downsampled)
                        if self.tepoch > 0.0:
                            self.tepoch += 0.039352 / 86400.0
                        if self.bestprof:
                            self.bestprof.epochf += 0.039352 / 86400.0
                    else:  # Spigot 800 MHz mode 14
                        # values measured with 1713+0747 wrt BCPM2 on 13 Sept 2007
                        if self.tepoch > 0.0:
                            self.tepoch += 0.039365 / 86400.0
                        if self.bestprof:
                            self.bestprof.epochf += 0.039365 / 86400.0
                elif (
                    self.chan_wid == 50.0 / 1024 or self.chan_wid == 50.0 / 2048
                ):  # Spigot 50 MHz modes
                    self.lofreq += 0.5 * self.chan_wid
                    # Note: the offset has _not_ been measured for the 2048-lag mode
                    if self.tepoch > 0.0:
                        self.tepoch += 0.039450 / 86400.0
                    if self.bestprof:
                        self.bestprof.epochf += 0.039450 / 86400.0
        (self.topo_pow, tmp) = struct.unpack(swapchar + "f" * 2, infile.read(2 * 4))
        (self.topo_p1, self.topo_p2, self.topo_p3) = struct.unpack(
            swapchar + "d" * 3, infile.read(3 * 8)
        )
        (self.bary_pow, tmp) = struct.unpack(swapchar + "f" * 2, infile.read(2 * 4))
        (self.bary_p1, self.bary_p2, self.bary_p3) = struct.unpack(
            swapchar + "d" * 3, infile.read(3 * 8)
        )
        (self.fold_pow, tmp) = struct.unpack(swapchar + "f" * 2, infile.read(2 * 4))
        (self.fold_p1, self.fold_p2, self.fold_p3) = struct.unpack(
            swapchar + "d" * 3, infile.read(3 * 8)
        )
        # Save current p, pd, pdd
        # NOTE: Fold values are actually frequencies!
        self.curr_p1, self.curr_p2, self.curr_p3 = psr_utils.p_to_f(
            self.fold_p1, self.fold_p2, self.fold_p3
        )
        self.pdelays_bins = np.zeros(self.npart, dtype="d")
        (
            self.orb_p,
            self.orb_e,
            self.orb_x,
            self.orb_w,
            self.orb_t,
            self.orb_pd,
            self.orb_wd,
        ) = struct.unpack(swapchar + "d" * 7, infile.read(7 * 8))
        self.dms = np.asarray(
            struct.unpack(swapchar + "d" * self.numdms, infile.read(self.numdms * 8))
        )
        if self.numdms == 1:
            self.dms = self.dms[0]
        self.periods = np.asarray(
            struct.unpack(
                swapchar + "d" * self.numperiods, infile.read(self.numperiods * 8)
            )
        )
        self.pdots = np.asarray(
            struct.unpack(
                swapchar + "d" * self.numpdots, infile.read(self.numpdots * 8)
            )
        )
        self.numprofs = self.nsub * self.npart
        if swapchar == "<":  # little endian
            self.profs = np.zeros((self.npart, self.nsub, self.proflen), dtype="d")
            for ii in range(self.npart):
                for jj in range(self.nsub):
                    self.profs[ii, jj, :] = np.fromfile(
                        infile, np.float64, self.proflen
                    )
        else:
            self.profs = np.asarray(
                struct.unpack(
                    swapchar + "d" * self.numprofs * self.proflen,
                    infile.read(self.numprofs * self.proflen * 8),
                )
            )
            self.profs = np.reshape(self.profs, (self.npart, self.nsub, self.proflen))
        if self.numchan == 1:
            try:
                idata = infodata.infodata(
                    self.filenm[: self.filenm.rfind(b".")] + b".inf"
                )
                try:
                    if idata.waveband == "Radio":
                        self.bestdm = idata.DM
                        self.numchan = idata.numchan
                except Exception:
                    self.bestdm = 0.0
                    self.numchan = 1
            except IOError:
                print("Warning!  Can't open the .inf file for " + filename + "!")
        self.binspersec = self.fold_p1 * self.proflen
        self.chanpersub = self.numchan // self.nsub
        self.subdeltafreq = self.chan_wid * self.chanpersub
        self.hifreq = self.lofreq + (self.numchan - 1) * self.chan_wid
        self.losubfreq = self.lofreq + self.subdeltafreq - self.chan_wid
        self.subfreqs = (
            np.arange(self.nsub, dtype="d") * self.subdeltafreq + self.losubfreq
        )
        self.subdelays_bins = np.zeros(self.nsub, dtype="d")
        # Save current DM
        self.currdm = 0
        self.killed_subbands = []
        self.killed_intervals = []
        self.pts_per_fold = []
        # Note: a foldstats struct is read in as a group of 7 doubles
        # the correspond to, in order:
        #    numdata, data_avg, data_var, numprof, prof_avg, prof_var, redchi
        self.stats = np.zeros((self.npart, self.nsub, 7), dtype="d")
        for ii in range(self.npart):
            currentstats = self.stats[ii]
            for jj in range(self.nsub):
                if swapchar == "<":  # little endian
                    currentstats[jj] = np.fromfile(infile, np.float64, 7)
                else:
                    currentstats[jj] = np.asarray(
                        struct.unpack(swapchar + "d" * 7, infile.read(7 * 8))
                    )
            self.pts_per_fold.append(self.stats[ii][0][0])  # numdata from foldstats
        self.start_secs = np.add.accumulate([0] + self.pts_per_fold[:-1]) * self.dt
        self.pts_per_fold = np.asarray(self.pts_per_fold)
        self.mid_secs = self.start_secs + 0.5 * self.dt * self.pts_per_fold
        if not self.tepoch == 0.0:
            self.start_topo_MJDs = self.start_secs / 86400.0 + self.tepoch
            self.mid_topo_MJDs = self.mid_secs / 86400.0 + self.tepoch
        if not self.bepoch == 0.0:
            self.start_bary_MJDs = self.start_secs / 86400.0 + self.bepoch
            self.mid_bary_MJDs = self.mid_secs / 86400.0 + self.bepoch
        self.Nfolded = np.add.reduce(self.pts_per_fold)
        self.T = self.Nfolded * self.dt
        self.avgprof = (self.profs / self.proflen).sum()
        self.varprof = self.calc_varprof()
        # nominal number of degrees of freedom for reduced chi^2 calculation
        self.DOFnom = float(self.proflen) - 1.0
        # corrected number of degrees of freedom due to inter-bin correlations
        self.dt_per_bin = self.curr_p1 / self.proflen / self.dt
        self.DOFcor = self.DOFnom * self.DOF_corr()
        infile.close()
        self.barysubfreqs = None
        if self.avgvoverc == 0 and self.candnm.startswith("PSR_"):
            try:
                psrname = self.candnm[4:]
                self.polycos = polycos.polycos(
                    psrname, filenm=self.pfd_filename + ".polycos"
                )
                midMJD = self.tepoch + 0.5 * self.T / 86400.0
                self.avgvoverc = self.polycos.get_voverc(
                    int(midMJD), midMJD - int(midMJD)
                )
                # sys.stderr.write("Approximate Doppler velocity (in c) is:  %.4g\n"%self.avgvoverc)
                # Make the Doppler correction
                self.barysubfreqs = self.subfreqs * (1.0 + self.avgvoverc)
            except IOError:
                self.polycos = 0
        if self.barysubfreqs is None:
            self.barysubfreqs = self.subfreqs

    def __str__(self) -> str:
        out = ""
        for k, v in list(self.__dict__.items()):
            if k[:2] != "__":
                if isinstance(self.__dict__[k], str):
                    out += "%10s = '%s'\n" % (k, v)
                elif isinstance(self.__dict__[k], numbers.Integral):
                    out += "%10s = %d\n" % (k, v)
                elif isinstance(self.__dict__[k], numbers.Real):
                    out += "%10s = %-20.15g\n" % (k, v)
        return out

    def dedisperse(
        self, DM: float | None = None, interp: bool = False, doppler: bool = False
    ) -> None:
        """
        Rotate the profiles (in-place) to dedisperse them at a given DM.

        Parameters
        ----------
        DM : float, optional
            The dispersion measure to dedisperse to. If None (default), use
            ``self.bestdm``.
        interp : bool, optional
            If True, use FFT-based interpolation (default False, as in
            prepfold).
        doppler : bool, optional
            If True, Doppler shift the subband frequencies (default False).
            Note that prepfold *does* Doppler correct frequencies when
            folding raw data for a search candidate that it searches, but
            does *not* when folding with polycos for timing.

        Returns
        -------
        None
        """
        if DM is None:
            DM = self.bestdm
        # Note:  Since TEMPO Doppler corrects observing frequencies, for
        #        TOAs, at least, we need to de-disperse using topocentric
        #        observing frequencies.
        if doppler:
            freqs = psr_utils.doppler(self.subfreqs, self.avgvoverc)
        else:
            freqs = self.subfreqs
        self.subdelays = psr_utils.delay_from_DM(DM, freqs)
        self.hifreqdelay = self.subdelays[-1]
        self.subdelays = self.subdelays - self.hifreqdelay
        delaybins = self.subdelays * self.binspersec - self.subdelays_bins
        if interp:
            new_subdelays_bins = delaybins
            for ii in range(self.npart):
                for jj in range(self.nsub):
                    tmp_prof = self.profs[ii, jj, :]
                    self.profs[ii, jj] = psr_utils.fft_rotate(tmp_prof, delaybins[jj])
            # Note: Since the rotation process slightly changes the values of the
            # profs, we need to re-calculate the average profile value
            self.avgprof = (self.profs / self.proflen).sum()
        else:
            new_subdelays_bins = np.floor(delaybins + 0.5)
            for ii in range(self.nsub):
                rotbins = int(new_subdelays_bins[ii]) % self.proflen
                if rotbins:  # i.e. if not zero
                    subdata = self.profs[:, ii, :]
                    self.profs[:, ii] = np.concatenate(
                        (subdata[:, rotbins:], subdata[:, :rotbins]), 1
                    )
        self.subdelays_bins += new_subdelays_bins
        self.sumprof = self.profs.sum(0).sum(0)
        if np.fabs((self.sumprof / self.proflen).sum() - self.avgprof) > 1.0:
            print("self.avgprof is not the correct value!")
        self.currdm = DM

    def freq_offsets(
        self, p: float | None = None, pd: float | None = None, pdd: float | None = None
    ) -> tuple[float, float, float]:
        """
        Return the offsets between given frequencies and the fold frequencies.

        Parameters
        ----------
        p : float, optional
            The period to compare against. If None (default), use the best
            value.
        pd : float, optional
            The period derivative. If None (default), use the best value.
        pdd : float, optional
            The period second derivative. If None (default), use the best
            value.

        Returns
        -------
        tuple of (float, float, float)
            The frequency, frequency-derivative, and frequency-second-
            derivative offsets ``(f_diff, fd_diff, fdd_diff)``.
        """
        if self.fold_pow == 1.0:
            bestp = self.bary_p1
            bestpd = self.bary_p2
            bestpdd = self.bary_p3
        else:
            if self.topo_p1 == 0.0:
                bestp = self.fold_p1
                bestpd = self.fold_p2
                bestpdd = self.fold_p3
            else:
                bestp = self.topo_p1
                bestpd = self.topo_p2
                bestpdd = self.topo_p3
        if p is not None:
            bestp = p
        if pd is not None:
            bestpd = pd
        if pdd is not None:
            bestpdd = pdd

        # self.fold_p[123] are actually frequencies, convert to periods
        foldf, foldfd, foldfdd = self.fold_p1, self.fold_p2, self.fold_p3
        foldp, foldpd, foldpdd = psr_utils.p_to_f(
            self.fold_p1, self.fold_p2, self.fold_p3
        )

        # Get best f, fd, fdd
        # Use folding values to be consistent with prepfold_plot.c
        bestfdd = psr_utils.p_to_f(foldp, foldpd, bestpdd)[2]
        bestfd = psr_utils.p_to_f(foldp, bestpd)[1]
        bestf = 1.0 / bestp

        # Get frequency and frequency derivative offsets
        f_diff = bestf - foldf
        fd_diff = bestfd - foldfd

        # bestpdd=0.0 only if there was no searching over pdd
        if bestpdd != 0.0:
            fdd_diff = bestfdd - foldfdd
        else:
            fdd_diff = 0.0

        return (f_diff, fd_diff, fdd_diff)

    def DOF_corr(self) -> float:
        """
        Return the effective-degrees-of-freedom correction for the chi^2.

        This multiplicative correction accounts for correlations between
        profile bins caused by the way ``prepfold`` folds data (treating a
        sample as finite in duration and smearing it over potentially
        several bins, rather than as instantaneous and going into a single
        bin). The correction is semi-analytic (thanks to Paul Demorest and
        Walter Brisken); the values for ``power`` and ``factor`` were
        determined from Monte Carlos. It is good to a fractional error of a
        few percent as long as ``dt_per_bin`` is greater than ~0.5 (usually
        the case for pulsar candidates), with a very minimal
        number-of-bins dependence apparent when ``dt_per_bin`` < ~0.7.
        ``dt_per_bin`` is the width of a profile bin in samples (pulse
        period / nbins / sample time).

        Returns
        -------
        float
            The multiplicative DOF correction factor. Note that its square
            root can also be used to 'inflate' the profile RMS, e.g. for
            radiometer-equation flux density estimates.
        """
        power, factor = 1.806, 0.96  # From Monte Carlo
        return (
            self.dt_per_bin
            * factor
            * (1.0 + self.dt_per_bin ** (power)) ** (-1.0 / power)
        )

    def use_for_timing(self) -> bool:
        """
        Return whether the ``.pfd`` file can be used for timing.

        For this to return True, the pulsar must have been folded with a
        parfile and ``-no[p/pd]search`` (this includes ``-timing``), or with
        a p/pdot/pdotdot and a corresponding ``-no[p/pd]search``. If you let
        prepfold search for the best p/pdot/pdotdot, you will get bogus TOAs
        if you try to time with it.

        Returns
        -------
        bool
            True if the file is suitable for timing, False otherwise.
        """
        T = self.T
        bin_dphi = 1.0 / self.proflen
        # If any of the offsets causes more than a 0.1-bin rotation over
        # the obs, then prepfold searched and we can't time using it
        # Allow up to a 0.5 bin shift for pdd/fdd since the conversions
        # back and forth can cause float issues.
        offsets = np.fabs(np.asarray(self.freq_offsets()))
        dphis = offsets * np.asarray([T, T**2.0 / 2.0, T**3.0 / 6.0])
        if max(dphis[:2]) > 0.1 * bin_dphi or dphis[2] > 0.5 * bin_dphi:
            return False
        else:
            return True

    def time_vs_phase(
        self,
        p: float | None = None,
        pd: float | None = None,
        pdd: float | None = None,
        interp: int = 0,
    ) -> np.ndarray:
        """
        Return the 2D time-vs-phase profiles for a given period and derivatives.

        The profiles are shifted so that the given period and period
        derivatives are applied.

        Parameters
        ----------
        p : float, optional
            The period. If None (default), use the best value.
        pd : float, optional
            The period derivative. If None (default), use the best value.
        pdd : float, optional
            The period second derivative. If None (default), use the best
            value.
        interp : int, optional
            If nonzero, use FFT-based interpolation (default 0, off, as in
            prepfold).

        Returns
        -------
        numpy.ndarray
            The 2D time-vs-phase profiles, shape ``(npart, proflen)``.
        """
        # Cast to single precision and back to double precision to
        # emulate prepfold_plot.c, where parttimes is of type "float"
        # but values are upcast to "double" during computations.
        # (surprisingly, it affects the resulting profile occasionally.)
        parttimes = self.start_secs.astype("float32").astype("float64")

        # Get delays
        f_diff, fd_diff, fdd_diff = self.freq_offsets(p, pd, pdd)
        # print "DEBUG: in myprepfold.py -- parttimes", parttimes
        delays = psr_utils.delay_from_foffsets(f_diff, fd_diff, fdd_diff, parttimes)

        # Convert from delays in phase to delays in bins
        bin_delays = np.fmod(delays * self.proflen, self.proflen) - self.pdelays_bins

        # Rotate subintegrations
        # subints = self.combine_profs(self.npart, 1)[:,0,:] # Slower than sum by ~9x
        subints = np.sum(self.profs, axis=1).squeeze()
        if interp:
            new_pdelays_bins = bin_delays
            for ii in range(self.npart):
                tmp_prof = subints[ii, :]
                # Negative sign in num bins to shift because we calculated delays
                # Assuming +ve is shift-to-right, psr_utils.rotate assumes +ve
                # is shift-to-left
                subints[ii, :] = psr_utils.fft_rotate(tmp_prof, -new_pdelays_bins[ii])
        else:
            new_pdelays_bins = np.floor(bin_delays + 0.5)
            indices = np.outer(np.arange(self.proflen), np.ones(self.npart))
            indices = np.mod(indices - new_pdelays_bins, self.proflen).T
            indices += np.outer(
                np.arange(self.npart) * self.proflen, np.ones(self.proflen)
            )
            subints = subints.flatten("C")[indices.astype("i8")]
        return subints

    def adjust_period(
        self,
        p: float | None = None,
        pd: float | None = None,
        pdd: float | None = None,
        interp: int = 0,
    ) -> None:
        """
        Rotate the profiles (in-place) to a given period and derivatives.

        By default, use the 'best' values as determined by prepfold's
        search. This should orient all of the profiles so that they are
        almost identical to what you see in a prepfold plot that used
        searching.

        Parameters
        ----------
        p : float, optional
            The period. If None (default), use the best value.
        pd : float, optional
            The period derivative. If None (default), use the best value.
        pdd : float, optional
            The period second derivative. If None (default), use the best
            value.
        interp : int, optional
            If nonzero, use FFT-based interpolation (default 0, off, as in
            prepfold).

        Returns
        -------
        None
        """
        if self.fold_pow == 1.0:
            bestp = self.bary_p1
            bestpd = self.bary_p2
            bestpdd = self.bary_p3
        else:
            bestp = self.topo_p1
            bestpd = self.topo_p2
            bestpdd = self.topo_p3
        if p is None:
            p = bestp
        if pd is None:
            pd = bestpd
        if pdd is None:
            pdd = bestpdd

        # Cast to single precision and back to double precision to
        # emulate prepfold_plot.c, where parttimes is of type "float"
        # but values are upcast to "double" during computations.
        # (surprisingly, it affects the resulting profile occasionally.)
        parttimes = self.start_secs.astype("float32").astype("float64")

        # Get delays
        f_diff, fd_diff, fdd_diff = self.freq_offsets(p, pd, pdd)
        delays = psr_utils.delay_from_foffsets(f_diff, fd_diff, fdd_diff, parttimes)

        # Convert from delays in phase to delays in bins
        bin_delays = np.fmod(delays * self.proflen, self.proflen) - self.pdelays_bins
        if interp:
            new_pdelays_bins = bin_delays
        else:
            new_pdelays_bins = np.floor(bin_delays + 0.5)

        # Rotate subintegrations
        for ii in range(self.nsub):
            for jj in range(self.npart):
                tmp_prof = self.profs[jj, ii, :]
                # Negative sign in num bins to shift because we calculated delays
                # Assuming +ve is shift-to-right, psr_utils.rotate assumes +ve
                # is shift-to-left
                if interp:
                    self.profs[jj, ii] = psr_utils.fft_rotate(
                        tmp_prof, -new_pdelays_bins[jj]
                    )
                else:
                    self.profs[jj, ii] = psr_utils.rotate(
                        tmp_prof, -new_pdelays_bins[jj]
                    )
        self.pdelays_bins += new_pdelays_bins
        if interp:
            # Note: Since the rotation process slightly changes the values of the
            # profs, we need to re-calculate the average profile value
            self.avgprof = (self.profs / self.proflen).sum()

        self.sumprof = self.profs.sum(0).sum(0)
        if np.fabs((self.sumprof / self.proflen).sum() - self.avgprof) > 1.0:
            print("self.avgprof is not the correct value!")

        # Save current p, pd, pdd
        self.curr_p1, self.curr_p2, self.curr_p3 = p, pd, pdd

    def combine_profs(self, new_npart: int, new_nsub: int) -> np.ndarray | None:
        """
        Combine intervals and/or subbands together into a new profile array.

        Parameters
        ----------
        new_npart : int
            The new number of intervals. Must be a divisor of the current
            number of intervals.
        new_nsub : int
            The new number of subbands. Must be a divisor of the current
            number of subbands.

        Returns
        -------
        numpy.ndarray or None
            The combined profiles, shape ``(new_npart, new_nsub, proflen)``,
            or None if `new_npart`/`new_nsub` are not valid divisors.
        """
        if self.npart % new_npart:
            print("Warning!  The new number of intervals (%d) is not a" % new_npart)
            print(
                "          divisor of the original number of intervals (%d)!"
                % self.npart
            )
            print("Doing nothing.")
            return None
        if self.nsub % new_nsub:
            print("Warning!  The new number of subbands (%d) is not a" % new_nsub)
            print(
                "          divisor of the original number of subbands (%d)!" % self.nsub
            )
            print("Doing nothing.")
            return None

        dp = self.npart // new_npart
        ds = self.nsub // new_nsub

        newprofs = np.zeros((new_npart, new_nsub, self.proflen), "d")
        for ii in range(new_npart):
            # Combine the subbands if required
            if self.nsub > 1:
                for jj in range(new_nsub):
                    subprofs = np.add.reduce(self.profs[:, jj * ds : (jj + 1) * ds], 1)
                    # Combine the time intervals
                    newprofs[ii][jj] = np.add.reduce(subprofs[ii * dp : (ii + 1) * dp])
            else:
                newprofs[ii][0] = np.add.reduce(self.profs[ii * dp : (ii + 1) * dp, 0])
        return newprofs

    def kill_intervals(self, intervals) -> None:
        """
        Zero out (in-place) a list of intervals, effectively 'killing' them.

        Parameters
        ----------
        intervals : sequence of int
            The indices of the intervals to zero out.

        Returns
        -------
        None
        """
        self.killed_intervals = []
        for part in intervals:
            self.profs[part, :, :] *= 0.0
            self.killed_intervals.append(part)
        # Update the stats and summed profile
        self.avgprof = (self.profs / self.proflen).sum()
        self.varprof = self.calc_varprof()
        self.sumprof = self.profs.sum(0).sum(0)

    def kill_subbands(self, subbands) -> None:
        """
        Zero out (in-place) a list of subbands, effectively 'killing' them.

        Parameters
        ----------
        subbands : sequence of int
            The indices of the subbands to zero out.

        Returns
        -------
        None
        """
        self.killed_subbands = []
        for sub in subbands:
            self.profs[:, sub, :] *= 0.0
            self.killed_subbands.append(sub)
        # Update the stats
        self.avgprof = (self.profs / self.proflen).sum()
        self.varprof = self.calc_varprof()
        self.sumprof = self.profs.sum(0).sum(0)

    def plot_sumprof(self, device: str = "/xwin", **kwargs) -> None:
        """
        Plot the dedispersed and summed profile.

        Parameters
        ----------
        device : str, optional
            The PGPLOT device (default "/xwin").
        **kwargs
            Additional keyword arguments passed to ``Pgplot.plotxy``.

        Returns
        -------
        None
        """
        if "subdelays" not in self.__dict__:
            print("Dedispersing first...")
            self.dedisperse()
        normprof = self.sumprof - min(self.sumprof)
        normprof /= max(normprof)
        Pgplot.plotxy(
            normprof, labx="Phase Bins", laby="Normalized Flux", device=device, **kwargs
        )

    def greyscale(self, array2d: np.ndarray, **kwargs) -> None:
        """
        Plot a 2D array as a greyscale image, using prepfold's scalings.

        Parameters
        ----------
        array2d : numpy.ndarray
            The 2D array to plot.
        **kwargs
            Additional keyword arguments passed to ``Pgplot.plot2d``.

        Returns
        -------
        None
        """
        # Use the same scaling as in prepfold_plot.c
        global_max = np.maximum.reduce(np.maximum.reduce(array2d))
        if global_max == 0.0:
            global_max = 1.0
        min_parts = np.minimum.reduce(array2d, 1)
        array2d = (array2d - min_parts[:, np.newaxis]) / np.fabs(global_max)
        Pgplot.plot2d(array2d, image="antigrey", **kwargs)

    def plot_intervals(
        self,
        phasebins: str | tuple[int, int] = "All",
        device: str = "/xwin",
        **kwargs,
    ) -> None:
        """
        Plot the subband-summed profiles versus time.

        Parameters
        ----------
        phasebins : str or tuple of (int, int), optional
            If a ``(low, high)`` tuple, restrict the plotted bins to that
            slice. If "All" (default), plot all bins.
        device : str, optional
            The PGPLOT device (default "/xwin").
        **kwargs
            Additional keyword arguments passed to :meth:`greyscale`.

        Returns
        -------
        None
        """
        if "subdelays" not in self.__dict__:
            print("Dedispersing first...")
            self.dedisperse()
        if phasebins != "All":
            lo, hi = phasebins
            profs = self.profs[:, :, lo:hi].sum(1)
        else:
            lo, hi = 0.0, self.proflen
            profs = self.profs.sum(1)
        self.greyscale(
            profs,
            rangex=[lo, hi],
            rangey=[0.0, self.npart],
            labx="Phase Bins",
            labx2="Pulse Phase",
            laby="Time Intervals",
            rangex2=np.asarray([lo, hi]) * 1.0 / self.proflen,
            laby2="Time (s)",
            rangey2=[0.0, self.T],
            device=device,
            **kwargs,
        )

    def plot_subbands(
        self,
        phasebins: str | tuple[int, int] = "All",
        device: str = "/xwin",
        **kwargs,
    ) -> None:
        """
        Plot the interval-summed profiles versus subband.

        Parameters
        ----------
        phasebins : str or tuple of (int, int), optional
            If a ``(low, high)`` tuple, restrict the plotted bins to that
            slice. If "All" (default), plot all bins.
        device : str, optional
            The PGPLOT device (default "/xwin").
        **kwargs
            Additional keyword arguments passed to :meth:`greyscale`.

        Returns
        -------
        None
        """
        if "subdelays" not in self.__dict__:
            print("Dedispersing first...")
            self.dedisperse()
        if phasebins != "All":
            lo, hi = phasebins
            profs = self.profs[:, :, lo:hi].sum(0)
        else:
            lo, hi = 0.0, self.proflen
            profs = self.profs.sum(0)
        lof = self.lofreq - 0.5 * self.chan_wid
        hif = lof + self.chan_wid * self.numchan
        self.greyscale(
            profs,
            rangex=[lo, hi],
            rangey=[0.0, self.nsub],
            labx="Phase Bins",
            labx2="Pulse Phase",
            laby="Subbands",
            rangex2=np.asarray([lo, hi]) * 1.0 / self.proflen,
            laby2="Frequency (MHz)",
            rangey2=[lof, hif],
            device=device,
            **kwargs,
        )

    def calc_varprof(self) -> float:
        """
        Calculate the summed-profile variance of the current pfd.

        Killed profiles (intervals and subbands) are ignored.

        Returns
        -------
        float
            The summed-profile variance.
        """
        varprof = 0.0
        submask = ~np.isin(np.arange(self.nsub), self.killed_subbands)
        for part in range(self.npart):
            if part in self.killed_intervals:
                continue
            varprof += self.stats[part, submask, 5].sum()
        return varprof

    def calc_redchi2(
        self,
        prof: np.ndarray | None = None,
        avg: float | None = None,
        var: float | None = None,
    ) -> float:
        """
        Return the reduced-chi^2 of the current (or a given) summed profile.

        Parameters
        ----------
        prof : numpy.ndarray, optional
            The profile to use. If None (default), use ``self.sumprof``.
        avg : float, optional
            The average to use. If None (default), use ``self.avgprof``.
        var : float, optional
            The variance to use. If None (default), use ``self.varprof``.

        Returns
        -------
        float
            The reduced-chi^2 (using the correlation-corrected DOF).
        """
        if "subdelays" not in self.__dict__:
            print("Dedispersing first...")
            self.dedisperse()
        if prof is None:
            prof = self.sumprof
        if avg is None:
            avg = self.avgprof
        if var is None:
            var = self.varprof
        # Note:  use the _corrected_ DOF for reduced chi^2 calculation
        return ((prof - avg) ** 2.0 / var).sum() / self.DOFcor

    def calc_sigma(self) -> float:
        """
        Return the equivalent-Gaussian sigma of the summed profile.

        Returns
        -------
        float
            The significance of the summed profile, in Gaussian sigma.
        """
        return chi2_sigma(self.calc_redchi2() * self.DOFcor, self.DOFcor)

    def plot_chi2_vs_DM(
        self,
        loDM: float,
        hiDM: float,
        N: int = 100,
        interp: int = 0,
        device: str = "/xwin",
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Plot (and return) the reduced-chi^2 versus DM.

        Parameters
        ----------
        loDM : float
            The low end of the DM range.
        hiDM : float
            The high end of the DM range.
        N : int, optional
            The number of DMs spanning ``loDM``-``hiDM`` (default 100).
        interp : int, optional
            If nonzero, use sinc interpolation (default 0, off).
        device : str, optional
            The PGPLOT device (default "/xwin").
        **kwargs
            Additional keyword arguments passed to ``Pgplot.plotxy``.

        Returns
        -------
        chis : numpy.ndarray
            The reduced-chi^2 at each DM.
        DMs : numpy.ndarray
            The DMs that were tried.
        """
        # Sum the profiles in time
        sumprofs = self.profs.sum(0)
        if not interp:
            profs = sumprofs
        else:
            profs = np.zeros(np.shape(sumprofs), dtype="d")
        DMs = psr_utils.span(loDM, hiDM, N)
        chis = np.zeros(N, dtype="f")
        subdelays_bins = self.subdelays_bins.copy()
        for ii, DM in enumerate(DMs):
            subdelays = psr_utils.delay_from_DM(DM, self.barysubfreqs)
            hifreqdelay = subdelays[-1]
            subdelays = subdelays - hifreqdelay
            delaybins = subdelays * self.binspersec - subdelays_bins
            if interp:
                interp_factor = 16
                for jj in range(self.nsub):
                    profs[jj] = psr_utils.interp_rotate(
                        sumprofs[jj], delaybins[jj], zoomfact=interp_factor
                    )
                # Note: Since the interpolation process slightly changes the values of the
                # profs, we need to re-calculate the average profile value
                avgprof = (profs / self.proflen).sum()
            else:
                new_subdelays_bins = np.floor(delaybins + 0.5)
                for jj in range(self.nsub):
                    profs[jj] = psr_utils.rotate(profs[jj], int(new_subdelays_bins[jj]))
                subdelays_bins += new_subdelays_bins
                avgprof = self.avgprof
            sumprof = profs.sum(0)
            chis[ii] = self.calc_redchi2(prof=sumprof, avg=avgprof)
        # Now plot it
        Pgplot.plotxy(
            chis, DMs, labx="DM", laby=r"Reduced-\gx\u2\d", device=device, **kwargs
        )
        return (chis, DMs)

    def plot_chi2_vs_sub(self, device: str = "/xwin", **kwargs) -> np.ndarray:
        """
        Plot (and return) the reduced-chi^2 versus subband number.

        Parameters
        ----------
        device : str, optional
            The PGPLOT device (default "/xwin").
        **kwargs
            Additional keyword arguments passed to ``Pgplot.plotxy``.

        Returns
        -------
        numpy.ndarray
            The reduced-chi^2 for each subband.
        """
        # Sum the profiles in each subband
        profs = self.profs.sum(0)
        # Compute the averages and variances for the subbands
        avgs = profs.sum(1) / self.proflen
        vars = []
        for sub in range(self.nsub):
            var = 0.0
            if sub in self.killed_subbands:
                vars.append(var)
                continue
            for part in range(self.npart):
                if part in self.killed_intervals:
                    continue
                var += self.stats[part][sub][5]  # foldstats prof_var
            vars.append(var)
        chis = np.zeros(self.nsub, dtype="f")
        for ii in range(self.nsub):
            chis[ii] = self.calc_redchi2(prof=profs[ii], avg=avgs[ii], var=vars[ii])
        # Now plot it
        Pgplot.plotxy(
            chis,
            labx="Subband Number",
            laby=r"Reduced-\gx\u2\d",
            rangey=[0.0, max(chis) * 1.1],
            device=device,
            **kwargs,
        )
        return chis

    def estimate_offsignal_redchi2(self, numtrials: int = 20) -> float:
        """
        Estimate the off-signal reduced-chi^2.

        The estimate is based on randomly shifting and summing all of the
        component profiles.

        Parameters
        ----------
        numtrials : int, optional
            The number of random trials to average over (default 20).

        Returns
        -------
        float
            The mean off-signal reduced-chi^2.
        """
        redchi2s = []
        for count in range(numtrials):
            prof = np.zeros(self.proflen, dtype="d")
            for ii in range(self.npart):
                for jj in range(self.nsub):
                    tmpprof = copy.copy(self.profs[ii][jj])
                    prof += psr_utils.rotate(tmpprof, random.randrange(0, self.proflen))
            redchi2s.append(self.calc_redchi2(prof=prof))
        return np.mean(redchi2s)

    def adjust_fold_frequency(
        self, phasebins: float, profs: np.ndarray | None = None, shiftsubs: bool = False
    ) -> tuple[np.ndarray, float]:
        """
        Change the apparent folding frequency by shifting intervals in phase.

        Linearly shifts the intervals by `phasebins` over the course of the
        observation.

        Parameters
        ----------
        phasebins : float
            The total number of phase bins to shift over the whole
            observation.
        profs : numpy.ndarray, optional
            The profiles to use instead of ``self.profs`` (default None).
        shiftsubs : bool, optional
            If True, correct the individual subbands instead of a 2D
            projection of them (default False).

        Returns
        -------
        profs : numpy.ndarray
            The (dedispersed) profiles as a function of time.
        redchi : float
            The reduced-chi^2 of the resulting summed profile.
        """
        if "subdelays" not in self.__dict__:
            print("Dedispersing first...")
            self.dedisperse()
        if shiftsubs:
            print("Shifting all the subbands...")
            if profs is None:
                profs = self.profs
            for ii in range(self.npart):
                bins_to_shift = int(round(float(ii) / self.npart * phasebins))
                for jj in range(self.nsub):
                    profs[ii, jj] = psr_utils.rotate(profs[ii, jj], bins_to_shift)
            redchi = self.calc_redchi2(prof=profs.sum(0).sum(0))
        else:
            print("Shifting just the projected intervals (not individual subbands)...")
            if profs is None:
                profs = self.profs.sum(1)
            for ii in range(self.npart):
                bins_to_shift = int(round(float(ii) / self.npart * phasebins))
                profs[ii] = psr_utils.rotate(profs[ii], bins_to_shift)
            redchi = self.calc_redchi2(prof=profs.sum(0))
        print("New reduced-chi^2 =", redchi)
        return profs, redchi

    def dynamic_spectra(
        self,
        onbins,
        combineints: int = 1,
        combinechans: int = 1,
        calibrate: bool = True,
        plot: bool = True,
        device: str = "/xwin",
        **kwargs,
    ) -> np.ndarray:
        """
        Return (and plot) the dynamic spectrum from the folds in the .pfd.

        Assumes the pulsar is 'on' during the bins specified in `onbins` and
        off elsewhere, forming ON-OFF.

        Parameters
        ----------
        onbins : sequence of int
            The phase bins during which the pulsar is 'on'.
        combineints : int, optional
            The number of adjacent intervals to combine (default 1).
        combinechans : int, optional
            The number of adjacent frequency channels to combine (default 1).
        calibrate : bool, optional
            If True, the DS will be (ON-OFF)/OFF (default True).
        plot : bool, optional
            If True, plot the dynamic spectrum (default True).
        device : str, optional
            The PGPLOT device (default "/xwin").
        **kwargs
            Additional keyword arguments passed to :meth:`greyscale`.

        Returns
        -------
        numpy.ndarray
            The dynamic spectrum.
        """
        # Determine the indices of the off-pulse region
        indices = np.arange(self.proflen)
        np.put(indices, np.asarray(onbins), -1)
        offbins = np.compress(indices >= 0, np.arange(self.proflen))
        numon = len(onbins)
        numoff = len(offbins)
        # De-disperse if required first
        if "subdelays" not in self.__dict__:
            print("Dedispersing first...")
            self.dedisperse()
        # The following is the average offpulse level
        offpulse = np.sum(np.take(self.profs, offbins, 2), 2) / float(numoff)
        # The following is the average onpulse level
        onpulse = np.sum(np.take(self.profs, onbins, 2), 2) / float(numon)
        # Now make the DS
        self.DS = onpulse - offpulse
        self.DSnpart = self.npart
        self.DSstart_secs = self.start_secs
        self.DSintdt = self.DSstart_secs[1] - self.DSstart_secs[0]
        self.DSnsub = self.nsub
        self.DSsubfreqs = self.subfreqs
        self.DSsubdeltafreq = self.subdeltafreq
        if calibrate:
            # Protect against division by zero
            offpulse[offpulse == 0.0] = 1.0
            self.DS /= offpulse
        # Combine intervals if required
        if combineints > 1:
            # First chop off any extra intervals
            if self.npart % combineints:
                self.DSnpart = (self.npart // combineints) * combineints
                self.DS = self.DS[: self.DSnpart, :]
            # Now reshape and add the neighboring intervals
            self.DS = np.reshape(
                self.DS, (self.DSnpart // combineints, combineints, self.DSnsub)
            )
            print(np.shape(self.DS))
            self.DS = np.sum(self.DS, 1)
            self.DSstart_secs = self.DSstart_secs[::combineints]
            self.DSintdt *= combineints
            self.DSnpart //= combineints
        # Combine channels if required
        if combinechans > 1:
            # First chop off any extra channels
            if self.nsub % combinechans:
                self.DSnsub = (self.nsub // combinechans) * combinechans
                self.DS = self.DS[:, : self.DSnsub]
            # Now reshape and add the neighboring intervals
            self.DS = np.reshape(
                self.DS, (self.DSnpart, self.DSnsub // combinechans, combinechans)
            )
            self.DS = np.sum(self.DS, 2)
            self.DSsubfreqs = psr_utils.running_avg(
                self.subfreqs[: self.DSnsub], combinechans
            )
            self.DSsubdeltafreq *= combinechans
            self.DSnsub //= combinechans
        print("DS shape = ", np.shape(self.DS))
        # Plot it if required
        if plot:
            lof = self.subfreqs[0] - 0.5 * self.DSsubdeltafreq
            hif = self.subfreqs[-1] + 0.5 * self.DSsubdeltafreq
            lot = 0.0
            hit = self.DSstart_secs[-1] + self.DSintdt
            self.greyscale(
                self.DS,
                rangex=[lof, hif],
                rangey=[lot, hit],
                labx="Frequency (MHz)",
                labx2="Subband Number",
                laby="Time (s)",
                laby2="Interval Number",
                rangex2=[0, self.DSnsub],
                rangey2=[0, self.DSnpart],
                device=device,
                **kwargs,
            )
        return self.DS


if __name__ == "__main__":
    # testpfd = "/home/ransom/tmp_pfd/M5_52725_W234_PSR_1518+0204A.pfd"
    # testpfd = "/home/ransom/tmp_pfd/M13_52724_W234_PSR_1641+3627C.pfd"
    testpfd = "M13_53135_W34_rficlean_DM30.10_PSR_1641+3627C.pfd"

    tp = pfd(testpfd)

    if 0:
        print(tp.start_secs)
        print(tp.mid_secs)
        print(tp.start_topo_MJDs)
        print(tp.mid_topo_MJDs)
        print(tp.T)

    # tp.kill_subbands([6,7,8,9,30,31,32,33])
    # tp.kill_intervals([2,3,4,5,6])

    # tp.plot_chi2_vs_sub()
    # (chis, DMs) = tp.plot_chi2_vs_DM(0.0, 50.0, 501, interp=1)
    # best_index = np.argmax(chis)
    # print "Best DM = ", DMs[best_index]

    (chis, DMs) = tp.plot_chi2_vs_DM(0.0, 50.0, 501)
    best_index = np.argmax(chis)
    print("Best DM = ", DMs[best_index])

    tp.dedisperse()
    tp.plot_subbands()
    tp.plot_sumprof()
    print("DM =", tp.bestdm, "gives reduced chi^2 =", tp.calc_redchi2())

    tp.dedisperse(27.0)
    tp.plot_subbands()
    tp.plot_sumprof()
    print("DM = 27.0 gives reduced chi^2 =", tp.calc_redchi2())

    tp.dedisperse(33.0)
    tp.plot_subbands()
    tp.plot_sumprof()
    print("DM = 33.0 gives reduced chi^2 =", tp.calc_redchi2())

    tp.plot_intervals()

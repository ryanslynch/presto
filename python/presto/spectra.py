from __future__ import annotations

import copy

import numpy as np
import scipy.signal

from presto import psr_utils


class Spectra(object):
    """
    A class to store spectra, mainly to provide reusable functionality.

    Parameters
    ----------
    freqs : numpy.ndarray
        The observing frequency for each channel.
    dt : float
        The sample time in seconds.
    data : numpy.ndarray
        A 2D array of pulsar data. Axis 0 contains channels (e.g.
        ``data[0, :]``) and axis 1 contains spectra (e.g. ``data[:, 0]``).
    starttime : float, optional
        The start time in seconds of the spectra with respect to the start
        of the observation (default 0).
    dm : float, optional
        The dispersion measure in pc/cm^3 (default 0).

    Attributes
    ----------
    numchans : int
        The number of channels.
    numspectra : int
        The number of spectra.
    freqs : numpy.ndarray
        The observing frequency for each channel.
    data : numpy.ndarray
        The 2D data array (cast to float).
    dt : float
        The sample time in seconds.
    starttime : float
        The start time in seconds.
    dm : float
        The dispersion measure in pc/cm^3.
    """

    def __init__(
        self,
        freqs: np.ndarray,
        dt: float,
        data: np.ndarray,
        starttime: float = 0,
        dm: float = 0,
    ):
        self.numchans, self.numspectra = data.shape
        assert len(freqs) == self.numchans

        self.freqs = freqs
        self.data = data.astype("float")
        self.dt = dt
        self.starttime = starttime
        self.dm = dm

    def __str__(self) -> str:
        return str(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value) -> None:
        self.data[key] = value

    def get_chan(self, channum: int) -> np.ndarray:
        """Return the data for channel `channum`."""
        return self.data[channum, :]

    def get_spectrum(self, specnum: int) -> np.ndarray:
        """Return the spectrum (all channels) at sample index `specnum`."""
        return self.data[:, specnum]

    def shift_channels(self, bins: np.ndarray, padval: float | str = 0) -> None:
        """
        Shift each channel to the left by the corresponding value in `bins`.

        Shifting happens in-place.

        Parameters
        ----------
        bins : numpy.ndarray
            The number of bins to shift each channel by.
        padval : float or str, optional
            The value to use when shifting near the edge of a channel. This
            can be a numeric value, or one of "median", "mean", or "rotate"
            (default 0). "median" and "mean" use the median/mean of the
            channel; "rotate" takes values from one end of the channel and
            shifts them to the other.

        Returns
        -------
        None
        """
        assert self.numchans == len(bins)
        for ii in range(self.numchans):
            chan = self.get_chan(ii)
            # Use 'chan[:]' so update happens in-place
            # this way the change effects self.data
            chan[:] = psr_utils.rotate(chan, bins[ii])
            if padval != "rotate":
                # Get padding value
                if padval == "mean":
                    pad = np.mean(chan)
                elif padval == "median":
                    pad = np.median(chan)
                else:
                    pad = padval

                # Replace rotated values with padval
                if bins[ii] > 0:
                    chan[-bins[ii] :] = pad
                elif bins[ii] < 0:
                    chan[: -bins[ii]] = pad

    def subband(
        self, nsub: int, subdm: float | None = None, padval: float | str = 0
    ) -> None:
        """
        Reduce the number of channels to `nsub` by subbanding.

        Subbanding happens in-place.

        Parameters
        ----------
        nsub : int
            The number of subbands. Must be a factor of the number of
            channels.
        subdm : float, optional
            The DM with which to combine channels within each subband. If
            None (default), channels within a subband are not shifted.
        padval : float or str, optional
            The padding value to use when shifting channels during
            dedispersion (default 0). See :meth:`shift_channels`.

        Returns
        -------
        None
        """
        assert (self.numchans % nsub) == 0
        assert (subdm is None) or (subdm >= 0)
        nchan_per_sub = self.numchans // nsub
        sub_hifreqs = self.freqs[np.arange(nsub) * nchan_per_sub]
        sub_lofreqs = self.freqs[(1 + np.arange(nsub)) * nchan_per_sub - 1]
        sub_ctrfreqs = 0.5 * (sub_hifreqs + sub_lofreqs)

        if subdm is not None:
            # Compute delays
            ref_delays = psr_utils.delay_from_DM(subdm - self.dm, sub_ctrfreqs)
            delays = psr_utils.delay_from_DM(subdm - self.dm, self.freqs)
            rel_delays = delays - ref_delays.repeat(nchan_per_sub)  # Relative delay
            rel_bindelays = np.round(rel_delays / self.dt).astype("int")
            # Shift channels
            self.shift_channels(rel_bindelays, padval)

        # Subband
        self.data = np.array(
            [np.sum(sub, axis=0) for sub in np.vsplit(self.data, nsub)]
        )
        self.freqs = sub_ctrfreqs
        self.numchans = nsub

    def scaled(self, indep: bool = False) -> Spectra:
        """
        Return a scaled copy of the Spectra object.

        Each channel has its median subtracted, then is divided by the
        global standard deviation (``indep=False``) or by its own standard
        deviation (``indep=True``).

        Parameters
        ----------
        indep : bool, optional
            If True, scale each channel independently (default False).

        Returns
        -------
        Spectra
            A scaled copy of the Spectra object.
        """
        other = copy.deepcopy(self)
        if not indep:
            std = other.data.std()
        for ii in range(other.numchans):
            chan = other.get_chan(ii)
            median = np.median(chan)
            if indep:
                std = chan.std()
            chan[:] = (chan - median) / std
        return other

    def scaled2(self, indep: bool = False) -> Spectra:
        """
        Return a scaled copy of the Spectra object.

        Each channel has its minimum subtracted, then is divided by the
        global maximum (``indep=False``) or by its own maximum
        (``indep=True``).

        Parameters
        ----------
        indep : bool, optional
            If True, scale each channel independently (default False).

        Returns
        -------
        Spectra
            A scaled copy of the Spectra object.
        """
        other = copy.deepcopy(self)
        if not indep:
            max = other.data.max()
        for ii in range(other.numchans):
            chan = other.get_chan(ii)
            min = chan.min()
            if indep:
                max = chan.max()
            chan[:] = (chan - min) / max
        return other

    def masked(
        self, mask: np.ndarray, maskval: float | str = "median-mid80"
    ) -> Spectra:
        """
        Replace fully-masked channels with `maskval`, in-place.

        Parameters
        ----------
        mask : numpy.ndarray
            A boolean array of the same shape as ``self.data``. True marks
            entries to be masked.
        maskval : float or str, optional
            The value to use when masking. This can be a numeric value, or
            one of "median", "mean", or "median-mid80" (default). "median"
            and "mean" use the median/mean of the channel; "median-mid80"
            uses the median of the channel after removing the top and bottom
            10% of the sorted values.

        Returns
        -------
        Spectra
            This object (modified in-place).

        Notes
        -----
        Only channels that are *entirely* masked (all True in the
        corresponding row of `mask`) are replaced; a per-channel mask value
        is computed for all channels but applied only to fully-masked ones.
        """
        assert self.data.shape == mask.shape
        maskvals = np.ones(self.numchans)
        for ii in range(self.numchans):
            chan = self.get_chan(ii)
            # Use 'chan[:]' so update happens in-place
            if maskval == "mean":
                maskvals[ii] = np.mean(chan)
            elif maskval == "median":
                maskvals[ii] = np.median(chan)
            elif maskval == "median-mid80":
                n = int(np.round(0.1 * self.numspectra))
                maskvals[ii] = np.median(sorted(chan)[n:-n])
            else:
                maskvals[ii] = maskval
            if np.all(mask[ii]):
                self.data[ii] = (
                    np.ones_like(self.data[ii]) * (maskvals[:, np.newaxis][ii])
                )
        return self

    def dedisperse(self, dm: float = 0, padval: float | str = 0) -> None:
        """
        Shift channels according to the delays predicted by a DM.

        Dedispersion happens in-place.

        Parameters
        ----------
        dm : float, optional
            The DM in pc/cm^3 to use (default 0).
        padval : float or str, optional
            The padding value to use when shifting channels during
            dedispersion (default 0). See :meth:`shift_channels`.

        Returns
        -------
        None
        """
        assert dm >= 0
        ref_delay = psr_utils.delay_from_DM(dm - self.dm, np.max(self.freqs))
        delays = psr_utils.delay_from_DM(dm - self.dm, self.freqs)
        rel_delays = delays - ref_delay  # Relative delay
        rel_bindelays = np.round(rel_delays / self.dt).astype("int")
        # Shift channels
        self.shift_channels(rel_bindelays, padval)

        self.dm = dm

    def smooth(self, width: int = 1, padval: float | str = 0) -> None:
        """
        Smooth each channel by convolving with a top-hat kernel.

        The height of the top-hat is chosen such that RMS=1 after smoothing.
        Smoothing is done in-place.

        Parameters
        ----------
        width : int, optional
            The number of bins to smooth by (default 1, i.e. no smoothing).
        padval : float or str, optional
            The padding value to use. Possible values are a numeric value,
            "mean", "median", or "wrap" (default 0).

        Returns
        -------
        None

        Notes
        -----
        This bit of code is taken from Scott Ransom's PRESTO
        ``single_pulse_search.py`` (line ~423).
        """
        if width > 1:
            kernel = np.ones(width, dtype="float32") / np.sqrt(width)
            for ii in range(self.numchans):
                chan = self.get_chan(ii)
                if padval == "wrap":
                    tosmooth = np.concatenate([chan[-width:], chan, chan[:width]])
                elif padval == "mean":
                    tosmooth = np.ones(self.numspectra + width * 2) * np.mean(chan)
                    tosmooth[width:-width] = chan
                elif padval == "median":
                    tosmooth = np.ones(self.numspectra + width * 2) * np.median(chan)
                    tosmooth[width:-width] = chan
                else:  # padval is a float
                    tosmooth = np.ones(self.numspectra + width * 2) * padval
                    tosmooth[width:-width] = chan

                smoothed = scipy.signal.convolve(tosmooth, kernel, "same")
                chan[:] = smoothed[width:-width]

    def trim(self, bins: int = 0) -> None:
        """
        Trim spectra off the end (or beginning) of the data, in-place.

        Trimming is irreversible.

        Parameters
        ----------
        bins : int, optional
            The number of spectra to trim off the end of the observation
            (default 0). If `bins` is negative, spectra are trimmed off the
            beginning instead.

        Returns
        -------
        None
        """
        assert bins < self.numspectra
        if bins == 0:
            return
        elif bins > 0:
            self.data = self.data[:, :-bins]
            self.numspectra = self.numspectra - bins
        elif bins < 0:
            self.data = self.data[:, bins:]
            self.numspectra = self.numspectra - bins
            self.starttime = self.starttime + bins * self.dt

    def downsample(self, factor: int = 1, trim: bool = True) -> None:
        """
        Downsample the spectra by co-adding adjacent bins, in-place.

        Parameters
        ----------
        factor : int, optional
            Reduce the number of spectra by this factor (default 1). Must be
            a factor of the number of spectra if `trim` is False.
        trim : bool, optional
            If True, trim off excess bins (default True).

        Returns
        -------
        None
        """
        assert trim or not (self.numspectra % factor)
        new_num_spectra = self.numspectra // factor
        num_to_trim = self.numspectra % factor
        self.trim(num_to_trim)
        self.data = np.array(
            np.column_stack(
                [
                    np.sum(subint, axis=1)
                    for subint in np.hsplit(self.data, new_num_spectra)
                ]
            )
        )
        self.numspectra = new_num_spectra
        self.dt = self.dt * factor

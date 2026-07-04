"""
A module for reading filterbank files.

Patrick Lazarus, June 26, 2012
(Minor modification from file originally from June 6th, 2009)
"""

from __future__ import annotations

import sys
import os
import os.path

import numpy as np

from presto import sigproc
from presto import spectra


DEBUG = False


def create_filterbank_file(
    outfn: str,
    header: dict,
    spectra: np.ndarray | None = None,
    nbits: int = 8,
    verbose: bool = False,
    mode: str = "append",
) -> FilterbankFile:
    """
    Write a filterbank header and spectra to a file.

    Parameters
    ----------
    outfn : str
        The output filterbank file's name.
    header : dict
        A dictionary of header parameters and values.
    spectra : numpy.ndarray, optional
        Spectra to write to the file. If None (default), only the header is
        written.
    nbits : int, optional
        The number of bits per sample of the filterbank file (default 8, so
        each sample is an 8-bit integer). This value always overrides the
        value in the `header` dictionary.
    verbose : bool, optional
        If True, be verbose (default False).
    mode : str, optional
        The mode for writing, either "append" or "write" (default "append").

    Returns
    -------
    FilterbankFile
        The resulting FilterbankFile object, opened in read-write mode.
    """
    dtype = get_dtype(nbits)  # Get dtype. This will check to ensure
    # 'nbits' is valid.
    header["nbits"] = nbits
    outfile = open(outfn, "wb")
    outfile.write(sigproc.addto_hdr("HEADER_START", None))
    for paramname in list(header.keys()):
        if paramname not in sigproc.header_params:
            # Only add recognized parameters
            continue
        if verbose:
            print("Writing header param (%s)" % paramname)
        value = header[paramname]
        outfile.write(sigproc.addto_hdr(paramname, value))
    outfile.write(sigproc.addto_hdr("HEADER_END", None))
    if spectra is not None:
        spectra.flatten().astype(dtype).tofile(outfile)
    outfile.close()
    return FilterbankFile(outfn, mode=mode)


def is_float(nbits: int) -> bool:
    """
    Return whether a number of bits per sample corresponds to floats.

    Parameters
    ----------
    nbits : int
        The number of bits per sample, as recorded in the filterbank file's
        header.

    Returns
    -------
    bool
        True if `nbits` indicates the data are encoded as floats.
    """
    check_nbits(nbits)
    if nbits == 32:
        return True
    else:
        return False


def check_nbits(nbits: int) -> None:
    """
    Check that a number of bits per sample is supported.

    Parameters
    ----------
    nbits : int
        The number of bits per sample, as recorded in the filterbank file's
        header.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``filterbank.py`` cannot cope with the given `nbits` (only 8- and
        16-bit integers and 32-bit floats are supported).
    """
    if nbits not in [32, 16, 8]:
        raise ValueError(
            "'filterbank.py' only supports "
            "files with 8- or 16-bit "
            "integers, or 32-bit floats "
            "(nbits provided: %g)!" % nbits
        )


def get_dtype(nbits: int) -> str:
    """
    Return a numpy dtype string for a number of bits per sample.

    Parameters
    ----------
    nbits : int
        The number of bits per sample, as recorded in the filterbank file's
        header.

    Returns
    -------
    str
        A numpy-recognized dtype string (e.g. ``"uint8"`` or ``"float32"``).
    """
    check_nbits(nbits)
    if is_float(nbits):
        dtype = "float%d" % nbits
    else:
        dtype = "uint%d" % nbits
    return dtype


def read_header(filename: str, verbose: bool = False) -> tuple[dict, int]:
    """
    Read the header of a filterbank file.

    Parameters
    ----------
    filename : str
        The name of the filterbank file.
    verbose : bool, optional
        If True, be verbose (default False).

    Returns
    -------
    header : dict
        A dictionary of header parameters.
    header_size : int
        The size of the header in bytes.
    """
    header = {}
    filfile = open(filename, "rb")
    filfile.seek(0)
    paramname = ""
    while paramname != "HEADER_END":
        if verbose:
            print("File location: %d" % filfile.tell())
        paramname, val = sigproc.read_hdr_val(filfile, stdout=verbose)
        if verbose:
            print("Read param %s (value: %s)" % (paramname, val))
        if paramname not in ["HEADER_START", "HEADER_END"]:
            header[paramname] = val
    header_size = filfile.tell()
    filfile.close()
    return header, header_size


class FilterbankFile(object):
    """
    A SIGPROC filterbank file.

    Header parameters are accessible as attributes via ``__getattr__``
    (e.g. ``fch1``, ``foff``, ``nchans``, ``nbits``, ``tsamp``), in addition
    to the derived attributes documented below.

    Parameters
    ----------
    filfn : str
        The path to the filterbank file.
    mode : str, optional
        How to open the file: "read"/"readonly", "write"/"readwrite", or
        "append" (default "readonly").

    Attributes
    ----------
    filename : str
        The path to the filterbank file.
    header : dict
        The parsed header parameters.
    header_size : int
        The size of the header in bytes.
    frequencies : numpy.ndarray
        The channel center frequencies (MHz).
    is_hifreq_first : bool
        True if the highest frequency channel is first (``foff < 0``).
    bytes_per_spectrum : int
        The number of bytes in a single spectrum.
    nspec : int
        The number of spectra in the file.
    isfold : bool
        True if the file is a folded-filterbank file.
    dt : float
        The time per spectrum in sec (the sample time, or period/nbins for a
        folded file).
    dtype : str
        The numpy dtype string of the samples.
    dtype_min, dtype_max : int or float
        The minimum and maximum representable sample values.

    Raises
    ------
    ValueError
        If the file does not exist, or `mode` is unrecognized.
    """

    def __init__(self, filfn: str, mode: str = "readonly"):
        self.filename = filfn
        self.filfile = None
        if not os.path.isfile(filfn):
            raise ValueError("ERROR: File does not exist!\n\t(%s)" % filfn)
        self.header, self.header_size = read_header(self.filename)
        self.frequencies = self.fch1 + self.foff * np.arange(self.nchans)
        self.is_hifreq_first = self.foff < 0
        self.bytes_per_spectrum = self.nchans * self.nbits // 8
        data_size = os.path.getsize(self.filename) - self.header_size
        self.nspec = data_size // self.bytes_per_spectrum

        # Check if this file is a folded-filterbank file
        if (
            "npuls" in self.header
            and "period" in self.header
            and "nbins" in self.header
            and "tsamp" not in self.header
        ):
            # Foleded file
            self.isfold = True
            self.dt = self.period / self.nbins
        else:
            self.isfold = False
            self.dt = self.tsamp

        # Get info about dtype
        self.dtype = get_dtype(self.nbits)
        if is_float(self.nbits):
            tinfo = np.finfo(self.dtype)
        else:
            tinfo = np.iinfo(self.dtype)
        self.dtype_min = tinfo.min
        self.dtype_max = tinfo.max

        if mode.lower() in ("read", "readonly"):
            self.filfile = open(self.filename, "rb")
        elif mode.lower() in ("write", "readwrite"):
            self.filfile = open(self.filename, "r+b")
        elif mode.lower() == "append":
            self.filfile = open(self.filename, "a+b")
        else:
            raise ValueError("Unrecognized mode (%s)!" % mode)

    @property
    def freqs(self) -> np.ndarray:
        # Alias for frequencies
        return self.frequencies

    @property
    def nchan(self) -> int:
        # more aliases..
        return self.nchans

    def close(self) -> None:
        """Close the underlying file if it is open."""
        if self.filfile is not None:
            self.filfile.close()

    def get_timeslice(self, start: float, stop: float) -> spectra.Spectra:
        """
        Return the spectra between two times.

        Parameters
        ----------
        start : float
            The start time in sec.
        stop : float
            The stop time in sec.

        Returns
        -------
        spectra.Spectra
            The spectra covering the requested time slice.
        """
        startspec = int(np.round(start / self.tsamp))
        stopspec = int(np.round(stop / self.tsamp))
        return self.get_spectra(startspec, stopspec - startspec)

    def get_spectra(self, start: int, nspec: int) -> spectra.Spectra:
        """
        Return a block of spectra from the file.

        Parameters
        ----------
        start : int
            The index of the first spectrum to read.
        nspec : int
            The number of spectra to read (clipped to the end of the file).

        Returns
        -------
        spectra.Spectra
            The requested spectra.
        """
        stop = min(start + nspec, self.nspec)
        pos = self.header_size + start * self.bytes_per_spectrum
        # Compute number of elements to read
        nspec = int(stop) - int(start)
        num_to_read = nspec * self.nchans
        num_to_read = max(0, num_to_read)
        self.filfile.seek(pos, os.SEEK_SET)
        spectra_dat = np.fromfile(self.filfile, dtype=self.dtype, count=num_to_read)
        spectra_dat = np.reshape(spectra_dat, (nspec, self.nchans))
        spec = spectra.Spectra(
            self.freqs,
            self.tsamp,
            spectra_dat.T,
            starttime=start * self.tsamp,
            dm=0.0,
        )
        return spec

    def append_spectra(self, spectra: np.ndarray) -> None:
        """
        Append spectra to the file if it is not read-only.

        Parameters
        ----------
        spectra : numpy.ndarray
            The spectra to append. The new spectra must have the correct
            number of channels (i.e. the dimension of axis=1).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the file is read-only, or the spectra have the wrong number
            of channels.
        """
        if self.filfile.mode.lower() in ("r", "rb"):
            raise ValueError(
                "FilterbankFile object for '%s' is read-only." % self.filename
            )
        nspec, nchans = spectra.shape
        if nchans != self.nchans:
            raise ValueError(
                "Cannot append spectra. Incorrect shape. "
                "Number of channels in file: %d; Number of "
                "channels in spectra to append: %d" % (self.nchans, nchans)
            )
        data = spectra.flatten()
        np.clip(data, self.dtype_min, self.dtype_max, out=data)
        # Move to end of file
        self.filfile.seek(0, os.SEEK_END)
        self.filfile.write(data.astype(self.dtype))
        self.nspec += nspec
        # self.filfile.flush()
        # os.fsync(self.filfile)

    def write_spectra(self, spectra: np.ndarray, ispec: int) -> None:
        """
        Write spectra to the file if it is writable.

        Parameters
        ----------
        spectra : numpy.ndarray
            The spectra to write. The new spectra must have the correct
            number of channels (i.e. the dimension of axis=1).
        ispec : int
            The index of the spectrum at which to start writing.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the file is not writable, the spectra have the wrong number
            of channels, or the write would go past the end of the file.
        """
        if "r+" not in self.filfile.mode.lower():
            raise ValueError(
                "FilterbankFile object for '%s' is not writable." % self.filename
            )
        nspec, nchans = spectra.shape
        if nchans != self.nchans:
            raise ValueError(
                "Cannot write spectra. Incorrect shape. "
                "Number of channels in file: %d; Number of "
                "channels in spectra to write: %d" % (self.nchans, nchans)
            )
        if ispec > self.nspec:
            raise ValueError(
                "Cannot write past end of file! "
                "Present number of spectra: %d; "
                "Requested index of write: %d" % (self.nspec, ispec)
            )
        data = spectra.flatten()
        np.clip(data, self.dtype_min, self.dtype_max, out=data)
        # Move to requested position
        pos = self.header_size + ispec * self.bytes_per_spectrum
        self.filfile.seek(pos, os.SEEK_SET)
        self.filfile.write(data.astype(self.dtype))
        if nspec + ispec > self.nspec:
            self.nspec = nspec + ispec

    def __getattr__(self, name: str):
        if name in self.header:
            if DEBUG:
                print("Fetching header param (%s)" % name)
            val = self.header[name]
        else:
            raise ValueError("No FilterbankFile attribute called '%s'" % name)
        return val

    def print_header(self) -> None:
        """Print the header parameters and values."""
        for param in sorted(self.header.keys()):
            if param in ("HEADER_START", "HEADER_END"):
                continue
            print("%s: %s" % (param, self.header[param]))


def main() -> None:
    fil = FilterbankFile(sys.argv[1])
    fil.print_header()


if __name__ == "__main__":
    main()

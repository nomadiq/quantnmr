import numpy as np
import scipy.signal as signal
from scipy.optimize.minpack import curve_fit
from scipy.fftpack import fftshift, fft
import math

# from pathlib import Path
import os.path

# from numba import jit, njit, prange
import colorama

# ------------------------------------- #
#              Variables                #
# ------------------------------------- #


# ------------------------------------- #
#              Functions                #
# ------------------------------------- #


def get_aq_mod(acq):

    if isinstance(acq, int):
        aq_mod_dict = {
            0: "qf",
            1: "qsim",
            2: "qseq",
            3: "DQD",
            4: "parallelQsim",
            5: "parallelDQD",
        }

        return aq_mod_dict[acq]
    else:
        return None


def get_fn_mod(acq):

    if isinstance(acq, int):
        fn_mod_dict = {
            0: "undefined",
            1: "QF",
            2: "QSEQ",
            3: "TPPI",
            4: "States",
            5: "States-TPPI",
            6: "Echo-Antiecho",
        }

        return fn_mod_dict[acq]
    else:
        return None


def make_complex(data):

    return data[..., ::2] + data[..., 1::2] * 1.0j


def next_fourier_number(num):

    return math.ceil(math.log(num, 2))


def remove_bruker_delay(data, grpdly):
    n = len(data)
    integer_shift = np.floor(grpdly)
    fract_shift = grpdly - integer_shift
    data = np.roll(data, -int(integer_shift))
    data = nmr_fft(data) / n
    data = data * np.exp(2.0j * np.pi * (fract_shift) * np.arange(n) / n)
    data = i_nmr_fft(data) * n
    return data


def nmr_fft(data):
    return np.fft.fftshift(np.fft.fft(data))


def i_nmr_fft(data):
    return np.fft.ifft(np.fft.ifftshift(data))


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    return signal.filtfilt(b, a, data)


def dd2g(dspfvs, decim):
    dspdic = {
        10: {
            2: 44.75,
            3: 33.5,
            4: 66.625,
            6: 59.083333333333333,
            8: 68.5625,
            12: 60.375,
            16: 69.53125,
            24: 61.020833333333333,
            32: 70.015625,
            48: 61.34375,
            64: 70.2578125,
            96: 61.505208333333333,
            128: 70.37890625,
            192: 61.5859375,
            256: 70.439453125,
            384: 61.626302083333333,
            512: 70.4697265625,
            768: 61.646484375,
            1024: 70.48486328125,
            1536: 61.656575520833333,
            2048: 70.492431640625,
        },
        11: {
            2: 46.0,
            3: 36.5,
            4: 48.0,
            6: 50.166666666666667,
            8: 53.25,
            12: 69.5,
            16: 72.25,
            24: 70.166666666666667,
            32: 72.75,
            48: 70.5,
            64: 73.0,
            96: 70.666666666666667,
            128: 72.5,
            192: 71.333333333333333,
            256: 72.25,
            384: 71.666666666666667,
            512: 72.125,
            768: 71.833333333333333,
            1024: 72.0625,
            1536: 71.916666666666667,
            2048: 72.03125,
        },
        12: {
            2: 46.0,
            3: 36.5,
            4: 48.0,
            6: 50.166666666666667,
            8: 53.25,
            12: 69.5,
            16: 71.625,
            24: 70.166666666666667,
            32: 72.125,
            48: 70.5,
            64: 72.375,
            96: 70.666666666666667,
            128: 72.5,
            192: 71.333333333333333,
            256: 72.25,
            384: 71.666666666666667,
            512: 72.125,
            768: 71.833333333333333,
            1024: 72.0625,
            1536: 71.916666666666667,
            2048: 72.03125,
        },
        13: {
            2: 2.75,
            3: 2.8333333333333333,
            4: 2.875,
            6: 2.9166666666666667,
            8: 2.9375,
            12: 2.9583333333333333,
            16: 2.96875,
            24: 2.9791666666666667,
            32: 2.984375,
            48: 2.9895833333333333,
            64: 2.9921875,
            96: 2.9947916666666667,
        },
    }
    return dspdic[dspfvs][decim]


def window_function(points=0, window="sb", window_p=0.5) -> np.ndarray:
    if window == "sb":
        return np.sin(
            (
                window_p * math.pi
                + (0.99 - window_p) * math.pi * np.arange(points) / points
            )
        )
    else:
        return np.ones(points)


# ------------------------------------- #
#               Classes                 #
# ------------------------------------- #


class Bruker1D:
    """
    Bruker1D:

    This class reads and holds Bruker 1D unprocessed data as well as processed data (if wanted).

    Usage Examples:

    First, import the module

    import mauanakini as mk

    1) Simple Reading of data from a directory

        my_bruker_data = mk.Bruker1D(data_dir='/path/to/data/')

        N.B. this assumes the file name for the 1D data is 'fid' which is usually the case for 1D data.
        If this is not the case, you can change the file name for the data to be read:

        my_bruker_data = mk.Bruker1D(data_dir='/path/to/data/', ser_file='serial_file_name')

        The raw data will be loaded and is available as a numpy array as

        my_bruker_data.raw_data

        This array is composed of a real value followed by an imaginary value (1st point) and so on, as
        4 byte integers, e.g. ririririririririririririririririririririririri

        The length of this array is

        my_bruker_data.td

        If this array has length % 2 == 0 (even number of points in array) it will be converted to a complex
        data array as well and is available as a complex numpy array as

        my_bruker_data.raw_data_complex

        The length of this array is stored in

        my_bruker_data.points

        By default, this class attempts to find the parameters for the Bruker digital filter delay from the acqus
        file in the data directory. These parameters, if found are available at

        my_bruker_data.decim
        my_bruker_data.dspfvs
        my_bruker_data.grpdly

        If not found, they are set to zero (0)

        If decim and dspfvs are found, but grpdly is not, grpdly is determined from a lookup dictionary

        If a complex data array is constructed and a grpdly can be determined, the class/object will automatically
        attempt to remove the Bruker digital filter delay using the grpdly value. The result is available at

        my_bruker_data.converted_data

        Details for how this conversion is done can be found in

        mk.remove_bruker_delay.__doc__

        Alternatviely, decim, dspfvs or grpdly can be manually set (in cases where they may not be recorded for some
        reason in the data directory). To do so, pass in their values when constructing the Data1D object from a Bruker
        data directory:

        my_bruker_data = mk.Bruker1D(data_dir='/path/to/data/', decim=123, dspfvs=456, grpdly=71.2345)

        If you do this, it will just use the grpdly value provided for filter delay removal. So it is just easier to do:

        my_bruker_data = mk.Bruker1D(data_dir='/path/to/data/', grpdly=71.2345)

        Constructing an object from the class can be done with two verbose levels:

        verbose=0 --> Will generally only tell you things went wrong. Maybe were as well
        verbose=1 --> Will tell you everything it is doing along the way and will give you reasons for all errors

        e.g.

        my_bruker_data = mk.Bruker1D(data_dir='/path/to/data/', verbose=1)

        TODO:

        Functions for custom processing of converted_data, including:

        zero fill
        window functions / first point correction
        baseline correction
        low frequency (solvent) suppression
        phasing

        Functions to read relevant frequency information. SW, Dwell, Carrier Frequency etc

    2) Reading in processed data as well from a processing (pdata) directory

        If processed data is available it can read in. Usually this data exists in the data directory under:

        /pdata/N/

        Where N is an integer and the number for the processed data. You can only read in one processed data set for now.
        To read in the processed data at '1' (/path/to/data/pdata/1/) as well as the raw data, you would do the following

        my_bruker_data = mk.Bruker1D(data_dir='/path/to/data/', processed_dir=1)

        The processed data will be available at:

        my_bruker_data.bruker_proc_data

        as a complex numpy array.
    """

    def __init__(
        self,
        data_dir=".",
        ser_file="fid",
        processed_dir=None,
        decim=None,
        dspfvs=None,
        grpdly=None,
        verbose=0,
    ):  # sourcery no-metrics

        # create the file names for where the data/params are
        self.acqu = os.path.join(data_dir, "acqu")
        self.acqus = os.path.join(data_dir, "acqus")
        self.ser = os.path.join(data_dir, ser_file)
        self.pp = os.path.join(data_dir, "pulseprogram")
        self.dir = data_dir
        self.proc_dir = processed_dir

        # check for presence of processed data in directory
        if isinstance(processed_dir, int):
            if processed_dir:
                self.proc_dir = data_dir + "/pdata/" + str(processed_dir) + "/"
                if verbose:
                    print("Importing Processed Data")
        else:
            self.proc_dir = 0
            if verbose:
                print("Not importing Any Processed Data")

        self.converted_data = 0
        self.verbose = verbose

        # check if we are a Bruker 1D data set
        if (
            os.path.isfile(self.acqus)
            and os.path.isfile(self.acqu)
            and os.path.isfile(self.ser)
            and os.path.isfile(self.pp)
        ):

            self.valid = True

            dec = dsp = grp = 0  # we'll find these in the files
            with open(self.acqus, "r") as acqusfile:
                for line in acqusfile:
                    if "##$TD=" in line:
                        (_, value) = line.split()
                        td = int(value)
                    if "##$DECIM=" in line:
                        (_, value) = line.split()
                        dec = int(value)
                    if "##$DSPFVS=" in line:
                        (_, value) = line.split()
                        dsp = int(value)
                    if "##$GRPDLY=" in line:
                        (_, value) = line.split()
                        grp = float(value)
                    if "##$BYTORDA=" in line:
                        (_, value) = line.split()
                        self.byte_order = float(value)
                    if "##$AQ_mod=" in line:
                        (_, value) = line.split()
                        self.aq_mod = get_aq_mod(int(value))
                    if "##$DTYPA=" in line:
                        (_, value) = line.split()
                        self.dtypa = int(value)

            self.td = td  # number of points

            if dec != 0:
                self.decim = dec
            if dsp != 0:
                self.dspfvs = dsp
            if grp:
                self.grpdly = grp
            elif dec != 0 and dsp != 0:
                self.grpdly = dd2g(dsp, dec)
            else:
                if self.verbose:
                    print(
                        "problem with detecting / determining grpdly - needed for Bruker conversion"
                    )
                self.valid = False

            if self.verbose:
                print("Data Points structure is: " + str(td))
                print(
                    "DECIM= "
                    + str(self.decim)
                    + " DSPFVS= "
                    + str(self.dspfvs)
                    + " GRPDLY= "
                    + str(self.grpdly)
                )

        else:
            self.valid = False
            print("Data Directory does not seem to contain Bruker 1D Data")

        if self.valid:
            self.load_serial_file()

        # do we have bruker processed data to load?
        if self.proc_dir:

            procsfile = open(self.proc_dir + "/procs")
            for line in procsfile:
                if "##$FTSIZE=" in line:
                    (_, value) = line.split()
                    self.proc_ft_size = int(value)
                if "##$PHC0=" in line:
                    (_, value) = line.split()
                    self.proc_ph0 = float(value)
                if "##$PHC1=" in line:
                    (_, value) = line.split()
                    self.proc_ph1 = float(value)
                if "##$BYTORDP=" in line:
                    (_, value) = line.split()
                    self.proc_byte_order = float(value)

            self.bruker_proc_data = np.zeros(self.proc_ft_size, dtype="complex128")

            self.load_bruker_proc()

    # init ends

    def load_serial_file(self):

        with open(self.ser, "rb") as serial_file:
            if self.dtypa == 0:
                dtype_string = "i4"
            elif self.dtypa == 2:
                dtype_string = "i8"

            if self.byte_order == 0:
                self.raw_data = np.frombuffer(
                    serial_file.read(), dtype="<" + dtype_string
                )

            elif self.byte_order == 1:
                self.raw_data = np.frombuffer(
                    serial_file.read(), dtype=">" + dtype_string
                )

            if len(self.raw_data) % 2 == 0:
                self.raw_data_complex = make_complex(self.raw_data)
            else:
                print(
                    "Raw data does not have even number of points, can not make complex"
                )
                self.valid = False

        self.converted_data = np.zeros(int(self.td / 2), dtype="complex128")

        # lets convert the data
        if self.decim and self.dspfvs:
            if not self.grpdly:
                self.grpdly = dd2g(self.dspfvs, self.decim)
            # self.grpdly = np.floor(self.grpdly)
            self.convert_bruker_1d()
        elif self.grpdly and not self.decim and not self.dspfvs:
            self.convert_bruker_1d()

        else:
            print(
                "Could not convert from Bruker data, incorrect or not found grpdly, dspfvs and/or decim"
            )
            self.valid = False

        if self.verbose:
            print("Converted Data Points structure is:", self.td)

        serial_file.close()

    def convert_bruker_1d(self):

        # edit the number of points in first dimension after Bruker filter removal
        # we now count points in complex numbers as well

        try:
            # self.converted_data = remove_bruker_filter(make_complex(self.raw_data), self.grpdly)
            self.converted_data = remove_bruker_delay(
                make_complex(self.raw_data), self.grpdly
            )
            self.points = len(self.converted_data)
            if self.verbose:
                print(f"After Bruker Delay Filter removal we have {self.points} points")
        except Exception as e:
            print(e)
            print("Could not convert bruker ser/fid file")
            self.valid = False

    def load_bruker_proc(self):

        # loading the 1r and 1i files
        with open(self.proc_dir + "/1r", "rb") as real_proc:
            if self.proc_byte_order == 0:
                try:
                    real = np.frombuffer(real_proc.read(), dtype="<i4")
                except:

                    real = np.asarray([0])
            elif self.proc_byte_order == 1:
                try:
                    real = np.frombuffer(real_proc.read(), dtype=">i4")
                except:

                    real = np.asarray([0])

        with open(self.proc_dir + "/1i", "rb") as imag_proc:
            if self.proc_byte_order == 0:
                try:
                    imag = np.frombuffer(imag_proc.read(), dtype="<i4")
                except:
                    imag = np.asarray([0])
            elif self.proc_byte_order == 1:
                try:
                    imag = np.frombuffer(imag_proc.read(), dtype=">i4")
                except:

                    imag = np.asarray([0])

        if len(real) == 1 or len(imag) == 1:
            self.bruker_proc_data = None
            print("Could not read Bruker Processed Data")
            self.valid = False
        else:
            # store the data
            self.bruker_proc_data = real + 1.0j * imag


class LINData2D:
    def __init__(
        self,
        data_dir=".",
        ser_file="ser",
        points=None,
        dim_status=None,
        decim=None,
        dspfvs=None,
        grpdly=None,
    ):  # sourcery no-metrics

        self.ac1 = os.path.join(data_dir, "acqus")
        self.ac2 = os.path.join(data_dir, "acqu2s")
        self.ser = os.path.join(data_dir, "ser")
        self.pp = os.path.join(data_dir, "pulseprogram")
        self.ser = os.path.join(data_dir, ser_file)
        self.dir = data_dir
        self.acq = [0, 0]  # acquisition modes start as undefined

        # dictionary of acquisition modes for Bruker
        self.acqDict = {
            0: "undefined",
            1: "qf",
            2: "qsec",
            3: "tppi",
            4: "states",
            5: "states-tppi",
            6: "echo-antiecho",
        }

        # check if we are a Bruker 2D data set
        if (
            os.path.isfile(self.ac1)
            and os.path.isfile(self.ac2)
            and os.path.isfile(self.ser)
            and os.path.isfile(self.pp)
        ):
            self.valid = True

        else:
            self.valid = False
            print("Data Directory does not seem to contain Bruker 2D Data")

        p0 = p1 = 0  # we'll find these in the files
        dec = dsp = grp = 0  # we'll find these in the files

        with open(self.ac1, "r") as acqusfile:
            for line in acqusfile:
                if "##$TD=" in line:
                    (_, value) = line.split()
                    p0 = int(value)
                if "##$DECIM=" in line:
                    (_, value) = line.split()
                    dec = int(value)
                if "##$DSPFVS=" in line:
                    (_, value) = line.split()
                    dsp = int(value)
                if "##$GRPDLY=" in line:
                    (_, value) = line.split()
                    grp = float(value)

                if "##$BYTORDA=" in line:
                    (_, value) = line.split()
                    self.byte_order = float(value)

                self.acq[0] = 0  # doesnt matter we assume DQD for direct anyway

        with open(self.ac2, "r") as acqusfile:
            for line in acqusfile:
                if "##$TD=" in line:
                    (_, value) = line.split()
                    p1 = int(value)

                if "##$FnMODE=" in line:
                    (_, value) = line.split()
                    self.acq[1] = int(value)

        if p0 and p1:
            points = [p0, p1]
        else:
            print("problem with detecting number of points in data")
            self.valid = False

        if dec != 0:
            decim = dec
        if dsp != 0:
            dspfvs = dsp
        if grp:
            grpdly = grp
        elif dec != 0 and dsp != 0:
            grpdly = dd2g(dspfvs, decim)
        else:
            print(
                "problem with detecting / determining grpdly - needed for Bruker conversion"
            )
            self.valid = False

        print("Data Points structure is: " + str(points))
        print(
            "DECIM= "
            + str(decim)
            + " DSPFVS= "
            + str(dspfvs)
            + " GRPDLY= "
            + str(grpdly)
        )

        self.dim_status = ["t", "t"] if dim_status is None else dim_status
        if dim_status:
            if len(dim_status) != len(points):
                raise ValueError(
                    "insanity: number of dimensions in 'points' and 'dim_status' don't match"
                )
            for i in range(len(dim_status)):
                if dim_status[i] not in ["t", "f"]:
                    print(dim_status[i])
                    raise ValueError(
                        "dimension domains must be 'f' - frequency or 't' - time"
                    )

        # lets store the points
        self.points = points

        # now lets load in the bruker serial file
        with open(self.ser, "rb") as serial_file:
            if self.byte_order == 0:
                self.raw_data = np.frombuffer(serial_file.read(), dtype="<i4")
            elif self.byte_order == 1:
                self.raw_data = np.frombuffer(serial_file.read(), dtype=">i4")

        # now reshape the data
        self.raw_data = np.reshape(self.raw_data, np.asarray(self.points), order="F")

        # TODO - set up some sort of sanity test

        self.converted_data = np.zeros(
            (int(self.points[0] / 2), self.points[1]), dtype="complex128"
        )

        # lets convert the data
        if decim and dspfvs:
            if not grpdly:
                grpdly = dd2g(dspfvs, decim)
            self.convert_bruker_2d(grpdly)
        elif grpdly and not decim and not dspfvs:
            self.convert_bruker_2d(grpdly)

        else:
            print(
                "Could not convert from Bruker data, incorrect or not found grpdly, dspfvs and/or decim"
            )

        print("Converted Data Points structure is:", self.points)

        self.phases = (0, 0)
        self.fp_corrections = (0.5, 0.5)
        self.windows = ("sb", "sb")
        self.windows_p = (0.5, 0.5)
        self.zero_fill = (1.0, 1.0)

        self.processed_data = []  # this will be filled out in proc method
        self.ft_points = []

    def convert_bruker_2d(self, grpdly):

        # edit the number of points in first dimension after Bruker filter removal
        # we now count points in complex numbers as well
        self.points[0] = len(
            remove_bruker_delay(make_complex(self.raw_data[:, 0]), grpdly)
        )

        # convert the data
        for i in range(
            self.points[1]
        ):  # inner loop for second dimension points from dataFID
            fid = remove_bruker_delay(make_complex(self.raw_data[:, i]), grpdly)
            self.converted_data[0 : len(fid), i] = fid

        self.converted_data = self.converted_data[
            0 : self.points[0],
            0 : self.points[1],
        ]

        if self.acq[1] == 6:  # Rance Kay Processing needed
            print("Echo-AntiEcho Detected in T1 - dealing with it...")
            for i in range(0, self.points[1], 2):
                a = self.converted_data[:, i]
                b = self.converted_data[:, i + 1]
                c = a + b
                d = a - b
                self.converted_data[:, i] = c * np.exp(1.0j * (90 / 180) * np.pi)
                self.converted_data[:, i + 1] = d * np.exp(1.0j * (180 / 180) * np.pi)

        self.raw_data = self.converted_data  # clean up memory a little

    def proc_t2(self, t2_ss=None, phase=0, c=1.0, window="sb", window_p=0.5):

        self.processed_data[0, :] = self.processed_data[0, :] * c

        for i in range(self.ft_points[1]):
            fid = self.processed_data[:, i]

            if t2_ss == "butter":
                fid = butter_highpass_filter(fid, 0.01, 0.1, order=1)

            elif t2_ss == "poly":
                co_ef = np.polynomial.polynomial.polyfit(np.arange(len(fid)), fid, 5)
                time_points = np.arange(len(fid))
                polyline = sum(
                    co_ef[iii] * time_points ** iii for iii in range(len(co_ef))
                )
                fid = fid - polyline

            fid = fid * window_function(
                points=len(fid),
                window=window,
                window_p=window_p,
            )
            self.processed_data[0 : len(fid), i] = fid
            self.processed_data[:, i] = np.fft.fftshift(
                np.fft.fft(
                    self.processed_data[:, i] * np.exp(1.0j * (phase / 180) * np.pi)
                )
            )[::-1]

    def proc_t1(self, phase=0, c=1.0, window="sb", window_p=0.5):

        self.processed_data[:, 0] = self.processed_data[:, 0] * c
        self.processed_data[:, 1] = self.processed_data[:, 1] * c

        for i in range(self.ft_points[0]):
            fid_r = np.real(self.processed_data[i, ::2])
            fid_i = np.real(self.processed_data[i, 1::2])
            fid = np.ravel((fid_r, fid_i), order="F")
            fid = make_complex(fid)
            fid = fid * window_function(
                points=len(fid), window=window, window_p=window_p
            )

            self.processed_data[i, 0 : len(fid)] = fid
            self.processed_data[i, len(fid) :] = np.zeros(self.ft_points[1] - len(fid))

            if self.acq[1] != 5:
                self.processed_data[i, :] = np.fft.fftshift(
                    np.fft.fft(
                        self.processed_data[i, :] * np.exp(1.0j * (phase / 180) * np.pi)
                    )
                )[::-1]

            else:
                self.processed_data[i, :] = np.fft.fft(
                    self.processed_data[i, :] * np.exp(1.0j * (phase / 180) * np.pi)
                )[::-1]

    # hypercomplex processing with imaginary parts
    def proc_t1_ii(self, phase=0, c=1.0, window="sb", window_p=0.5):

        self.processed_data[:, 0] = self.processed_data[:, 0] * c
        self.processed_data[:, 1] = self.processed_data[:, 1] * c

        for i in range(self.ft_points[0]):
            fid_r = np.imag(self.processed_data[i, ::2])
            fid_i = np.imag(self.processed_data[i, 1::2])
            fid = np.ravel((fid_r, fid_i), order="F")
            fid = make_complex(fid)
            fid = fid * window_function(
                points=len(fid), window=window, window_p=window_p
            )

            self.processed_data[i, 0 : len(fid)] = fid
            self.processed_data[i, len(fid) :] = np.zeros(self.ft_points[1] - len(fid))

            if self.acq[1] != 5:
                self.processed_data[i, :] = np.fft.fftshift(
                    np.fft.fft(
                        self.processed_data[i, :] * np.exp(1.0j * (phase / 180) * np.pi)
                    )
                )[::-1]

            else:
                self.processed_data[i, :] = np.fft.fft(
                    self.processed_data[i, :] * np.exp(1.0j * (phase / 180) * np.pi)
                )[::-1]

    def proc(
        self,
        phases=(0, 0),
        t2_ss=None,
        fp_corrections=(0.5, 0.5),
        windows=("sb", "sb"),
        windows_p=(0.5, 0.5),
        zero_fill=(1.0, 1.0),
    ):

        t1_ac_mode = int(self.acq[1])
        if (
            t1_ac_mode >= 3 or t1_ac_mode <= 6
        ):  # hypercomplex data. T1 points is really half
            points_t2 = int(self.points[1] / 2)
        else:
            points_t2 = self.points[1]

        self.ft_points = (
            int(2 ** (next_fourier_number(self.points[0]) + zero_fill[0])),
            int(2 ** (next_fourier_number(points_t2) + zero_fill[1])),
        )
        print(self.ft_points)
        self.processed_data = np.zeros(self.ft_points, dtype="complex128")

        self.processed_data[
            0 : self.points[0], 0 : self.points[1]
        ] = self.converted_data

        self.proc_t2(
            t2_ss=t2_ss,
            phase=phases[0],
            c=fp_corrections[0],
            window=windows[0],
            window_p=windows_p[0],
        )

        self.proc_t1(
            phase=phases[1],
            c=fp_corrections[1],
            window=windows[1],
            window_p=windows_p[1],
        )

    # this is an test form of processing - should be deleted eventually
    def proc_ii(
        self,
        t2_ss=None,
        phases=(0, 0),
        fp_corrections=(0.5, 0.5),
        windows=("sb", "sb"),
        windows_p=(0.5, 0.5),
        zero_fill=(1.0, 1.0),
    ):

        t1_ac_mode = int(self.acq[1])
        if (
            t1_ac_mode >= 3 or t1_ac_mode <= 6
        ):  # hypercomplex data. T1 points is really half
            points_t2 = int(self.points[1] / 2)
        else:
            points_t2 = self.points[1]

        self.ft_points = (
            int(2 ** (next_fourier_number(self.points[0]) + zero_fill[0])),
            int(2 ** (next_fourier_number(points_t2) + zero_fill[1])),
        )
        print(self.ft_points)
        self.processed_data = np.zeros(self.ft_points, dtype="complex128")

        self.processed_data[
            0 : self.points[0], 0 : self.points[1]
        ] = self.converted_data

        self.proc_t2(
            t2_ss=t2_ss,
            phase=phases[0],
            c=fp_corrections[0],
            window=windows[0],
            window_p=windows_p[0],
        )

        self.proc_t1_ii(
            phase=phases[1],
            c=fp_corrections[1],
            window=windows[1],
            window_p=windows_p[1],
        )


class LINData3D:
    def __init__(
        self,
        data_dir=".",
        ser_file="ser",
        points=None,
        dim_status=None,
        decim=None,
        dspfvs=None,
        grpdly=None,
    ):  # sourcery no-metrics

        self.ac1 = os.path.join(data_dir, "acqus")
        self.ac2 = os.path.join(data_dir, "acqu2s")
        self.ac3 = os.path.join(data_dir, "acqu3s")
        self.ser = os.path.join(data_dir, "ser")
        self.pp = os.path.join(data_dir, "pulseprogram")
        self.ser = os.path.join(data_dir, ser_file)
        self.dir = data_dir
        self.acq = [0, 0, 0]  # acquisition modes start as undefined

        # dictionary of acquisition modes for Bruker
        self.acqDict = {
            0: "undefined",
            1: "qf",
            2: "qsec",
            3: "tppi",
            4: "states",
            5: "states-tppi",
            6: "echo-antiecho",
        }

        # check if we are a Bruker 2D data set
        if (
            os.path.isfile(self.ac1)
            and os.path.isfile(self.ac2)
            and os.path.isfile(self.ac3)
            and os.path.isfile(self.ser)
            and os.path.isfile(self.pp)
        ):
            self.valid = True

        else:
            self.valid = False
            print("Data Directory does not seem to contain Bruker 3D Data")

        p0 = p1 = p2 = 0  # we'll find these in the files
        dec = dsp = grp = 0  # we'll find these in the files

        with open(self.ac1, "r") as acqusfile:
            for line in acqusfile:
                if "##$TD=" in line:
                    (_, value) = line.split()
                    p0 = int(value)
                if "##$DECIM=" in line:
                    (_, value) = line.split()
                    dec = int(value)
                if "##$DSPFVS=" in line:
                    (_, value) = line.split()
                    dsp = int(value)
                if "##$GRPDLY=" in line:
                    (_, value) = line.split()
                    grp = float(value)

                if "##$BYTORDA=" in line:
                    (_, value) = line.split()
                    self.byte_order = float(value)

                self.acq[0] = 0  # doesnt matter we assume DQD for direct anyway

        with open(self.ac2, "r") as acqusfile:
            for line in acqusfile:
                if "##$TD=" in line:
                    (_, value) = line.split()
                    p1 = int(value)

                if "##$FnMODE=" in line:
                    (_, value) = line.split()
                    self.acq[1] = int(value)

        with open(self.ac3, "r") as acqusfile:
            for line in acqusfile:
                if "##$TD=" in line:
                    (_, value) = line.split()
                    p2 = int(value)

                if "##$FnMODE=" in line:
                    (_, value) = line.split()
                    self.acq[2] = int(value)

        if p0 and p1 and p2:  # we got # points for all three dimensions
            points = [p0, p1, p2]
        else:
            print("problem with detecting number of points in data")
            self.valid = False

        if dec != 0:
            decim = dec
        if dsp != 0:
            dspfvs = dsp
        if grp:
            grpdly = grp
        elif dec != 0 and dsp != 0:
            grpdly = dd2g(dspfvs, decim)
        else:
            print(
                "problem with detecting / determining grpdly - needed for Bruker conversion"
            )
            self.valid = False

        print("Data Points structure is: " + str(points))
        print(
            "DECIM= "
            + str(decim)
            + " DSPFVS= "
            + str(dspfvs)
            + " GRPDLY= "
            + str(grpdly)
        )

        self.dim_status = ["t", "t", "t"] if dim_status is None else dim_status
        if dim_status:
            if len(dim_status) != len(points):
                raise ValueError(
                    "insanity: number of dimensions in 'points' and 'dim_status' don't match"
                )
            for i in range(len(dim_status)):
                if dim_status[i] not in ["t", "f"]:
                    print(dim_status[i])
                    raise ValueError(
                        "dimension domains must be 'f' - frequency or 't' - time"
                    )

        # lets store the points to the class instance
        self.points = points

        # now lets load in the bruker serial file
        with open(self.ser, "rb") as serial_file:
            if self.byte_order == 0:
                self.raw_data = np.frombuffer(serial_file.read(), dtype="<i4")
            elif self.byte_order == 1:
                self.raw_data = np.frombuffer(serial_file.read(), dtype=">i4")

        # now reshape the data
        self.raw_data = np.reshape(self.raw_data, np.asarray(self.points), order="F")

        # TODO - set up some sort of sanity test

        self.converted_data = np.zeros(
            (int(self.points[0] / 2), self.points[1], self.points[2]),
            dtype="complex128",
        )

        # lets convert the data
        if decim and dspfvs:
            if not grpdly:
                grpdly = dd2g(dspfvs, decim)
            self.convert_bruker_3d(grpdly)
        elif grpdly and not decim and not dspfvs:
            self.convert_bruker_3d(grpdly)

        else:
            print(
                "Could not convert from Bruker data, incorrect or not found grpdly, dspfvs and/or decim"
            )

        print("Converted Data Points structure is:", self.points)

        self.phases = (0, 0, 0)
        self.fp_corrections = (0.5, 0.5, 0.5)
        self.windows = ("sb", "sb", "sb")
        self.windows_p = (0.5, 0.5, 0.5)
        self.zero_fill = (1.0, 1.0, 1.0)

        self.processed_data = []  # this will be filled out in proc method
        self.ft_points = []

    def convert_bruker_3d(self, grpdly):

        # edit the number of points in first dimension after Bruker filter removal
        # we now count points in complex numbers as well
        self.points[0] = len(
            remove_bruker_delay(make_complex(self.raw_data[:, 0, 0]), grpdly)
        )
        for ii in range(
            self.points[2]
        ):  # outer loop for third dimension points from dataFID
            for i in range(
                self.points[1]
            ):  # inner loop for second dimension points from dataFID
                fid = remove_bruker_delay(make_complex(self.raw_data[:, i, ii]), grpdly)
                self.converted_data[0 : len(fid), i, ii] = fid

        self.converted_data = self.converted_data[
            0 : self.points[0],
            0 : self.points[1],
            0 : self.points[2],
        ]
        self.raw_data = self.converted_data  # clean up memory a little

        if self.acq[1] == 6:  # Rance Kay Processing needed
            print("Echo-AntiEcho Detected in T2 - dealing with it...")
            for i in range(0, self.points[1], 2):
                for ii in range(self.points[2]):
                    a = self.converted_data[:, i, ii]
                    b = self.converted_data[:, i + 1, ii]
                    c = a + b
                    d = a - b
                    self.converted_data[:, i, ii] = c * np.exp(
                        1.0j * (90 / 180) * np.pi
                    )
                    self.converted_data[:, i + 1, ii] = d * np.exp(
                        1.0j * (180 / 180) * np.pi
                    )

        if self.acq[2] == 6:  # Rance Kay Processing needed
            print("Echo-AntiEcho Detected in T1 - dealing with it...")
            for i in range(0, self.points[2], 2):
                for ii in range(self.points[1]):
                    a = self.converted_data[:, ii, i]
                    b = self.converted_data[:, ii, i + 1]
                    c = a + b
                    d = a - b
                    self.converted_data[:, ii, i] = c * np.exp(
                        1.0j * (90 / 180) * np.pi
                    )
                    self.converted_data[:, ii, i + 1] = d * np.exp(
                        1.0j * (180 / 180) * np.pi
                    )

        self.raw_data = self.converted_data  # clean up memory a little

    def proc_t3(self, phase=0, t3_ss=None, c=1.0, window="sb", window_p=0.5):

        self.processed_data[0, :, :] = self.processed_data[0, :, :] * c
        window = window_function(
            points=self.points[0],
            window=window,
            window_p=window_p,
        )
        for i in range(self.ft_points[2]):
            for ii in range(self.ft_points[1]):
                fid = self.processed_data[0 : self.points[0], ii, i]

                if t3_ss == "butter":
                    fid = butter_highpass_filter(fid, 0.01, 0.1, order=1)

                elif t3_ss == "poly":
                    co_ef = np.polynomial.polynomial.polyfit(
                        np.arange(len(fid)), fid, 5
                    )
                    time_points = np.arange(len(fid))
                    polyline = sum(
                        co_ef[iii] * time_points ** iii for iii in range(len(co_ef))
                    )
                    fid = fid - polyline

                # fid[0:len(window)] = fid[0:len(window)] * window
                fid = fid * window

                self.processed_data[0 : self.points[0], ii, i] = fid
                self.processed_data[:, ii, i] = np.fft.fftshift(
                    np.fft.fft(
                        self.processed_data[:, ii, i]
                        * np.exp(1.0j * (phase / 180) * np.pi)
                    )
                )[::-1]

    def proc_t2(self, phase=0, c=1.0, window="sb", window_p=0.5):

        self.processed_data[:, 0, :] = self.processed_data[:, 0, :] * c
        self.processed_data[:, 1, :] = self.processed_data[:, 1, :] * c

        window = window_function(
            points=self.points[1] / 2,  # hypercomplex so halve this
            window=window,
            window_p=window_p,
        )

        for i in range(self.ft_points[2]):
            for ii in range(self.ft_points[0]):
                fid = np.real(
                    self.processed_data[ii, : int(self.points[1]) : 2, i]
                ) + 1.0j * np.real(
                    self.processed_data[ii, 1 : int(self.points[1]) : 2, i]
                )
                # fid[0:int(self.points[1]/2)] = fid[0:int(self.points[1]/2)] * window
                fid = fid * window
                self.processed_data[ii, 0 : len(fid), i] = fid
                self.processed_data[ii, len(fid) :, i] = np.zeros(
                    self.ft_points[1] - len(fid)
                )

                if self.acq[1] not in [5, 3]:
                    self.processed_data[ii, :, i] = fftshift(
                        fft(
                            self.processed_data[ii, :, i]
                            * np.exp(1.0j * (phase / 180) * np.pi)
                        )
                    )[::-1]

                else:
                    self.processed_data[ii, :, i] = fft(
                        self.processed_data[ii, :, i]
                        * np.exp(1.0j * (phase / 180) * np.pi)
                    )[::-1]

    def proc_t1(self, phase=0, c=1.0, window="sb", window_p=0.5):

        self.processed_data[:, :, 0] = self.processed_data[:, :, 0] * c
        self.processed_data[:, :, 1] = self.processed_data[:, :, 1] * c
        window = window_function(
            points=self.points[2] / 2,  # hypercomplex so halve this
            window=window,
            window_p=window_p,
        )
        for i in range(self.ft_points[1]):
            for ii in range(self.ft_points[0]):
                fid = np.real(
                    self.processed_data[ii, i, : int(self.points[2]) : 2]
                ) + 1.0j * np.real(
                    self.processed_data[ii, i, 1 : int(self.points[2]) : 2]
                )
                # fid[0:int(self.points[2] / 2)] = fid[0:int(self.points[2] / 2)] * window
                fid = fid * window
                self.processed_data[ii, i, 0 : len(fid)] = fid
                self.processed_data[ii, i, len(fid) :] = np.zeros(
                    self.ft_points[2] - len(fid)
                )

                if self.acq[2] in [5, 3]:
                    self.processed_data[ii, i, :] = fft(
                        self.processed_data[ii, i, :]
                        * np.exp(1.0j * (phase / 180) * np.pi)
                    )[::-1]

                else:  # states tppi or tppi - don't fftshift
                    self.processed_data[ii, i, :] = fftshift(
                        fft(
                            self.processed_data[ii, i, :]
                            * np.exp(1.0j * (phase / 180) * np.pi)
                        )
                    )[::-1]

    def proc(
        self,
        phases=(0, 0, 0),
        t3_ss=None,
        fp_corrections=(0.5, 0.5, 0.5),
        windows=("sb", "sb", "sb"),
        windows_p=(0.5, 0.5, 0.5),
        zero_fill=(1.0, 1.0, 1.0),
    ):

        t1_ac_mode = int(self.acq[1])
        if (
            t1_ac_mode >= 3 or t1_ac_mode <= 6
        ):  # hypercomplex data. T2 points is really half
            points_t2 = int(self.points[1] / 2)
        else:
            points_t2 = self.points[1]

        t1_ac_mode = int(self.acq[2])
        if (
            t1_ac_mode >= 3 or t1_ac_mode <= 6
        ):  # hypercomplex data. T1 points is really half
            points_t1 = int(self.points[2] / 2)
        else:
            points_t1 = self.points[2]

        self.ft_points = (
            int(2 ** (next_fourier_number(self.points[0]) + zero_fill[0])),
            int(2 ** (next_fourier_number(points_t2) + zero_fill[1])),
            int(2 ** (next_fourier_number(points_t1) + zero_fill[2])),
        )
        print(self.ft_points)
        self.processed_data = np.zeros(self.ft_points, dtype="complex128")

        self.processed_data[
            0 : self.points[0], 0 : self.points[1], 0 : self.points[2]
        ] = self.converted_data[
            0 : self.points[0], 0 : self.points[1], 0 : self.points[2]
        ]

        print(colorama.Fore.RED + "Processing t3", flush=True)
        self.proc_t3(
            phase=phases[0],
            t3_ss=t3_ss,
            c=fp_corrections[0],
            window=windows[0],
            window_p=windows_p[0],
        )
        print(colorama.Fore.RED + "Processing t2", flush=True)
        self.proc_t2(
            phase=phases[1],
            c=fp_corrections[1],
            window=windows[1],
            window_p=windows_p[1],
        )
        print(colorama.Fore.RED + "Processing t1", flush=True)
        self.proc_t1(
            phase=phases[2],
            c=fp_corrections[2],
            window=windows[2],
            window_p=windows_p[2],
        )


# define class for NUS data - this includes the NUS schedule and assumes Bruker ser file
# the object contains the entire serial file data and the NUS schedule
# points_in_fid is the total (R+I) points in the direct FIDs
class NUSData3D:
    def __init__(
        self,
        data_dir=".",
        ser_file="ser",
        nus_list="nuslist",
        points=None,
        decim=None,
        dspfvs=None,
        grpdly=None,
    ):  # sourcery no-metrics

        self.ac1 = os.path.join(data_dir, "acqus")
        self.ac2 = os.path.join(data_dir, "acqu2s")
        self.ac3 = os.path.join(data_dir, "acqu3s")
        self.ser = os.path.join(data_dir, "ser")
        self.pp = os.path.join(data_dir, "pulseprogram")
        self.ser = os.path.join(data_dir, ser_file)
        self.dir = data_dir
        self.acq = [0, 0, 0]  # acquisition modes start as undefined

        # dictionary of acquisition modes for Bruker
        self.acqDict = {
            0: "undefined",
            1: "qf",
            2: "qsec",
            3: "tppi",
            4: "states",
            5: "states-tppi",
            6: "echo-antiecho",
        }

        # check if we are a Bruker 2D data set
        if (
            os.path.isfile(self.ac1)
            and os.path.isfile(self.ac2)
            and os.path.isfile(self.ac3)
            and os.path.isfile(self.ser)
            and os.path.isfile(self.pp)
        ):
            self.valid = True

        else:
            self.valid = False
            print("Data Directory does not seem to contain Bruker 3D Data")

        p0 = p1 = p2 = 0  # we'll find these in the files
        dec = dsp = grp = 0  # we'll find these in the files

        with open(self.ac1, "r") as acqusfile:
            for line in acqusfile:
                if "##$TD=" in line:
                    (name, value) = line.split()
                    p0 = int(int(value) / 2)
                if "##$DECIM=" in line:
                    (name, value) = line.split()
                    dec = int(value)
                if "##$DSPFVS=" in line:
                    (name, value) = line.split()
                    dsp = int(value)
                if "##$GRPDLY=" in line:
                    (name, value) = line.split()
                    grp = float(value)

        if not decim:
            decim = dec
        if not dspfvs:
            dspfvs = dsp
        if not grpdly:
            grpdly = grp if "grp" in locals() else dd2g(dspfvs, decim)
        points_in_direct_fid = p0 * 2
        print("Number of R+I points: " + str(points_in_direct_fid))
        print(
            "DECIM= "
            + str(decim)
            + " DSPFVS= "
            + str(dspfvs)
            + " GRPDLY= "
            + str(grpdly)
        )

        # we need ot know how many points are in the direct dimension
        self.pointsInDirectFid = points_in_direct_fid

        # lets open and parse the nuslist file - or sched file - of samples taken
        with open(data_dir + "/" + nus_list, "r") as nuslist:
            lines = nuslist.readlines()
            nuslist = []
            for line in lines:
                point = line.split()
                coordinate = [int(coord) for coord in point]
                nuslist.append(coordinate)
        self.nusList = np.array(nuslist)
        self.nusDimensions = len(nuslist[0])
        self.nusPoints = len(nuslist)

        self.orderedNUSlistIndex = []  # place holder

        # we also want some way to know the order of the samples. This generates indexes
        # that give ordered samples from nuslist based on first column being fast dimension
        # self.ordered_nuslist_index = np.lexsort((self.nuslist[:,0], self.nuslist[:,1]))

        # lets load in the actual serial data

        with open(data_dir + "/" + ser_file, "rb") as serial_file:
            self.nusData = np.frombuffer(serial_file.read(), dtype="<i4")

        # bruker data is four bytes per point so
        # len(nus_data) should equal 4 * 2**self.nus_dimensions * self.nus_points * self.points_in_direct_fid
        if (
            4 * 2 ** self.nusDimensions * self.nusPoints * self.pointsInDirectFid
            == len(self.nusData)
        ):
            self.sane = True
        else:
            self.sane = False

        # reshape the data
        self.nusData = np.reshape(
            self.nusData, (self.pointsInDirectFid, 4, self.nusPoints), order="F"
        )
        self.convertedNUSData = []  # placeholder
        # remove bruker filter
        print("remove bruker filter")
        self.convertBruker(grpdly)

    def truncate(self, trunc):
        """
        This function truncates the nusData, nusList and convertedNUSData variables to have only
        'trunc' number of sampled points
        """
        self.nusList = self.nusList[0:trunc]
        self.nusData = self.nusData[:, :, 0:trunc]
        self.convertedNUSData = self.convertedNUSData[:, :, 0:trunc]
        self.nusPoints = len(self.nusList)

    def convertBruker(self, grpdly):
        # edit the number of points in first dimension after Bruker filter removal
        self.points = len(
            remove_bruker_delay(make_complex(np.copy(self.nusData[:, 0, 0])), grpdly)
        )
        # zero fill in a 3D matrix with complex zeros
        self.convertedNUSData = np.zeros(
            (self.points, 4, self.nusPoints), dtype="complex128"
        )
        # load the data
        for ii in range(self.nusPoints):  #
            for i in range(4):
                fid = remove_bruker_delay(
                    make_complex(np.copy(self.nusData[:, i, ii])), grpdly
                )
                self.convertedNUSData[0 : len(fid), i, ii] = fid

    def orderData(self):
        # we want some way to know the order of the samples. This generates indexes
        # that give ordered samples from nuslist based on first column being fast dimension
        self.orderedNUSlistIndex = np.lexsort((self.nusList[:, 0], self.nusList[:, 1]))
        # print(self.nusList[self.ordered_nuslist_index])
        # orderedData = np.zeros( 2**self.nusDimensions * self.nusPoints * self.pointsInDirectFid)
        orderedData = np.zeros(
            (self.pointsInDirectFid, 4, self.nusPoints), dtype="int64"
        )
        orderedConvertedData = np.zeros(
            (self.points, 4, self.nusPoints), dtype="complex128"
        )
        for i, point in enumerate(self.orderedNUSlistIndex):
            # self.nusData[:,i,ii]
            orderedData[:, :, i] = self.nusData[:, :, point]
            orderedConvertedData[:, :, i] = self.convertedNUSData[:, :, point]
        # now set the object attributes to the ordered state
        self.nusData = orderedData
        self.convertedNUSData = orderedConvertedData
        self.nusList = self.nusList[self.orderedNUSlistIndex]

    def writeSer(self, file):
        f = open(file, "wb")
        f.write(self.nusData.flatten(order="F").astype("<i4"))

    def writeNuslist(self, file):
        # f = open(file, 'w')
        # f.write(str(self.nuslist))
        np.savetxt(file, self.nusList, fmt="%i", delimiter="\t")
